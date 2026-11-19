"""CLI and LangGraph workflow for the coding agent."""

import argparse
import json
import os
import sys
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from openai import APIStatusError, AuthenticationError, OpenAIError

from harness.context import (
    ContextBudget,
    ContextLimitError,
    History,
    ModelCall,
    assemble_messages,
    compact_history,
    context_overflow,
    needs_compaction,
    trim_history,
)
from harness.skills import ConfigurationError, Skill, discover_skills
from harness.tools import BUILTIN_TOOLS, Workspace, clip_text, format_command_result

SKILLS_DIR = Path(__file__).resolve().parent / "skills"
MAX_TURNS = 1000
MAX_API_RETRIES = 2
TOOL_SCHEMAS = [
    {"type": "function", "function": tool.definition(name)} for name, tool in BUILTIN_TOOLS.items()
]

SYSTEM_PROMPT = """You are a coding agent working in the user's current project.
Use a relevant skill when it matches the task. Call load_skill with its name to get
its full instructions and tool definitions. Call run_tool with the name and arguments
of a tool enabled by an active skill. You can reload a skill to recover its definitions.
Follow a ReAct cycle: assess the task and observations, choose an action, then use
the returned observation to choose the next action. Use bash as needed. Explain
your actions briefly before tool calls.
If a command times out, inspect the workspace before deciding whether to repeat it.
Use tool results and progress summaries as evidence. End with a plain final answer
when the task is complete or further progress is blocked."""


class AgentState(TypedDict):
    task: str
    history: History
    summary: str
    active_skills: list[str]
    response: AIMessage | None
    turns: int
    compacted_turns: int
    overflow: bool
    retried: bool


def initial_state(task: str) -> AgentState:
    return {
        "task": task,
        "history": [],
        "summary": "",
        "active_skills": [],
        "response": None,
        "turns": 0,
        "compacted_turns": 0,
        "overflow": False,
        "retried": False,
    }


def build_agent(
    call_model: ModelCall,
    workspace: Workspace,
    skills: dict[str, Skill],
    budget: ContextBudget,
    max_turns: int = MAX_TURNS,
    *,
    compaction: Literal["summary", "trim"] = "trim",
    summarize: ModelCall | None = None,
):
    if compaction not in ("summary", "trim"):
        raise ValueError("Compaction must be summary or trim")
    if compaction == "summary" and summarize is None:
        raise ValueError("Summary compaction requires a summarize callable")
    catalog = "\n".join(f"- {skill.name}: {skill.description}" for skill in skills.values())
    if max_turns < 1:
        raise ValueError("The turn limit must be positive")
    for skill in skills.values():
        for name in skill.tools:
            if name not in workspace.tools:
                raise ConfigurationError(f"Skill {skill.name!r} references unregistered tool {name!r}")
    system_prompt = SystemMessage(content=f"{SYSTEM_PROMPT}\n\nAvailable skills:\n{catalog or '(none)'}")

    def prefix(state: AgentState) -> list[BaseMessage]:
        return [system_prompt, HumanMessage(content=state["task"])]

    def messages(state: AgentState) -> list[BaseMessage]:
        return assemble_messages(prefix(state), state["summary"], state["history"])

    def route_context(state: AgentState) -> Literal["compact_context", "model", "__end__"]:
        if state["turns"] >= max_turns:
            return END
        if needs_compaction(prefix(state), state["history"], state["summary"], TOOL_SCHEMAS, budget):
            return "compact_context"
        return "model"

    def model_node(state: AgentState) -> dict:
        try:
            response = call_model(messages(state))
        except APIStatusError as error:
            if not context_overflow(error):
                raise
            if state["retried"]:
                raise ContextLimitError(
                    "The provider still reports context overflow after compaction. "
                    "Check LLM_CONTEXT_WINDOW and LLM_MAX_OUTPUT_TOKENS."
                ) from error
            return {"overflow": True, "response": None}
        return {"response": response, "turns": state["turns"] + 1, "retried": False}

    def route_response(state: AgentState) -> Literal["compact_context", "tools", "__end__"]:
        if state["overflow"]:
            return "compact_context"
        response = state["response"]
        if response and (response.tool_calls or response.invalid_tool_calls):
            return "tools"
        return END

    def compact_node(state: AgentState) -> dict:
        try:
            if compaction == "summary":
                assert summarize is not None
                remaining, summary = compact_history(
                    prefix(state), state["history"], state["summary"], TOOL_SCHEMAS,
                    summarize, budget, force=state["overflow"], active_skills=state["active_skills"],
                )
            else:
                remaining = trim_history(
                    prefix(state), state["history"], TOOL_SCHEMAS,
                    budget, force=state["overflow"],
                )
                summary = ""
        except APIStatusError as error:
            if not context_overflow(error):
                raise
            raise ContextLimitError(
                "The provider rejected the summarizer context. Check LLM_CONTEXT_WINDOW."
            ) from error
        return {
            "history": remaining,
            "summary": summary,
            "overflow": False,
            "retried": state["overflow"],
            "compacted_turns": state["compacted_turns"] + len(state["history"]) - len(remaining),
        }

    def tools_node(state: AgentState) -> dict:
        response = state["response"]
        assert response is not None
        active = list(state["active_skills"])
        turn: list[BaseMessage] = [response]
        for call in response.tool_calls + response.invalid_tool_calls:
            tool_name = call["name"]
            call_id = call["id"]
            if not call_id:
                raise ValueError("The model returned a tool call without an ID")
            try:
                if call.get("type") == "invalid_tool_call":
                    raise ValueError("Malformed arguments. Pass a JSON object.")
                outcome = dispatch(tool_name, call["args"], active)
            except TimeoutExpired as error:
                result = (f"Error: command timed out after {error.timeout} seconds. "
                          "It may have changed files. Inspect the workspace before retrying.")
                status = "error"
            except (ValueError, OSError) as error:
                result = f"Error: {error}"
                status = "error"
            else:
                if isinstance(outcome, CompletedProcess):
                    result = format_command_result(outcome)
                    status = "error" if outcome.returncode else "success"
                else:
                    result = outcome
                    status = "success"
            if tool_name != "load_skill" or status == "error":
                result = clip_text(result, budget.tool_result_bytes)
            turn.append(ToolMessage(
                content=result, name=tool_name, tool_call_id=call_id, status=status,
            ))
        return {"history": [*state["history"], turn], "active_skills": active}

    def dispatch(name: str, supplied: dict, active: list[str]) -> str | CompletedProcess[str]:
        builtin = BUILTIN_TOOLS.get(name)
        if builtin is None:
            raise ValueError(f"Unknown tool {name!r}. Use run_tool for skill tools.")
        arguments = builtin.arguments.model_validate(supplied).model_dump()
        if name == "bash":
            return workspace.bash(**arguments)
        if name == "load_skill":
            skill_name = arguments["name"]
            skill = skills.get(skill_name)
            if skill is None:
                raise ValueError(f"Unknown skill {skill_name!r}. Choose a name from the catalog.")
            definitions = [workspace.tools[name][0].definition(name) for name in skill.tools]
            instructions = skill.instructions()
            if skill_name not in active:
                active.append(skill_name)
            return json.dumps({
                "name": skill_name, "instructions": instructions, "tools": definitions,
            }, ensure_ascii=False)

        tool_name = arguments["name"]
        if not any(tool_name in skills[skill_name].tools for skill_name in active):
            raise ValueError(f"Tool {tool_name!r} is unavailable. Load a skill that exposes it.")
        definition, handler = workspace.tools[tool_name]
        validated = definition.arguments.model_validate(arguments["arguments"])
        return handler(**validated.model_dump())

    graph = StateGraph(AgentState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.add_node("compact_context", compact_node)
    graph.add_conditional_edges(START, route_context)
    graph.add_conditional_edges("model", route_response)
    graph.add_conditional_edges("tools", route_context)
    graph.add_edge("compact_context", "model")
    return graph.compile().with_config({"recursion_limit": 5 * max_turns + 3})


def agent_loop(user_message: str, compaction: Literal["summary", "trim"] = "trim") -> None:
    import tiktoken

    encoding = tiktoken.get_encoding(os.environ.get("LLM_TOKEN_ENCODING", "cl100k_base"))
    budget = ContextBudget(
        int(os.environ.get("LLM_CONTEXT_WINDOW", "32768")),
        int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "4096")),
        int(os.environ.get("LLM_HISTORY_TOKENS", "16000")),
        lambda text: len(encoding.encode(text, disallowed_special=())),
    )
    max_turns = int(os.environ.get("LLM_MAX_TURNS", str(MAX_TURNS)))
    settings = {
        "model": os.environ["LLM_MODEL"],
        "base_url": os.environ.get("LLM_BASE_URL"),
        "api_key": os.environ.get("LLM_API_KEY"),
        "timeout": 120,
        "max_retries": MAX_API_RETRIES,
        "use_responses_api": False,
    }
    model = ChatOpenAI(**settings, max_tokens=budget.output_tokens).bind_tools(TOOL_SCHEMAS)
    summarize = None
    if compaction == "summary":
        summarize = ChatOpenAI(**settings, max_tokens=budget.summary_tokens).invoke
    graph = build_agent(
        model.invoke, Workspace(Path.cwd()), discover_skills(SKILLS_DIR), budget, max_turns,
        compaction=compaction, summarize=summarize,
    )
    turns = 0
    for update in graph.stream(initial_state(user_message), stream_mode="updates"):
        if "model" in update:
            response = update["model"]["response"]
            if response is not None:
                turns = update["model"]["turns"]
                cached = cached_tokens(response)
                if cached is not None:
                    print(f"[cache] cached_tokens={cached}", file=sys.stderr)
                if response.text:
                    print(response.text, flush=True)
        elif "compact_context" in update:
            total = update["compact_context"]["compacted_turns"]
            print(f"[context] {compaction}: {total} completed turns removed so far", file=sys.stderr)
        elif "tools" in update:
            for result in update["tools"]["history"][-1][1:]:
                print(f"[{result.name}] {clip_text(result.text, 500)}", file=sys.stderr)
            if turns == max_turns:
                print(f"Reached the limit of {max_turns} agent turns.", file=sys.stderr)


def cached_tokens(response: AIMessage) -> int | None:
    usage = response.usage_metadata or {}
    count = (usage.get("input_token_details") or {}).get("cache_read")
    if count is not None:
        return count
    raw_usage = response.response_metadata.get("token_usage") or {}
    return (raw_usage.get("prompt_tokens_details") or {}).get("cached_tokens")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="+", help="Task for the coding agent")
    parser.add_argument("--compaction", choices=("trim", "summary"), default="trim")
    args = parser.parse_args()
    task = " ".join(args.task)
    if not task.strip():
        parser.error("provide a task")
    if not os.environ.get("LLM_MODEL"):
        parser.error("set LLM_MODEL to the configured model name")
    try:
        agent_loop(task, compaction=args.compaction)
    except AuthenticationError:
        print("Agent error: provider rejected credentials. Check LLM_API_KEY and LLM_BASE_URL.", file=sys.stderr)
        return 1
    except (OpenAIError, OSError, ValueError, ConfigurationError, ContextLimitError) as error:
        print(f"Agent error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
