"""Compare the original coding agent with the LangGraph harness on a long Python plugin migration."""

import argparse
import hashlib
import json
from json import dumps
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
from contextlib import chdir, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import httpx
import tiktoken
from langchain_openai import ChatOpenAI
from openai import APIError, APIStatusError

from agent import EmptyResponseError, MAX_TURNS, SKILLS_DIR, TOOL_SCHEMAS, build_agent, initial_state
from harness.context import ContextBudget, ContextLimitError, context_overflow
from harness.skills import discover_skills
from harness.tools import Workspace

ROOT = Path(__file__).resolve().parent
SEED_PROJECT = ROOT / "benchmarks" / "plugin_project"
BASELINE = "c19bf52"
TASK = (
    "Migrate all twelve plugins/loader_01.py through plugins/loader_12.py to contract.md. "
    "Read the contract first and preserve all its requirements throughout the task. "
    "For each plugin run 'python check.py N' with N from 1 through 12, inspect the failures, "
    "fix the implementation, and run that check again. Run checks individually. "
    "The diagnostic traces are verbose integration output. "
    "Add unittest regression tests covering the shared contract and run them. "
    "Only edit plugin implementations and add tests. Finish when all twelve checks pass."
)


@dataclass
class Usage:
    requests: int = 0
    responses_with_usage: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int | None = None
    summarizer_requests: int = 0
    main_turns: int = 0
    compactions: int = 0
    context_rejected: bool = False
    last_had_tools: bool = False
    last_had_text: bool = False
    finish_reason: str | None = None
    max_prompt_tokens: int = 0
    skill_loads: int = 0
    skill_tool_calls: int = 0
    local_over_budget: int = 0
    empty_retries: int = 0
    token_scale: float = 1.0

    def record(self, response: httpx.Response) -> None:
        self.requests += 1
        try:
            payload = response.read() and response.json()
        except ValueError:
            return
        if not isinstance(payload, dict):
            return
        if response.is_error:
            error = payload.get("error")
            if error:
                diagnostic = json.dumps(error, ensure_ascii=False).lower()
                self.context_rejected |= any(marker in diagnostic for marker in (
                    "context_length_exceeded", "maximum context length", "context window",
                    "context length exceeded", "prompt is too long", "too many input tokens",
                ))
        usage = payload.get("usage") or {}
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if isinstance(prompt, int) and isinstance(completion, int):
            self.responses_with_usage += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.max_prompt_tokens = max(self.max_prompt_tokens, prompt)
        for choice in payload.get("choices") or []:
            for call in choice.get("message", {}).get("tool_calls") or []:
                name = call.get("function", {}).get("name")
                self.skill_loads += name == "load_skill"
                self.skill_tool_calls += name == "run_tool"
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if isinstance(cached, int):
            self.cached_tokens = (self.cached_tokens or 0) + cached


def grade_project(project: Path, holdout: Path) -> dict[str, bool]:
    checks = {}
    rows = [
        ('[]', []),
        ('["", " "]', []),
        (repr(['{"id": "x"}', 'broken', '{"id": 0}', '{}', '{"id": "x"}']), ["x", 0, "x"]),
    ]
    for number in range(1, 13):
        script = (
            "import importlib; "
            f"f=importlib.import_module('plugins.loader_{number:02d}').decode_rows; "
            + "; ".join(f"assert f({data}) == {expected!r}" for data, expected in rows)
        )
        result = subprocess.run([sys.executable, "-c", script], cwd=project,
                                capture_output=True, text=True, timeout=15)
        checks[f"loader_{number:02d}"] = result.returncode == 0
    return checks


def run_legacy(source: bytes, config, client: httpx.Client, usage: Usage) -> str:
    baseline = {"__name__": "benchmark_baseline"}
    exec(compile(source, f"{BASELINE}:agent.py", "exec"), baseline)

    def post(url, *, json, headers):
        # Keep the committed loop and tool protocol, with a shared generation budget.
        json["temperature"] = config.temperature
        json["max_tokens"] = config.max_output_tokens
        if config.reasoning_effort:
            json["reasoning_effort"] = config.reasoning_effort
        if config.token_count(dumps(json, ensure_ascii=False)) > config.input_budget:
            usage.local_over_budget += 1
        response = client.post(url, json=json, headers=headers)
        if response.is_success:
            usage.main_turns += 1
            choice = response.json()["choices"][0]
            message = choice["message"]
            usage.last_had_tools = bool(message.get("tool_calls"))
            usage.last_had_text = bool((message.get("content") or "").strip())
            usage.finish_reason = choice.get("finish_reason")
        return response

    baseline.update(
        LLM_BASE_URL=config.base_url, LLM_MODEL=config.model, MAX_TURNS=config.max_turns,
        LLM_HEADERS={"Authorization": f"Bearer {config.api_key}"},
        requests=SimpleNamespace(post=post),
    )
    baseline["agent_loop"](config.task)
    return stop_reason(usage, config.max_turns)


def run_langgraph(variant, config, project, client, usage, budget) -> str:
    settings = {
        "base_url": config.base_url, "api_key": config.api_key, "model": config.model,
        "temperature": config.temperature, "http_client": client, "use_responses_api": False,
        "max_retries": 0,  # Each benchmarked request gets one transport attempt.
        "reasoning_effort": config.reasoning_effort,
    }
    model = ChatOpenAI(**settings, extra_body={"max_tokens": config.max_output_tokens}).bind_tools(TOOL_SCHEMAS)
    summarize = None
    if variant == "summary":
        summary_model = ChatOpenAI(**settings, extra_body={"max_tokens": budget.summary_tokens})

        def summarize(messages):
            usage.summarizer_requests += 1
            return summary_model.invoke(messages)

    graph = build_agent(
        model.invoke, Workspace(project), discover_skills(SKILLS_DIR), budget,
        config.max_turns, compaction=variant, summarize=summarize,
    )
    for update in graph.stream(initial_state(config.task), stream_mode="updates"):
        if "compact_context" in update:
            usage.compactions += 1
            print(f"compaction={usage.compactions}; turn={usage.main_turns}", flush=True)
        if "model" in update and (response := update["model"]["response"]) is not None:
            usage.main_turns = update["model"]["turns"]
            usage.empty_retries += bool(update["model"]["empty_retries"])
            usage.token_scale = update["model"]["token_scale"]
            usage.last_had_tools = bool(response.tool_calls or response.invalid_tool_calls)
            usage.last_had_text = bool(response.text.strip())
            usage.finish_reason = response.response_metadata.get("finish_reason")
            print(f"turn={usage.main_turns} {response.text}", flush=True)
            for call in response.tool_calls:
                print(call, flush=True)
        if "tools" in update:
            for observation in update["tools"]["history"][-1][1:]:
                print(f"[{observation.status}] {observation.text[:500]}", flush=True)
    return stop_reason(usage, config.max_turns)


def stop_reason(usage: Usage, max_turns: int) -> str:
    if usage.main_turns == max_turns and usage.last_had_tools:
        return "turn_limit"
    if usage.finish_reason == "length":
        return "output_limit"
    return "finished" if usage.last_had_text else "empty_response"


def execute_run(variant, index, config, source, budget, transport=None) -> dict:
    usage = Usage()
    log_path = config.output / f"{index:02d}-{variant}.log"
    with tempfile.TemporaryDirectory(prefix="jsonl-agent-bench-") as run_dir:
        run_dir = Path(run_dir)
        project = shutil.copytree(SEED_PROJECT, run_dir / "project", ignore=shutil.ignore_patterns("__pycache__"))
        subprocess.run(["git", "init", "-q"], cwd=project, check=True)
        subprocess.run(["git", "add", "."], cwd=project, check=True)
        subprocess.run(
            ["git", "-c", "user.name=Benchmark", "-c", "user.email=bench@example.com",
             "commit", "-qm", "Plugin migration seed"], cwd=project, check=True,
        )
        with httpx.Client(timeout=config.request_timeout, trust_env=False, transport=transport,
                          event_hooks={"response": [usage.record]}) as client:
            error = None
            try:
                with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log), chdir(project):
                    status = (
                        run_legacy(source, config, client, usage) if variant == "legacy"
                        else run_langgraph(variant, config, project, client, usage, budget)
                    )
            except (EmptyResponseError, ContextLimitError, APIError, httpx.HTTPError, ValueError) as problem:
                overflow = usage.context_rejected or isinstance(problem, ContextLimitError)
                if isinstance(problem, APIStatusError):
                    overflow |= context_overflow(problem)
                status = "context_overflow" if overflow else "agent_error"
                error = str(problem)[:700]
            checks = grade_project(project, run_dir / "holdout")
            tests = list(project.glob("**/test_*.py"))
            test_result = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s",
                 "tests" if (project / "tests").is_dir() else ".", "-v"],
                cwd=project, capture_output=True, text=True, timeout=30,
            )
            checks["regression_tests"] = (
                bool(tests) and test_result.returncode == 0
                and "Ran 0 tests" not in test_result.stderr
            )
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[grader tests]\n{test_result.stdout}\n{test_result.stderr}")
            changed = subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"], cwd=project, text=True,
            ).splitlines()
    return {
        "variant": variant, "run": index, "status": status,
        "success": all(checks.values()), "checks": checks,
        "error": error, "changed": changed,
        **vars(usage), "log": log_path.name,
    }


def tabulate(rows: list[dict]) -> str:
    lines = [
        "| Версия | Успешных прогонов | Медиана входных токенов | Медиана выходных токенов | Токенов из кеша (сумма) | Сжатий (сумма) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in dict.fromkeys(row["variant"] for row in rows):
        runs = [row for row in rows if row["variant"] == variant]
        metered = [row for row in runs if row["responses_with_usage"]]
        prompt = statistics.median(row["prompt_tokens"] for row in metered) if metered else "—"
        output = statistics.median(row["completion_tokens"] for row in metered) if metered else "—"
        cached = [row["cached_tokens"] for row in runs if row["cached_tokens"] is not None]
        hits = sum(cached) if cached else "—"
        lines.append(f"| {variant} | {sum(row['success'] for row in runs)}/{len(runs)} | "
                     f"{prompt} | {output} | {hits} | {sum(row['compactions'] for row in runs)} |")
    return "\n".join(lines)


def main(argv=None, transport=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=("legacy", "summary", "trim"), default=["legacy", "summary"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gemma4:12b"))
    parser.add_argument(
        "--history-tokens", type=int, default=os.getenv("LLM_HISTORY_TOKENS") or None,
        help="Optional history cap for reproducing stress tests (default: use the context window)",
    )
    parser.add_argument("--context-window", type=int, default=int(os.getenv("LLM_CONTEXT_WINDOW", "32768")))
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--reasoning-effort", help="Optional shared reasoning setting, e.g. none for Ollama")
    parser.add_argument("--request-timeout", type=float, default=180)
    config = parser.parse_args(argv)
    if not (config.runs > 0 and config.max_turns > 0 and config.request_timeout > 0):
        parser.error("runs, max turns and request timeout must be positive")
    config.output = config.output.resolve()
    config.output.mkdir(parents=True, exist_ok=True)
    config.api_key = os.getenv("LLM_API_KEY", "ollama")
    config.task = TASK
    # Shell tools must find the same interpreter that runs the harness and grader.
    os.environ["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), os.environ["PATH"]])
    source = (
        subprocess.check_output(["git", "show", f"{BASELINE}:agent.py"], cwd=ROOT)
        if "legacy" in config.variants else b""
    )
    encoding = tiktoken.get_encoding(os.getenv("LLM_TOKEN_ENCODING", "cl100k_base"))
    budget = ContextBudget(config.context_window, config.max_output_tokens, config.history_tokens,
                           lambda text: len(encoding.encode_ordinary(text)))
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(config).items() if key != "api_key"}
    settings["python_version"] = platform.python_version()
    settings["baseline_commit"] = BASELINE
    settings["token_encoding"] = os.getenv("LLM_TOKEN_ENCODING", "cl100k_base")
    (config.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    config.token_count = budget.count
    config.input_budget = budget.input_tokens
    rows = []
    with (config.output / "runs.jsonl").open("w", encoding="utf-8") as raw:
        for index in range(1, config.runs + 1):
            # Alternate first position to limit order effects without resetting the server cache.
            order = config.variants if index % 2 else config.variants[::-1]
            for variant in order:
                print(f"Starting {index}/{config.runs} {variant}", flush=True)
                row = execute_run(variant, index, config, source, budget, transport)
                rows.append(row)
                raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw.flush()
                checks = row["checks"]
                print(f"{index}/{config.runs} {variant}: {row['status']}, {sum(checks.values())}/{len(checks)} checks, "
                      f"{row['prompt_tokens']} in / {row['completion_tokens']} out, "
                      f"{row['compactions']} compactions", flush=True)
    table = tabulate(rows)
    (config.output / "table.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
