import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from openai import AuthenticationError, BadRequestError, InternalServerError

from agent import MAX_API_RETRIES, SKILLS_DIR, TOOL_SCHEMAS, build_agent, cached_tokens, initial_state
from harness.context import ContextBudget, ContextLimitError, assemble_messages, compact_history, trim_history
from harness.skills import ConfigurationError, discover_skills
from harness.tools import Workspace


def count(text):
    return (len(text.encode('utf-8')) + 2) // 3


def action(name, args, call_id='call_1'):
    return AIMessage(content='Checking the project.', tool_calls=[{'name': name, 'args': args, 'id': call_id}])


def turn(text, call_id='call_1'):
    return [action('bash', {'command': 'pwd'}, call_id), ToolMessage(content=text, tool_call_id=call_id)]


def overflow():
    response = httpx.Response(400, request=httpx.Request('POST', 'https://example.com/v1/chat/completions'))
    return BadRequestError('context_length_exceeded', response=response, body={'code': 'context_length_exceeded'})


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'agent.py').write_text('print("hello")\n')
        self.workspace = Workspace(self.root)
        self.skills = discover_skills(SKILLS_DIR)
        self.budget = ContextBudget(8192, 512, 3000, count)

    def graph(self, model, **options):
        return build_agent(model, self.workspace, self.skills, self.budget, **options)

    def test_lazy_load_and_tool_execution(self):
        model = Mock(side_effect=[
            action('load_skill', {'name': 'inspect-project'}),
            action('run_tool', {'name': 'read_file', 'arguments': {'path': 'agent.py'}}),
            AIMessage(content='Read the entry point.'),
        ])
        result = self.graph(model).invoke(initial_state('Inspect this project.'))
        first = model.call_args_list[0].args[0]
        second = model.call_args_list[1].args[0]
        self.assertNotIn('parameters', first[0].text)
        loaded = json.loads(second[-1].text)
        self.assertEqual(loaded['instructions'], self.skills['inspect-project'].instructions())
        self.assertEqual({tool['name'] for tool in loaded['tools']}, {'list_files', 'search_text', 'read_file'})
        self.assertIn('hello', model.call_args_list[2].args[0][-1].text)
        self.assertEqual(result['turns'], 3)
        self.assertEqual([tool['function']['name'] for tool in TOOL_SCHEMAS], ['bash', 'load_skill', 'run_tool'])

    def test_chat_completions_payload_keeps_three_schemas(self):
        requests = []

        def respond(request):
            self.assertEqual(request.url.path, '/v1/chat/completions')
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                message = {
                    'role': 'assistant', 'content': None,
                    'tool_calls': [{
                        'id': 'load', 'type': 'function',
                        'function': {'name': 'load_skill', 'arguments': '{"name":"inspect-project"}'},
                    }],
                }
            else:
                message = {'role': 'assistant', 'content': 'Loaded.'}
            return httpx.Response(200, json={
                'id': 'test', 'object': 'chat.completion', 'model': 'test', 'created': 0,
                'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}],
                'usage': {
                    'prompt_tokens': 1400, 'completion_tokens': 10, 'total_tokens': 1410,
                    'prompt_tokens_details': {'cached_tokens': 1024},
                },
            })

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            model = ChatOpenAI(
                model='test', api_key='test', base_url='https://example.com/v1',
                http_client=client, use_responses_api=False,
            ).bind_tools(TOOL_SCHEMAS)
            result = self.graph(model.invoke).invoke(initial_state('Inspect the project'))
        self.assertEqual(result['response'].text, 'Loaded.')
        self.assertEqual(len(requests), 2)
        for payload in requests:
            self.assertEqual([tool['function']['name'] for tool in payload['tools']],
                             ['bash', 'load_skill', 'run_tool'])
        first, second = requests
        self.assertEqual(first['messages'], second['messages'][:len(first['messages'])])
        self.assertEqual(first['tools'], second['tools'])
        self.assertIn('inspect-project', first['messages'][0]['content'])
        self.assertNotIn('list_files', first['messages'][0]['content'])
        self.assertEqual(second['messages'][-1]['tool_call_id'], 'load')
        self.assertIn('list_files', second['messages'][-1]['content'])
        self.assertEqual(cached_tokens(result['response']), 1024)
        self.assertIsNone(cached_tokens(AIMessage(content='Missing usage')))

    def test_temporary_api_errors_retry_but_authentication_errors_stop(self):
        for status, exception, attempts in (
            (503, InternalServerError, 1 + MAX_API_RETRIES),
            (401, AuthenticationError, 1),
        ):
            with self.subTest(status=status):
                handler = Mock(side_effect=lambda request: httpx.Response(
                    status, headers={'retry-after-ms': '1'}, json={'error': {'message': 'API error'}},
                ))
                with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                    model = ChatOpenAI(
                        model='test', api_key='test', base_url='https://example.com/v1',
                        http_client=client, use_responses_api=False, max_retries=MAX_API_RETRIES,
                    ).bind_tools(TOOL_SCHEMAS)
                    with self.assertRaises(exception):
                        self.graph(model.invoke).invoke(initial_state('Inspect the project'))
                self.assertEqual(handler.call_count, attempts)

    def test_tool_observations_report_nonzero_exit_and_timeout(self):
        model = Mock(side_effect=[
            action('bash', {'command': 'failing-command'}),
            action('bash', {'command': 'slow-command'}),
            AIMessage(content='Done'),
        ])
        with patch.object(self.workspace, 'bash', side_effect=[
            subprocess.CompletedProcess(['failing-command'], 7, '', 'useful error'),
            subprocess.TimeoutExpired(['slow-command'], 120),
        ]) as bash:
            result = self.graph(model).invoke(initial_state('Run checks'))
        failure, timeout = [entry[-1] for entry in result['history']]
        self.assertEqual(failure.status, 'error')
        self.assertIn('Exit code: 7 (failed)', failure.text)
        self.assertIn('useful error', failure.text)
        self.assertEqual(timeout.status, 'error')
        self.assertIn('Inspect the workspace before retrying', timeout.text)
        self.assertEqual(bash.call_count, 2)

    def test_broken_skill_registry_stops_before_model_call(self):
        self.skills['inspect-project'] = replace(self.skills['inspect-project'], tools=('missing',))
        model = Mock()
        with self.assertRaisesRegex(ConfigurationError, 'unregistered tool'):
            self.graph(model)
        model.assert_not_called()

    def test_tools_require_activation_and_validate_arguments(self):
        model = Mock(side_effect=[
            action('run_tool', {'name': 'read_file', 'arguments': {'path': 'agent.py'}}),
            action('load_skill', {'name': 'inspect-project'}),
            action('run_tool', {'name': 'read_file', 'arguments': {'path': 'agent.py', 'start_line': 0}}),
            AIMessage(content='Done'),
        ])
        result = self.graph(model).invoke(initial_state('Read a file'))
        self.assertIn('unavailable', result['history'][0][-1].text)
        self.assertEqual(result['history'][2][-1].status, 'error')

    def test_multiple_tool_calls_have_one_assistant_message(self):
        response = AIMessage(content='', tool_calls=[
            {'name': 'load_skill', 'args': {'name': 'inspect-project'}, 'id': 'load'},
            {'name': 'run_tool', 'args': {'name': 'read_file', 'arguments': {'path': 'agent.py'}}, 'id': 'read'},
        ])
        result = self.graph(Mock(side_effect=[response, AIMessage(content='Done')])).invoke(initial_state('Inspect'))
        self.assertEqual(len(result['history'][0]), 3)
        self.assertEqual([message.tool_call_id for message in result['history'][0][1:]], ['load', 'read'])

    def test_missing_and_malformed_calls_return_observations(self):
        model = Mock(side_effect=[
            action('load_skill', {'name': 'missing'}),
            AIMessage(content='', invalid_tool_calls=[{'name': 'bash', 'args': '{', 'id': 'bad', 'error': 'bad JSON'}]),
            AIMessage(content='Done'),
        ])
        result = self.graph(model).invoke(initial_state('Inspect'))
        self.assertEqual([entry[-1].status for entry in result['history']], ['error', 'error'])

    def test_turn_limit_and_final_answer(self):
        model = Mock(return_value=action('load_skill', {'name': 'inspect-project'}))
        self.assertEqual(self.graph(model, max_turns=1).invoke(initial_state('Inspect'))['turns'], 1)
        model.assert_called_once()
        final = self.graph(lambda _: AIMessage(content='Done')).invoke(initial_state('Say done'))
        self.assertEqual(final['response'].text, 'Done')

    def test_trim_graph_preserves_active_skills(self):
        self.budget = ContextBudget(8192, 512, 650, count)
        model = Mock(side_effect=[action('load_skill', {'name': 'inspect-project'}), AIMessage(content='Done')])
        result = self.graph(model).invoke(initial_state('Inspect'))
        self.assertEqual(result['active_skills'], ['inspect-project'])
        self.assertLessEqual(self.budget.size(assemble_messages([], '', result['history'])), 650)

    def test_summary_graph_and_build_time_selection(self):
        self.budget = ContextBudget(8192, 512, 650, count)
        summarize = Mock(return_value=AIMessage(content='Loaded inspect-project. Reload to see schemas.'))
        model = Mock(side_effect=[action('load_skill', {'name': 'inspect-project'}), AIMessage(content='Done')])
        result = self.graph(model, compaction='summary', summarize=summarize).invoke(initial_state('Inspect'))
        self.assertTrue(result['summary'])
        self.assertIn('Active skills: inspect-project', result['summary'])
        self.assertEqual(model.call_args_list[0].args[0][:2], model.call_args_list[-1].args[0][:2])
        self.assertIn('Progress summary', model.call_args_list[-1].args[0][-1].text)
        self.assertGreater(summarize.call_count, 0)
        with self.assertRaises(ValueError):
            self.graph(model, compaction='summary')
        with self.assertRaises(ValueError):
            self.graph(model, compaction='missing')

    def test_overflow_retries_request_without_replaying_command(self):
        model = Mock(side_effect=[action('bash', {'command': 'echo value >> calls.txt'}), overflow(), AIMessage(content='Done')])
        result = self.graph(model).invoke(initial_state('Run once'))
        self.assertEqual((self.root / 'calls.txt').read_text(), 'value\n')
        self.assertEqual(result['turns'], 2)

    def test_repeated_overflow_stops(self):
        model = Mock(side_effect=[action('bash', {'command': 'echo value'}), overflow(), overflow()])
        with self.assertRaises(ContextLimitError):
            self.graph(model).invoke(initial_state('Run once'))

    def test_history_trim_and_summary_preserve_valid_turns(self):
        prefix = [SystemMessage(content='Rules'), HumanMessage(content='Task')]
        budget = ContextBudget(8192, 512, 650, count)
        history = [turn('old ' * 350, 'old'), turn('recent ' * 100, 'new')]
        recent = trim_history(prefix, history, [], budget)
        self.assertEqual(recent, [history[-1]])
        summarize = Mock(return_value=AIMessage(content='Completed the old step.'))
        recent, summary = compact_history(prefix, history, '', [], summarize, budget)
        self.assertEqual(recent[-1][-1].tool_call_id, 'new')
        self.assertTrue(summary)
        self.assertLessEqual(budget.size(assemble_messages([], summary, recent)), budget.history_tokens)
        self.assertEqual(history[0][-1].text, 'old ' * 350)

    def test_large_observation_keeps_tail_and_call_id(self):
        prefix = [SystemMessage(content='Rules'), HumanMessage(content='Task')]
        history = [turn('start ' * 1000 + 'THE_END')]
        recent = trim_history(prefix, history, [], ContextBudget(4096, 256, 300, count))
        self.assertTrue(recent[-1][-1].text.endswith('THE_END'))
        self.assertEqual(recent[-1][-1].tool_call_id, 'call_1')

    def test_skill_body_is_loaded_only_on_activation(self):
        directory = self.root / 'skills' / 'demo'
        directory.mkdir(parents=True)
        file = directory / 'SKILL.md'
        file.write_text('---\nname: demo\ndescription: Demo skill\n---\nOriginal body')
        skills = discover_skills(directory.parent)
        file.write_text('---\nname: demo\ndescription: Demo skill\n---\nUpdated body')
        self.assertEqual(skills['demo'].instructions(), 'Updated body')

    def test_workspace_tools(self):
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        self.assertIn('agent.py', self.workspace.list_files(pattern='*.py'))
        self.assertIn('agent.py:1', self.workspace.search_text('hello'))
        self.assertIn('1 |', self.workspace.read_file('agent.py'))
        self.assertEqual(self.workspace.run_checks('syntax').returncode, 0)
        patch = 'diff --git a/agent.py b/agent.py\n--- a/agent.py\n+++ b/agent.py\n@@ -1 +1 @@\n-print("hello")\n+print("changed")\n'
        self.assertEqual(self.workspace.apply_patch(patch).returncode, 0)
        self.assertIn('changed', (self.root / 'agent.py').read_text())
        with self.assertRaises(ValueError):
            self.workspace.read_file('../outside.py')


if __name__ == '__main__':
    unittest.main()
