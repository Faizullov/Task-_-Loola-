# ReAct-агент для работы с кодом

Агент запускается из командной строки и использует `StateGraph` из LangGraph. Модель выбирает действие, харнесс выполняет вызов инструмента и передаёт результат обратно модели. Цикл завершается, когда модель отвечает без вызовов инструментов или достигнут лимит ходов.

## Ключевые возможности проекта

- **ReAct на LangGraph.** Граф связывает ответы модели, вызовы инструментов и наблюдения. Ответ без вызовов завершает задачу.
- **Ленивая загрузка.** Четыре скилла и шесть инструментов подгружаются по запросу через три постоянные схемы API.
- **Два режима сжатия.** `trim` сохраняет свежую историю в пределах бюджета, `summary` сворачивает старые ходы в сводку. Режим выбирается при сборке агента или через CLI.
- **Стабильный префикс для кеширования.** Активация скилла дописывает сообщения в историю. CLI выводит `cached_tokens`, когда сервер присылает эту метрику.
- **Обработка сбоев.** Модель получает ошибки вызовов и результаты команд. Харнесс ограничивает повторы API, а после переполнения сжимает историю и повторяет запрос к модели.

## Локальный запуск

Понадобятся Python 3.11 или новее, Git, Bash и сервер модели с поддержкой вызовов инструментов через Chat Completions.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

export LLM_BASE_URL="https://example.com/v1"
export LLM_MODEL="your-model"
export LLM_API_KEY="your-api-key"
python agent.py --compaction trim "Найди лимит ходов и проверь синтаксис agent.py"
```

Агент работает в текущем каталоге. Bash запускает команды с правами пользователя, который запустил агента. При первом запуске токенизатор может загрузить словарь.

## Ленивая загрузка скиллов и инструментов

При запуске модель получает название и краткое описание каждого скилла. В поле `tools` каждого запроса передаются три схемы.

| Функция | Что делает |
| --- | --- |
| `bash(command)` | Выполняет команду в рабочем каталоге |
| `load_skill(name)` | Активирует скилл и возвращает его инструкции и схемы инструментов |
| `run_tool(name, arguments)` | Проверяет доступность инструмента, валидирует аргументы и выполняет вызов |

| Скилл | Какие инструменты подключает |
| --- | --- |
| [inspect-project](skills/inspect-project/SKILL.md) | `list_files`, `search_text`, `read_file` |
| [write-tests](skills/write-tests/SKILL.md) | `search_text`, `read_file`, `apply_patch`, `run_checks` |
| [fix-bug](skills/fix-bug/SKILL.md) | `search_text`, `read_file`, `apply_patch`, `run_checks`, `git_diff` |
| [verify-changes](skills/verify-changes/SKILL.md) | `git_diff`, `run_checks` |

Чтобы добавить скилл, создай `skills/<name>/SKILL.md` с полями `name` и `description`. В `metadata.tools` перечисли имена инструментов через пробел. [Реестр инструментов](harness/tools.py)

## Сжатие контекста

Режим задаётся при сборке графа аргументом `compaction`. В командной строке доступны `--compaction trim` и `--compaction summary`.

| Режим | Как работает |
| --- | --- |
| `trim` (по умолчанию) | Удаляет старые завершённые ходы до нужного лимита. Если последний вывод сам превышает бюджет, сохраняет его конец, который помещается в лимит. |
| `summary` | Отправляет старые ходы на суммаризацию отдельным запросом к модели. Обновляет сводку и сохраняет свежие ходы целиком, пока они помещаются в бюджет. |

```python
import os
from pathlib import Path

import tiktoken
from langchain_openai import ChatOpenAI

from agent import MAX_API_RETRIES, SKILLS_DIR, TOOL_SCHEMAS, build_agent, initial_state
from harness.context import ContextBudget
from harness.skills import discover_skills
from harness.tools import Workspace

encoding = tiktoken.get_encoding("cl100k_base")
budget = ContextBudget(32_768, 4_096, 16_000, lambda text: len(encoding.encode_ordinary(text)))
workspace = Workspace(Path.cwd())
skills = discover_skills(SKILLS_DIR)

settings = {
    "model": os.environ["LLM_MODEL"],
    "base_url": os.environ["LLM_BASE_URL"],
    "api_key": os.environ["LLM_API_KEY"],
    "use_responses_api": False,
    "max_retries": MAX_API_RETRIES,
}
chat_model = ChatOpenAI(**settings, max_tokens=budget.output_tokens)
summary_model = ChatOpenAI(**settings, max_tokens=budget.summary_tokens)
call_model = chat_model.bind_tools(TOOL_SCHEMAS).invoke
trim_graph = build_agent(call_model, workspace, skills, budget, compaction="trim")
summary_graph = build_agent(
    call_model, workspace, skills, budget,
    compaction="summary", summarize=summary_model.invoke,
)
result = summary_graph.invoke(initial_state("Изучи структуру проекта"))
print(result["response"].text)
```

Проверка бюджета учитывает историю, системный промпт, исходную задачу и схемы инструментов. Харнесс резервирует токены для ответа. В обоих режимах он хранит исходную задачу и список активных скиллов, а вызовы инструментов обрабатывает вместе с их результатами. При ошибке переполнения харнесс сильнее сжимает историю и один раз повторяет запрос к модели. Уже выполненные команды при этом не запускаются повторно. [Реализация сжатия](harness/context.py)

| Переменная окружения | Значение по умолчанию |
| --- | --- |
| `LLM_CONTEXT_WINDOW` | `32768` токенов на вход и ответ вместе |
| `LLM_MAX_OUTPUT_TOKENS` | `4096` токенов в резерве на ответ |
| `LLM_HISTORY_TOKENS` | `16000` токенов на историю и сводку |
| `LLM_TOKEN_ENCODING` | `cl100k_base` для локальной оценки |
| `LLM_MAX_TURNS` | `1000` ходов модели |

Локальная оценка токенов может отличаться от подсчёта сервера. Размер окна и токенизатор задаются под используемую модель.

## Кеширование промпта

В начале каждого запроса идут фиксированный системный промпт с кратким каталогом скиллов и исходная задача. Три схемы инструментов передаются в том же порядке. Полные инструкции и схемы инструментов скилла добавляются в конец истории как результат `load_skill`. Харнесс хранит `active_skills` в состоянии графа для проверки доступа и добавляет список в сводку при сжатии в режиме `summary`. [Сборка префикса](agent.py#L98)

В режиме `summary` история меняется с места вставки сводки. Системный промпт, исходная задача и схемы остаются пригодны для повторного использования сервером. Между сжатиями запросы снова растут добавлением новых сообщений в конец. CLI выводит `[cache] cached_tokens=…` в `stderr`, когда сервер отдаёт число токенов из кеша. [OpenAI Prompt Caching](https://developers.openai.com/api/docs/guides/prompt-caching) · [vLLM Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)

## Обработка ошибок

| Ситуация | Как обрабатывается |
| --- | --- |
| Битый JSON, неверный тип, неизвестный инструмент или скилл ещё не активирован | Харнесс возвращает короткий `ToolMessage` со статусом `error`. Модель может исправить вызов. |
| Тест упал, Git отклонил патч, команда вернула ненулевой код | Харнесс помечает `ToolMessage` как `error` и передаёт модели код выхода и фрагмент вывода. |
| Таймаут команды | Харнесс прерывает ожидание через 120 секунд. В сообщении модели указано, что команда могла изменить файлы и перед повтором нужно проверить рабочий каталог. |
| Переполнение контекста | Харнесс сжимает историю и один раз повторяет запрос к модели. |
| Временный сбой API | Клиент может повторить запрос до двух раз. Повтор касается только запроса к модели. |
| Ошибка авторизации или некорректная конфигурация скиллов | Запуск завершается с диагностикой. Ссылки на незарегистрированные инструменты проверяются при сборке графа. |

## Тесты

Тесты с имитацией ответов модели проверяют активацию скиллов, неизменность префикса, `cached_tokens`, ошибки инструментов, оба режима сжатия и повторы API.

```sh
python -m unittest discover -s tests -v
```
