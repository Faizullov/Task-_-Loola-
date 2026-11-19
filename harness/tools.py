import fnmatch
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class BashArgs(Arguments):
    command: str


class LoadSkillArgs(Arguments):
    name: str


class RunToolArgs(Arguments):
    name: str
    arguments: dict[str, object]


class PathArgs(Arguments):
    path: str = "."


class ListFilesArgs(PathArgs):
    pattern: str = "*"


class SearchTextArgs(PathArgs):
    query: str = Field(min_length=1)


class ReadFileArgs(Arguments):
    path: str
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class ApplyPatchArgs(Arguments):
    patch: str = Field(min_length=1)


class RunChecksArgs(Arguments):
    check: Literal["syntax", "tests"]
    target: str | None = None


@dataclass(frozen=True)
class Tool:
    description: str
    arguments: type[Arguments]

    def definition(self, name: str) -> dict:
        return {
            "name": name,
            "description": self.description,
            "parameters": self.arguments.model_json_schema(),
        }


BUILTIN_TOOLS = {
    "bash": Tool("Run a shell command in the project and return its exit code and output.", BashArgs),
    "load_skill": Tool("Activate a skill by name and get its instructions and tool definitions.", LoadSkillArgs),
    "run_tool": Tool("Run a tool from an active skill using the arguments its definition describes.", RunToolArgs),
}

MAX_RESULTS = 100
MAX_READ_LINES = 200


def format_command_result(result: subprocess.CompletedProcess[str]) -> str:
    status = " (failed)" if result.returncode else ""
    stderr = f"\nSTDERR:\n{result.stderr}" if result.stderr else ""
    return f"Exit code: {result.returncode}{status}\n{result.stdout}{stderr}"


def clip_text(text: str, max_bytes: int) -> str:
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    marker = b"\n...[output shortened]...\n"
    half = max(0, (max_bytes - len(marker)) // 2)
    if not half:
        return marker[:max_bytes].decode()
    return (data[:half].decode(errors="ignore") + marker.decode()
            + data[-half:].decode(errors="ignore"))


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.tools: dict[str, tuple[Tool, Callable[..., str | subprocess.CompletedProcess[str]]]] = {
            "list_files": (
                Tool("List project files under a path, optionally filtered by a glob.", ListFilesArgs),
                self.list_files,
            ),
            "search_text": (
                Tool("Find literal text and return matching paths, line numbers, and text.", SearchTextArgs),
                self.search_text,
            ),
            "read_file": (
                Tool("Read UTF-8 text with 1-based inclusive line numbers, up to 200 lines.", ReadFileArgs),
                self.read_file,
            ),
            "apply_patch": (
                Tool("Apply a unified Git diff to files in the workspace.", ApplyPatchArgs),
                self.apply_patch,
            ),
            "run_checks": (
                Tool("Run syntax for a Python path or tests for a dotted unittest target.", RunChecksArgs),
                self.run_checks,
            ),
            "git_diff": (
                Tool("Show staged and unstaged changes to tracked Git files, optionally for one path.", PathArgs),
                self.git_diff,
            ),
        }

    def _path(self, path: str) -> Path:
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root) or ".git" in resolved.relative_to(self.root).parts:
            raise ValueError("Path must resolve inside the project outside Git metadata")
        return resolved

    def _run(self, command: list[str], input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command, cwd=self.root, input=input_text, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=120,
        )

    def bash(self, command: str) -> subprocess.CompletedProcess[str]:
        return self._run(["bash", "-c", command])

    def _files(self, path: str):
        directory = self._path(path)
        if not directory.exists():
            raise ValueError(f"Path does not exist: {path}")
        if directory.is_file():
            yield directory
            return
        result = self._run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"])
        candidates = (
            (self.root / name for name in dict.fromkeys(result.stdout.split("\0")) if name)
            if result.returncode == 0 else directory.rglob("*")
        )
        for candidate in candidates:
            if candidate.is_symlink() or not candidate.is_file():
                continue
            relative = candidate.relative_to(self.root)
            if any(part in {".git", ".venv", "__pycache__"} for part in relative.parts):
                continue
            if candidate.is_relative_to(directory) and candidate.resolve().is_relative_to(self.root):
                yield candidate

    def list_files(self, path: str = ".", pattern: str = "*") -> str:
        matches = []
        for file in self._files(path):
            name = file.relative_to(self.root).as_posix()
            if fnmatch.fnmatchcase(name, pattern) or fnmatch.fnmatchcase(name, pattern.removeprefix("**/")):
                matches.append(name)
                if len(matches) > MAX_RESULTS:
                    break
        return self._format_matches(matches)

    def search_text(self, query: str, path: str = ".") -> str:
        matches = []
        for file in self._files(path):
            try:
                with file.open(encoding="utf-8") as stream:
                    name = file.relative_to(self.root).as_posix()
                    for number, line in enumerate(stream, 1):
                        if query in line:
                            matches.append(f"{name}:{number}: {clip_text(line.rstrip(), 500)}")
                            if len(matches) > MAX_RESULTS:
                                return self._format_matches(matches)
            except (OSError, UnicodeError):
                continue
        return self._format_matches(matches)

    @staticmethod
    def _format_matches(matches: list[str]) -> str:
        if len(matches) > MAX_RESULTS:
            return "\n".join([*matches[:MAX_RESULTS], "More results available. Narrow the query."])
        return "\n".join(matches) or "No matches."

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        last_line = end_line if end_line is not None else start_line + 99
        if not start_line <= last_line < start_line + MAX_READ_LINES:
            raise ValueError("Specify an inclusive range of up to 200 lines")
        with self._path(path).open(encoding="utf-8") as stream:
            selected = islice(stream, start_line - 1, last_line)
            result = [f"{number:>5} | {line.rstrip(chr(10))}"
                      for number, line in enumerate(selected, start_line)]
        return "\n".join(result) or "No lines in the requested range."

    def apply_patch(self, patch: str) -> subprocess.CompletedProcess[str]:
        return self._run(["git", "apply", "--verbose", "-"], patch.rstrip("\n") + "\n")

    def run_checks(self, check: str, target: str | None = None) -> subprocess.CompletedProcess[str]:
        if check == "syntax":
            path = self._path(target or "agent.py")
            if not path.is_file() or path.suffix != ".py":
                raise ValueError("Syntax target must be an existing Python file")
            script = (
                "import sys; from pathlib import Path; "
                "p = Path(sys.argv[1]); compile(p.read_bytes(), str(p), 'exec'); print('Python syntax OK')"
            )
            command = [sys.executable, "-c", script, str(path)]
        elif target:
            if not re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", target):
                raise ValueError("Test target must be a dotted unittest name")
            command = [sys.executable, "-m", "unittest", "-v", target]
        else:
            start = "tests" if (self.root / "tests").is_dir() else "."
            command = [sys.executable, "-m", "unittest", "discover", "-s", start, "-v"]
        return self._run(command)

    def git_diff(self, path: str = ".") -> subprocess.CompletedProcess[str]:
        relative = self._path(path).relative_to(self.root)
        return self._run(["git", "diff", "--no-ext-diff", "HEAD", "--", str(relative)])
