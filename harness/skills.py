from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    tools: tuple[str, ...]

    def instructions(self) -> str:
        try:
            with self.path.open(encoding="utf-8") as stream:
                _read_frontmatter(stream, self.path)
                body = stream.read().strip()
        except (OSError, UnicodeError) as error:
            raise ConfigurationError(f"Cannot read skill instructions from {self.path}: {error}") from error
        if not body:
            raise ConfigurationError(f"{self.path} has no skill instructions")
        return body


def _read_frontmatter(stream, path: Path) -> str:
    if stream.readline().strip() != "---":
        raise ConfigurationError(f"{path} must begin with YAML frontmatter")
    header = []
    for line in stream:
        if line.strip() == "---":
            return "".join(header)
        header.append(line)
    raise ConfigurationError(f"{path} has unclosed YAML frontmatter")


def discover_skills(directory: Path) -> dict[str, Skill]:
    skills = {}
    for path in sorted(directory.glob("*/SKILL.md")):
        try:
            with path.open(encoding="utf-8") as stream:
                header = yaml.safe_load(_read_frontmatter(stream, path))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ConfigurationError(f"Cannot read skill metadata from {path}: {error}") from error
        if not isinstance(header, dict):
            raise ConfigurationError(f"{path} must define skill metadata")
        name = header.get("name")
        description = header.get("description")
        if name != path.parent.name or not isinstance(description, str) or not description.strip():
            raise ConfigurationError(f"{path} must define its directory name and a description")
        metadata = header.get("metadata", {})
        if not isinstance(metadata, dict) or not isinstance(metadata.get("tools", ""), str):
            raise ConfigurationError(f"{path} must list tool names in metadata.tools as a string")
        skills[name] = Skill(name, description.strip(), path, tuple(metadata.get("tools", "").split()))
    return skills
