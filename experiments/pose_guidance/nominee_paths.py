"""Resolve active managed Promera paths from the NOMINEE site configuration.

Only paths.installation_parent is required. Parse its scalar using Python's
stdlib so the launcher does not require host-side PyYAML.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys

_VARIABLE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def installation_parent(config: Path) -> Path:
    section = ""
    values: list[str] = []
    for line in config.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace():
            section = line.partition(":")[0].strip()
            continue
        match = re.fullmatch(r"\s+installation_parent\s*:\s*(.*?)\s*", line)
        if section == "paths" and match:
            values.append(match.group(1))
    if len(values) != 1:
        raise ValueError("expected exactly one paths.installation_parent in config.yaml")
    raw = values[0]
    if raw.startswith('"'):
        match = re.fullmatch(r'("(?:[^"\\]|\\.)*")\s*(?:#.*)?', raw)
        if not match:
            raise ValueError("invalid double-quoted paths.installation_parent")
        value = json.loads(match.group(1))
    elif raw.startswith("'"):
        match = re.fullmatch(r"'((?:[^']|'')*)'\s*(?:#.*)?", raw)
        if not match:
            raise ValueError("invalid single-quoted paths.installation_parent")
        value = match.group(1).replace("''", "'")
    else:
        value = re.split(r"\s+#", raw, maxsplit=1)[0].strip()
    if not value or value in {"null", "~", "|", ">"}:
        raise ValueError("paths.installation_parent is blank or unsupported")

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        if name not in os.environ:
            raise ValueError(f"undefined environment variable in paths.installation_parent: {name}")
        return os.environ[name]

    value = _VARIABLE.sub(substitute, value)
    if _VARIABLE.search(value):
        raise ValueError("recursive environment variable in paths.installation_parent")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config.parent / path
    path = path.resolve(strict=False)
    if path == Path("/"):
        raise ValueError("paths.installation_parent cannot be /")
    if "\n" in str(path) or "\r" in str(path):
        raise ValueError("paths.installation_parent must not contain newlines")
    return path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 nominee_paths.py /path/to/dev/config.yaml", file=sys.stderr)
        return 2
    config = Path(sys.argv[1]).expanduser().resolve(strict=False)
    try:
        parent = installation_parent(config)
    except (OSError, ValueError) as exc:
        print(f"ERROR reading {config}: {exc}", file=sys.stderr)
        return 2
    root = parent / ".nominee"
    print(root / "images/promera/current.sif")
    print(root / "assets/promera/current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
