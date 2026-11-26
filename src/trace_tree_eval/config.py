"""Zero-dependency .env loader and masked environment status helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _strip_quotes(val: str) -> str:
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
        return val[1:-1]
    return val


def find_dotenv(start: Path | str | None = None, max_up: int = 3) -> Path | None:
    """Locate a .env file at `start` or up to `max_up` parent directories."""
    if start is not None:
        p = Path(start)
        if p.is_file():
            return p
        if (p / ".env").is_file():
            return p / ".env"

    cur = Path.cwd().resolve()
    for _ in range(max_up + 1):
        candidate = cur / ".env"
        if candidate.is_file():
            return candidate
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def load_dotenv(path: Path | str | None = None, override: bool = False) -> Path | None:
    """Load environment variables from a .env file without extra dependencies."""
    env_file = Path(path) if (path is not None and Path(path).is_file()) else find_dotenv(path)
    if env_file is None or not env_file.is_file():
        return None

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        val = _strip_quotes(val.strip())
        if override or key not in os.environ or os.environ.get(key, "") == "":
            os.environ[key] = val

    return env_file


def _mask_secret(val: str) -> str:
    if not val:
        return "(unset)"
    if len(val) <= 8:
        return "***"
    return f"{val[:4]}...{val[-4:]}"


def env_summary() -> dict[str, Any]:
    """Return masked environment status for diagnostics without exposing secrets."""
    ts_key = os.getenv("TYPESAFE_API_KEY", "")
    or_key = os.getenv("OPENROUTER_API_KEY", "")
    oa_key = os.getenv("OPENAI_API_KEY", "")
    return {
        "dotenv_found": str(find_dotenv() or "(none)"),
        "TYPESAFE_API_KEY": _mask_secret(ts_key or or_key),
        "TYPESAFE_BASE_URL": os.getenv("TYPESAFE_BASE_URL", "https://openrouter.ai/api"),
        "JEV_MODEL": os.getenv("JEV_MODEL", "typesafe/jev-latest"),
        "OPENAI_API_KEY": _mask_secret(oa_key or or_key),
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        "OVERVIEW_MODEL": os.getenv("OVERVIEW_MODEL", "google/gemini-2.5-flash-lite"),
        "JUDGE_MODEL": os.getenv("JUDGE_MODEL", "anthropic/claude-sonnet-4"),
    }
