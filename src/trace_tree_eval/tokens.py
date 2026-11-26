"""Lightweight token estimation with optional tiktoken support."""

from __future__ import annotations

import math
import re
from functools import lru_cache

try:
    import tiktoken  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


@lru_cache(maxsize=1)
def _get_encoding():
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover
        return None


def estimate_tokens(text: str) -> int:
    """Estimate token count for mixed CJK/ASCII text."""
    if not text:
        return 0
    enc = _get_encoding()
    if enc is not None:
        return max(1, len(enc.encode(text, disallowed_special=())))

    cjk_count = len(_CJK_RE.findall(text))
    ascii_count = max(0, len(text) - cjk_count)
    # Empirical average: ~1.5 tokens per CJK char, ~0.28 tokens per ASCII char
    approx = math.ceil(cjk_count * 1.5 + ascii_count * 0.28)
    return max(1, approx)
