"""OpenAI-compatible HTTP text backend for overviews and Evidence-First Judge."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from trace_tree_eval.backends.cache import DiskCache
from trace_tree_eval.config import load_dotenv
from trace_tree_eval.tokens import estimate_tokens

_RETRY_STATUSES = {429, 500, 502, 503, 504}


class OpenAICompatTextBackend:
    """Calls OpenAI or OpenRouter /chat/completions via httpx without extra SDK dependencies."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        price_in_per_1m: float = 3.0,
        price_out_per_1m: float = 15.0,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_base_delay: float = 0.5,
        cache_dir: Path | str | None = None,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        self.model = model or os.getenv("JUDGE_MODEL") or "anthropic/claude-sonnet-4"
        self.price_in_per_1m = price_in_per_1m
        self.price_out_per_1m = price_out_per_1m
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._cache = DiskCache(cache_dir) if cache_dir is not None else None

        if not self.api_key:
            raise ValueError(
                "OpenAICompatTextBackend requires OPENAI_API_KEY (or OPENROUTER_API_KEY). "
                "Use MockTextBackend for offline runs."
            )

        self._lock = threading.Lock()
        self._calls = 0
        self._cache_hits = 0
        self._input_tokens = 0
        self._output_tokens = 0

    def complete(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        cache_key = None
        if self._cache is not None:
            cache_key = DiskCache.make_key("chat_completions", self.model, payload)
            cached = self._cache.get(cache_key)
            if isinstance(cached, dict) and "content" in cached:
                with self._lock:
                    self._cache_hits += 1
                return str(cached["content"])

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    res = client.post(endpoint, headers=headers, json=payload)
                    res.raise_for_status()
                    data = res.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                in_tok = int(usage.get("prompt_tokens") or estimate_tokens(system_prompt + user_prompt))
                out_tok = int(usage.get("completion_tokens") or estimate_tokens(content))
                with self._lock:
                    self._calls += 1
                    self._input_tokens += in_tok
                    self._output_tokens += out_tok
                if self._cache is not None and cache_key is not None:
                    self._cache.set(cache_key, {"content": content})
                return content
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code not in _RETRY_STATUSES or attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Text API HTTP {exc.response.status_code}: {exc.response.text[:200]}"
                    ) from exc
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Text API network error: {exc}") from exc
            time.sleep(self.retry_base_delay * (2**attempt))

        raise RuntimeError(f"Text API failed after retries: {last_exc}")

    def ping(self) -> tuple[bool, str]:
        """Verify live OpenAI-compatible text endpoint connectivity."""
        try:
            reply = self.complete("Reply with one word: OK.", "ping", json_mode=False)
            return True, f"Connected to {self.model} (reply={reply.strip()[:40]!r})"
        except Exception as exc:
            return False, str(exc)

    def usage_stats(self) -> dict[str, Any]:
        with self._lock:
            cost = (
                (self._input_tokens / 1_000_000.0) * self.price_in_per_1m
                + (self._output_tokens / 1_000_000.0) * self.price_out_per_1m
            )
            return {
                "backend": "openai_compat",
                "model": self.model,
                "calls": self._calls,
                "cache_hits": self._cache_hits,
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "estimated_cost_usd": round(cost, 6),
            }
