"""TypeSafe AI Jev backend with safe-window guard and concurrent batching."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from trace_tree_eval.backends.cache import DiskCache
from trace_tree_eval.config import load_dotenv
from trace_tree_eval.tokens import estimate_tokens

try:
    from typesafe_sdk import Noul, TypeSafeClient  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    TypeSafeClient = None
    Noul = None

# Jev pricing: $0.042 per 1M input tokens, output tokens $0.00
JEV_PRICE_PER_1M_INPUT_USD = 0.042
_RETRY_STATUSES = {429, 500, 502, 503, 504}


class JevBackend:
    """Wrapper around TypeSafe AI Jev (supports official endpoint or OpenRouter gateway)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_state_tokens: int = 18000,
        max_workers: int = 8,
        timeout_seconds: float = 30.0,
        allow_danger_zone_long_context: bool = False,
        max_retries: int = 3,
        retry_base_delay: float = 0.5,
        cache_dir: Path | str | None = None,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = (base_url or os.getenv("TYPESAFE_BASE_URL") or "https://openrouter.ai/api").rstrip("/")
        self.model = model or os.getenv("JEV_MODEL") or "typesafe/jev-latest"
        self.max_state_tokens = max_state_tokens
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.allow_danger_zone_long_context = allow_danger_zone_long_context
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._cache = DiskCache(cache_dir) if cache_dir is not None else None

        if not self.api_key:
            raise ValueError(
                "JevBackend requires TYPESAFE_API_KEY (or OPENROUTER_API_KEY). "
                "Use MockDecisionBackend for offline runs."
            )

        self._lock = threading.Lock()
        self._calls = 0
        self._input_tokens = 0
        self._cache_hits = 0
        self._sdk_client = None
        if TypeSafeClient is not None:
            self._sdk_client = TypeSafeClient(api_key=self.api_key, base_url=self.base_url)

    def score_noul(self, state: str, questions: dict[str, str]) -> dict[str, float]:
        if not questions:
            return {}

        chunks = self._split_safe_chunks(state)
        if len(chunks) == 1:
            return self._call_single(chunks[0], questions)

        # Conservative max-pooling across safe-window sub-chunks avoids Danger-Zone degradation
        merged = {qid: 0.0 for qid in questions}
        for chunk in chunks:
            probs = self._call_single(chunk, questions)
            for qid, val in probs.items():
                if val > merged[qid]:
                    merged[qid] = val
        return merged

    def batch_score_noul(
        self, items: list[tuple[str, dict[str, str]]]
    ) -> list[dict[str, float]]:
        if not items:
            return []
        if len(items) == 1 or self.max_workers <= 1:
            return [self.score_noul(state, q) for state, q in items]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(items))) as pool:
            return list(pool.map(lambda item: self.score_noul(item[0], item[1]), items))

    def usage_stats(self) -> dict[str, Any]:
        with self._lock:
            cost = (self._input_tokens / 1_000_000.0) * JEV_PRICE_PER_1M_INPUT_USD
            return {
                "backend": "jev",
                "model": self.model,
                "calls": self._calls,
                "cache_hits": self._cache_hits,
                "input_tokens": self._input_tokens,
                "output_tokens": 0,
                "estimated_cost_usd": round(cost, 6),
            }

    def ping(self) -> tuple[bool, str]:
        """Verify live Jev endpoint connectivity with a minimal Noul decision probe."""
        try:
            probs = self.score_noul("System connectivity test.", {"ok": "Is this a test?"})
            return True, f"Connected to {self.model} (prob={probs.get('ok', 0.0):.2f})"
        except Exception as exc:
            return False, str(exc)

    def _split_safe_chunks(self, state: str) -> list[str]:
        if estimate_tokens(state) <= self.max_state_tokens:
            return [state]
        lines = state.splitlines(keepends=True)
        chunks: list[str] = []
        buf: list[str] = []
        buf_tokens = 0
        for line in lines:
            lt = estimate_tokens(line)
            if buf and buf_tokens + lt > self.max_state_tokens:
                chunks.append("".join(buf))
                overlap = buf[-max(1, len(buf) // 10) :]
                buf = list(overlap)
                buf_tokens = estimate_tokens("".join(buf))
            buf.append(line)
            buf_tokens += lt
        if buf:
            chunks.append("".join(buf))
        return chunks or [state]

    def _call_single(self, state: str, questions: dict[str, str]) -> dict[str, float]:
        qids = list(questions.keys())
        cache_key = None
        if self._cache is not None:
            cache_key = DiskCache.make_key(
                "jev_noul",
                self.model,
                {"state": state, "questions": questions},
            )
            cached = self._cache.get(cache_key)
            if isinstance(cached, dict):
                with self._lock:
                    self._cache_hits += 1
                return {qid: float(cached.get(qid, 0.5)) for qid in qids}

        tokens = estimate_tokens(state) + sum(estimate_tokens(q) for q in questions.values())
        endpoint = f"{self.base_url}/v1/systemone"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.allow_danger_zone_long_context:
            headers["TypeSafe-Danger-Zone-Long-Context"] = "true"
        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                qid: {"type": "noul", "instructions": instr}
                for qid, instr in questions.items()
            },
        }

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if self._sdk_client is not None and Noul is not None:
                    q_objs = {qid: Noul(instructions=instr) for qid, instr in questions.items()}
                    resp = self._sdk_client.system_one(model=self.model, state=state, questions=q_objs)
                    probs = self._extract_probs_from_response(resp, qids)
                else:
                    with httpx.Client(timeout=self.timeout_seconds) as client:
                        res = client.post(endpoint, headers=headers, json=payload)
                        res.raise_for_status()
                        data = res.json()
                    probs = self._extract_probs_from_response(data, qids)

                with self._lock:
                    self._calls += 1
                    self._input_tokens += tokens
                if self._cache is not None and cache_key is not None:
                    self._cache.set(cache_key, probs)
                return probs
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code not in _RETRY_STATUSES or attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Jev API HTTP {exc.response.status_code}: {exc.response.text[:200]}"
                    ) from exc
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Jev API network error: {exc}") from exc
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Jev API error: {exc}") from exc
            time.sleep(self.retry_base_delay * (2**attempt))

        raise RuntimeError(f"Jev API failed after retries: {last_exc}")

    @staticmethod
    def _extract_probs_from_response(resp: Any, qids: list[str]) -> dict[str, float]:
        # Handle both SDK object attributes and raw HTTP JSON dicts
        raw = resp
        if hasattr(resp, "model_dump"):
            raw = resp.model_dump()
        elif hasattr(resp, "answers"):
            raw = {"answers": getattr(resp, "answers")}
        elif hasattr(resp, "results"):
            raw = {"results": getattr(resp, "results")}

        if not isinstance(raw, dict):
            return {qid: 0.5 for qid in qids}

        container = raw.get("answers") or raw.get("results") or raw.get("decisions") or raw
        out: dict[str, float] = {}
        for qid in qids:
            entry = container.get(qid) if isinstance(container, dict) else None
            prob = 0.5
            if isinstance(entry, (int, float)):
                prob = float(entry)
            elif isinstance(entry, dict):
                if "probability" in entry:
                    prob = float(entry["probability"])
                elif "prob" in entry:
                    prob = float(entry["prob"])
                elif "probabilities" in entry and isinstance(entry["probabilities"], dict):
                    p_dict = entry["probabilities"]
                    prob = float(p_dict.get("true", p_dict.get("yes", p_dict.get("True", 0.5))))
                elif "score" in entry:
                    prob = float(entry["score"])
            out[qid] = max(0.0, min(1.0, prob))
        return out
