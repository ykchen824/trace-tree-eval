"""Real embedding backends (SentenceTransformers & OpenAI-compatible) plus factory."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
import numpy as np

from trace_tree_eval.backends.base import EmbeddingBackend
from trace_tree_eval.backends.mock import HashingEmbeddingBackend
from trace_tree_eval.config import load_dotenv

_RETRY_STATUSES = {429, 500, 502, 503, 504}


class SentenceTransformerEmbeddingBackend:
    """Dense local embedding backend backed by sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "SentenceTransformerEmbeddingBackend requires `sentence-transformers`. "
                "Install with: pip install 'trace-tree-eval[embed]'"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


class OpenAICompatEmbeddingBackend:
    """OpenAI/OpenRouter compatible /embeddings HTTP backend with retry."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "text-embedding-3-small",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_base_delay: float = 0.5,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        if not self.api_key:
            raise ValueError("OpenAICompatEmbeddingBackend requires OPENAI_API_KEY or OPENROUTER_API_KEY.")

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        endpoint = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {"model": self.model, "input": texts}

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    res = client.post(endpoint, headers=headers, json=payload)
                    res.raise_for_status()
                    data = res.json()
                rows = sorted(data["data"], key=lambda item: item["index"])
                mat = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1.0)
                return mat / norms
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code not in _RETRY_STATUSES or attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Embedding API HTTP {exc.response.status_code}: {exc.response.text[:200]}"
                    ) from exc
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Embedding API network error: {exc}") from exc
            time.sleep(self.retry_base_delay * (2**attempt))

        raise RuntimeError(f"Embedding API failed: {last_exc}")


def create_embedding_backend(provider: str = "hashing", **kwargs: Any) -> EmbeddingBackend:
    """Create an embedding backend by provider name: hashing | sentence_transformers | openai_compat."""
    key = provider.strip().lower().replace("-", "_")
    if key == "hashing":
        return HashingEmbeddingBackend(**kwargs)
    if key in ("sentence_transformers", "st", "local"):
        return SentenceTransformerEmbeddingBackend(**kwargs)
    if key in ("openai_compat", "openai", "api"):
        return OpenAICompatEmbeddingBackend(**kwargs)
    raise ValueError(
        f"Unknown embedding provider '{provider}'. Choose from: hashing, sentence_transformers, openai_compat."
    )
