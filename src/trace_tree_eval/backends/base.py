"""Protocols for decision models (Jev), text models (overview/judge), and embeddings."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class DecisionBackend(Protocol):
    """Non-generative calibrated decision backend returning Noul probabilities."""

    def score_noul(self, state: str, questions: dict[str, str]) -> dict[str, float]:
        """Evaluate multiple boolean questions against one state in parallel."""
        ...

    def batch_score_noul(
        self, items: list[tuple[str, dict[str, str]]]
    ) -> list[dict[str, float]]:
        """Evaluate multiple (state, questions) items concurrently."""
        ...

    def usage_stats(self) -> dict[str, Any]:
        """Return token usage, call count, and estimated USD cost."""
        ...


class TextModelBackend(Protocol):
    """Text generation backend for branch overviews and Evidence-First Judge."""

    def complete(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        """Generate a completion string."""
        ...

    def usage_stats(self) -> dict[str, Any]:
        """Return token usage, call count, and estimated USD cost."""
        ...


class EmbeddingBackend(Protocol):
    """Dense vector embedding backend for semantic topic splitting & retrieval."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized float32 embeddings of shape (len(texts), dim)."""
        ...
