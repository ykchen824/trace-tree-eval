"""High-level one-line Python API for loading, compressing, and evaluating agent traces."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from trace_tree_eval.adapters import ATIFAdapter, AgentRewardBenchAdapter, DeepResearchTrajAdapter
from trace_tree_eval.backends import (
    DecisionBackend,
    EmbeddingBackend,
    JevBackend,
    MockDecisionBackend,
    MockTextBackend,
    OpenAICompatTextBackend,
    TextModelBackend,
    create_embedding_backend,
)
from trace_tree_eval.compression import CompressionConfig, HybridTreeCompressor
from trace_tree_eval.config import load_dotenv
from trace_tree_eval.evaluation import EvidenceFirstJudge, JudgeConfig
from trace_tree_eval.presets import resolve_rubrics
from trace_tree_eval.schema import EvalDossier, JudgeEvaluation, RubricItem, Trajectory


def load_traces(
    source: str | Path | list[dict[str, Any]] | dict[str, Any],
    dataset_format: str = "auto",
    split: str = "test",
    min_steps: int = 1,
    limit: int | None = None,
) -> list[Trajectory]:
    """Load traces from local JSON/JSONL or Hugging Face datasets with auto-detection."""
    fmt = dataset_format.strip().lower()
    src_str = str(source) if isinstance(source, (str, Path)) else ""

    if fmt == "auto":
        lowered = src_str.lower()
        if "agent-reward-bench" in lowered or "agent_reward_bench" in lowered:
            fmt = "agent-reward-bench"
        elif "deepresearch" in lowered or "openresearcher" in lowered:
            fmt = "deep-research"
        else:
            fmt = "atif"

    if fmt in ("agent-reward-bench", "agent_reward_bench", "arb"):
        return AgentRewardBenchAdapter(split=split, min_steps=min_steps, limit=limit).load(source)
    if fmt in ("deep-research", "deep_research", "dr"):
        return DeepResearchTrajAdapter(split=split, min_steps=min_steps, limit=limit).load(source)

    trajectories = ATIFAdapter().load(source)
    filtered = [t for t in trajectories if len(t.steps) >= min_steps]
    return filtered[:limit] if limit is not None else filtered


def build_runtime_backends(
    live: bool = False,
    cache_dir: Path | str | None = None,
    embedding_provider: str = "hashing",
) -> tuple[DecisionBackend, TextModelBackend, TextModelBackend, EmbeddingBackend]:
    """Build (jev_backend, overview_llm, judge_llm, embedder) from .env or offline mocks."""
    load_dotenv()
    embedder = create_embedding_backend(embedding_provider)
    if not live:
        return (
            MockDecisionBackend(),
            MockTextBackend(price_in_per_1m=0.10, price_out_per_1m=0.40),
            MockTextBackend(price_in_per_1m=3.0, price_out_per_1m=15.0),
            embedder,
        )

    jev = JevBackend(cache_dir=cache_dir)
    overview_llm = OpenAICompatTextBackend(
        model=os.getenv("OVERVIEW_MODEL", "google/gemini-2.5-flash-lite"),
        price_in_per_1m=0.10,
        price_out_per_1m=0.40,
        cache_dir=cache_dir,
    )
    judge_llm = OpenAICompatTextBackend(
        model=os.getenv("JUDGE_MODEL", "anthropic/claude-sonnet-4"),
        price_in_per_1m=3.0,
        price_out_per_1m=15.0,
        cache_dir=cache_dir,
    )
    return jev, overview_llm, judge_llm, embedder


def compress_trace(
    trajectory: Trajectory,
    rubrics: str | Path | list[RubricItem] = "deep_research",
    budget_tokens: int = 6000,
    live: bool = False,
    cache_dir: Path | str | None = None,
    embedding_provider: str = "hashing",
    config: CompressionConfig | None = None,
) -> EvalDossier:
    """One-line helper to compress a Trajectory into an EvalDossier."""
    resolved_rubrics = resolve_rubrics(rubrics)
    jev, overview_llm, _, embedder = build_runtime_backends(
        live=live, cache_dir=cache_dir, embedding_provider=embedding_provider
    )
    cfg = config or CompressionConfig(budget_tokens=budget_tokens)
    compressor = HybridTreeCompressor(
        decision_backend=jev,
        text_backend=overview_llm,
        embedder=embedder,
        config=cfg,
    )
    return compressor.compress(trajectory, resolved_rubrics, budget_tokens=budget_tokens)


def evaluate_trace(
    trajectory: Trajectory,
    rubrics: str | Path | list[RubricItem] = "deep_research",
    budget_tokens: int = 6000,
    live: bool = False,
    cache_dir: Path | str | None = None,
    embedding_provider: str = "hashing",
    compression_config: CompressionConfig | None = None,
    judge_config: JudgeConfig | None = None,
) -> tuple[EvalDossier, JudgeEvaluation]:
    """One-line helper to compress a Trajectory and run Evidence-First Judge."""
    resolved_rubrics = resolve_rubrics(rubrics)
    jev, overview_llm, judge_llm, embedder = build_runtime_backends(
        live=live, cache_dir=cache_dir, embedding_provider=embedding_provider
    )
    cfg = compression_config or CompressionConfig(budget_tokens=budget_tokens)
    compressor = HybridTreeCompressor(
        decision_backend=jev,
        text_backend=overview_llm,
        embedder=embedder,
        config=cfg,
    )
    dossier = compressor.compress(trajectory, resolved_rubrics, budget_tokens=budget_tokens)
    judge = EvidenceFirstJudge(text_backend=judge_llm, config=judge_config)
    evaluation = judge.evaluate_dossier(
        trajectory, dossier, resolved_rubrics, compressor=compressor
    )
    return dossier, evaluation
