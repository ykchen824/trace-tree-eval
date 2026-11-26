"""Harness-controlled baseline context builders for fair compression comparisons."""

from __future__ import annotations

import numpy as np

from trace_tree_eval.backends.base import EmbeddingBackend
from trace_tree_eval.backends.mock import HashingEmbeddingBackend
from trace_tree_eval.schema import RubricItem, TraceStep, Trajectory
from trace_tree_eval.tokens import estimate_tokens


def _format_baseline_doc(
    trajectory: Trajectory,
    steps: list[TraceStep],
    label: str,
) -> str:
    lines = [
        f"## Context Method: {label}",
        f"**Trace ID:** `{trajectory.trace_id}`",
        f"**Task Instruction:**\n{trajectory.task_instruction}",
    ]
    if trajectory.deliverables:
        lines.append("Required Deliverables: " + " | ".join(trajectory.deliverables))
    lines.append("\n## Trajectory Steps")
    for s in steps:
        lines.append(s.render_text())
    lines.append("\n## 4. Final Output & Deliverable Check")
    lines.append(f"Final Output:\n{trajectory.final_output}")
    return "\n".join(lines)


class FullTraceBaseline:
    """Upper-bound reference: feeds the entire uncompressed trajectory."""

    name = "FullTrace"

    def build_context(
        self,
        trajectory: Trajectory,
        rubrics: list[RubricItem] | None = None,
        budget_tokens: int | None = None,
    ) -> str:
        return _format_baseline_doc(trajectory, trajectory.steps, self.name)


class HeadTailBaseline:
    """Naive truncation baseline: keeps head and tail steps within budget."""

    name = "HeadTail"

    def build_context(
        self,
        trajectory: Trajectory,
        rubrics: list[RubricItem] | None = None,
        budget_tokens: int = 8000,
    ) -> str:
        overhead = (
            estimate_tokens(trajectory.task_instruction)
            + estimate_tokens(trajectory.final_output)
            + 80
        )
        avail = max(400, budget_tokens - overhead)
        head_budget = avail // 2
        tail_budget = avail - head_budget

        head: list[TraceStep] = []
        used = 0
        for s in trajectory.steps:
            t = s.token_count()
            if used + t > head_budget and head:
                break
            head.append(s)
            used += t

        head_ids = {s.step_id for s in head}
        tail: list[TraceStep] = []
        used_t = 0
        for s in reversed(trajectory.steps):
            if s.step_id in head_ids:
                continue
            t = s.token_count()
            if used_t + t > tail_budget and tail:
                break
            tail.append(s)
            used_t += t
        tail.reverse()
        return _format_baseline_doc(trajectory, head + tail, self.name)


class LexicalRagBaseline:
    """Flat Top-k cosine similarity retrieval over steps against rubric text."""

    name = "LexicalRAG"

    def __init__(self, embedder: EmbeddingBackend | None = None) -> None:
        self.embedder = embedder or HashingEmbeddingBackend()

    def build_context(
        self,
        trajectory: Trajectory,
        rubrics: list[RubricItem],
        budget_tokens: int = 8000,
    ) -> str:
        if not trajectory.steps:
            return _format_baseline_doc(trajectory, [], self.name)

        overhead = (
            estimate_tokens(trajectory.task_instruction)
            + estimate_tokens(trajectory.final_output)
            + 80
        )
        avail = max(400, budget_tokens - overhead)

        query = " ".join(f"{r.name} {r.question}" for r in rubrics)
        step_texts = [s.render_text(include_header=False) or s.role for s in trajectory.steps]
        q_vec = self.embedder.embed([query])[0]
        s_vecs = self.embedder.embed(step_texts)
        sims = s_vecs @ q_vec

        ranked = np.argsort(-sims)
        chosen: list[TraceStep] = []
        used = 0
        for idx in ranked:
            s = trajectory.steps[int(idx)]
            t = s.token_count()
            if used + t > avail and chosen:
                continue
            chosen.append(s)
            used += t
            if used >= avail:
                break

        chosen.sort(key=lambda s: s.step_id)
        return _format_baseline_doc(trajectory, chosen, self.name)
