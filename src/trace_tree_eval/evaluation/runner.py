"""Batch evaluation runner with method comparison and failure-mode clustering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from trace_tree_eval.backends.base import DecisionBackend, EmbeddingBackend, TextModelBackend
from trace_tree_eval.backends.mock import (
    HashingEmbeddingBackend,
    MockDecisionBackend,
    MockTextBackend,
)
from trace_tree_eval.compression import CompressionConfig, HybridTreeCompressor
from trace_tree_eval.evaluation.baselines import (
    FullTraceBaseline,
    HeadTailBaseline,
    LexicalRagBaseline,
)
from trace_tree_eval.evaluation.judge import EvidenceFirstJudge, JudgeConfig
from trace_tree_eval.evaluation.metrics import compute_sufficiency_report
from trace_tree_eval.schema import EvalDossier, JudgeEvaluation, RubricItem, Trajectory


@dataclass
class MethodAggregate:
    """Aggregated batch statistics for a single context/compression method."""

    method_name: str
    num_traces: int
    mean_tokens: float
    mean_savings_pct: float
    mean_agreement: float
    mean_f1: float
    mean_false_pass_rate: float
    mean_false_fail_rate: float
    mean_js_vs_full: float
    escalation_rate: float
    total_judge_cost_usd: float
    total_compressor_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchReport:
    """Full batch evaluation report including failure-mode clustering."""

    num_traces: int
    budget_tokens: int
    method_summaries: dict[str, MethodAggregate] = field(default_factory=dict)
    failure_mode_counts: dict[str, int] = field(default_factory=dict)
    failure_mode_rates: dict[str, float] = field(default_factory=dict)
    rubric_names: dict[str, str] = field(default_factory=dict)
    per_trace: list[dict[str, Any]] = field(default_factory=list)
    compressor_usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_traces": self.num_traces,
            "budget_tokens": self.budget_tokens,
            "method_summaries": {k: v.to_dict() for k, v in self.method_summaries.items()},
            "failure_mode_counts": dict(self.failure_mode_counts),
            "failure_mode_rates": dict(self.failure_mode_rates),
            "rubric_names": dict(self.rubric_names),
            "per_trace": list(self.per_trace),
            "compressor_usage": dict(self.compressor_usage),
        }


def _extract_reference_verdicts(
    traj: Trajectory,
    rubrics: list[RubricItem],
    full_eval: JudgeEvaluation | None,
) -> dict[str, bool]:
    gt = traj.ground_truth or {}
    if gt and all(r.id in gt for r in rubrics):
        return {r.id: bool(gt[r.id]) for r in rubrics}
    if full_eval is not None:
        return full_eval.verdicts_dict()
    return {r.id: True for r in rubrics}


class BatchEvaluator:
    """Runs controlled-harness batch evaluation across trajectories and baselines."""

    def __init__(
        self,
        decision_backend: DecisionBackend | None = None,
        overview_backend: TextModelBackend | None = None,
        judge_backend: TextModelBackend | None = None,
        embedder: EmbeddingBackend | None = None,
        compression_config: CompressionConfig | None = None,
        judge_config: JudgeConfig | None = None,
    ) -> None:
        self.decision_backend = decision_backend or MockDecisionBackend()
        self.overview_backend = overview_backend or MockTextBackend(price_in_per_1m=0.10, price_out_per_1m=0.40)
        self.judge_backend = judge_backend or MockTextBackend(price_in_per_1m=3.0, price_out_per_1m=15.0)
        self.embedder = embedder or HashingEmbeddingBackend()
        self.compression_config = compression_config or CompressionConfig()
        self.judge_config = judge_config or JudgeConfig()

    def run(
        self,
        trajectories: list[Trajectory],
        rubrics: list[RubricItem],
        budget_tokens: int | None = None,
        include_baselines: bool = True,
        methods: list[str] | None = None,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> BatchReport:
        target_budget = budget_tokens or self.compression_config.budget_tokens
        if methods is None:
            active_methods = (
                ["FullTrace", "HeadTail", "LexicalRAG", "HybridTree"]
                if include_baselines
                else ["HybridTree"]
            )
        else:
            active_methods = list(methods)

        compressor = HybridTreeCompressor(
            decision_backend=self.decision_backend,
            text_backend=self.overview_backend,
            embedder=self.embedder,
            config=CompressionConfig(
                budget_tokens=target_budget,
                branch_prune_threshold=self.compression_config.branch_prune_threshold,
                critical_lambda=self.compression_config.critical_lambda,
                causal_trigger_threshold=self.compression_config.causal_trigger_threshold,
                causal_confirm_threshold=self.compression_config.causal_confirm_threshold,
                dedup_jaccard_threshold=self.compression_config.dedup_jaccard_threshold,
                min_loop_len=self.compression_config.min_loop_len,
                tree_build=self.compression_config.tree_build,
            ),
        )
        judge = EvidenceFirstJudge(text_backend=self.judge_backend, config=self.judge_config)
        full_b = FullTraceBaseline()
        head_b = HeadTailBaseline()
        rag_b = LexicalRagBaseline(embedder=self.embedder)

        # Accumulators per method
        acc: dict[str, list[dict[str, float]]] = {m: [] for m in active_methods}
        failure_counts: dict[str, int] = {r.id: 0 for r in rubrics}
        per_trace_rows: list[dict[str, Any]] = []

        before_jev_cost = float(self.decision_backend.usage_stats().get("estimated_cost_usd", 0.0))
        before_ov_cost = float(self.overview_backend.usage_stats().get("estimated_cost_usd", 0.0))

        total = len(trajectories)
        for idx, traj in enumerate(trajectories, start=1):
            if on_progress is not None:
                on_progress(idx, total, traj.trace_id)

            raw_tokens = max(1, traj.total_tokens())
            evals: dict[str, JudgeEvaluation] = {}
            tokens_used: dict[str, int] = {}
            dossier: EvalDossier | None = None

            if "FullTrace" in active_methods:
                ctx = full_b.build_context(traj)
                evals["FullTrace"] = judge.evaluate_context(traj.trace_id, ctx, rubrics, "FullTrace")
                tokens_used["FullTrace"] = raw_tokens

            if "HeadTail" in active_methods:
                ctx = head_b.build_context(traj, budget_tokens=target_budget)
                evals["HeadTail"] = judge.evaluate_context(traj.trace_id, ctx, rubrics, "HeadTail")
                tokens_used["HeadTail"] = evals["HeadTail"].total_input_tokens

            if "LexicalRAG" in active_methods:
                ctx = rag_b.build_context(traj, rubrics, budget_tokens=target_budget)
                evals["LexicalRAG"] = judge.evaluate_context(traj.trace_id, ctx, rubrics, "LexicalRAG")
                tokens_used["LexicalRAG"] = evals["LexicalRAG"].total_input_tokens

            if "HybridTree" in active_methods:
                dossier = compressor.compress(traj, rubrics, budget_tokens=target_budget)
                evals["HybridTree"] = judge.evaluate_dossier(
                    traj, dossier, rubrics, compressor=compressor, method_name="HybridTree"
                )
                tokens_used["HybridTree"] = dossier.dossier_tokens

            full_eval = evals.get("FullTrace")
            ref_verdicts = _extract_reference_verdicts(traj, rubrics, full_eval)
            full_verdicts = full_eval.verdicts_dict() if full_eval else ref_verdicts
            full_probs = full_eval.probs_dict() if full_eval else None

            for m in active_methods:
                ev = evals[m]
                rep_ref = compute_sufficiency_report(ref_verdicts, ev.verdicts_dict())
                rep_full = compute_sufficiency_report(
                    full_verdicts, ev.verdicts_dict(), full_probs, ev.probs_dict()
                )
                tok = tokens_used[m]
                savings = max(0.0, (1.0 - tok / raw_tokens) * 100.0)
                acc[m].append(
                    {
                        "tokens": float(tok),
                        "savings_pct": savings,
                        "agreement": rep_ref["agreement"],
                        "f1": rep_ref["f1"],
                        "false_pass_rate": rep_ref["false_pass_rate"],
                        "false_fail_rate": rep_ref["false_fail_rate"],
                        "js_vs_full": rep_full["mean_js_divergence"],
                        "escalated": 1.0 if ev.escalated else 0.0,
                        "judge_cost": ev.estimated_cost_usd,
                    }
                )

            primary_eval = evals.get("HybridTree") or evals[active_methods[0]]
            failed_rids = [r.id for r in rubrics if not primary_eval.judgments[r.id].verdict]
            for rid in failed_rids:
                failure_counts[rid] = failure_counts.get(rid, 0) + 1

            per_trace_rows.append(
                {
                    "trace_id": traj.trace_id,
                    "steps": len(traj.steps),
                    "raw_tokens": raw_tokens,
                    "dossier_tokens": dossier.dossier_tokens if dossier else raw_tokens,
                    "escalated": primary_eval.escalated,
                    "cross_edges": len(dossier.cross_edges) if dossier else 0,
                    "failed_rubrics": failed_rids,
                }
            )

        after_jev_cost = float(self.decision_backend.usage_stats().get("estimated_cost_usd", 0.0))
        after_ov_cost = float(self.overview_backend.usage_stats().get("estimated_cost_usd", 0.0))
        total_comp_cost = max(0.0, (after_jev_cost - before_jev_cost) + (after_ov_cost - before_ov_cost))

        summaries: dict[str, MethodAggregate] = {}
        n = max(1, total)
        for m in active_methods:
            rows = acc[m]
            summaries[m] = MethodAggregate(
                method_name=m,
                num_traces=total,
                mean_tokens=round(sum(x["tokens"] for x in rows) / n, 1),
                mean_savings_pct=round(sum(x["savings_pct"] for x in rows) / n, 1),
                mean_agreement=round(sum(x["agreement"] for x in rows) / n, 4),
                mean_f1=round(sum(x["f1"] for x in rows) / n, 4),
                mean_false_pass_rate=round(sum(x["false_pass_rate"] for x in rows) / n, 4),
                mean_false_fail_rate=round(sum(x["false_fail_rate"] for x in rows) / n, 4),
                mean_js_vs_full=round(sum(x["js_vs_full"] for x in rows) / n, 4),
                escalation_rate=round(sum(x["escalated"] for x in rows) / n, 4),
                total_judge_cost_usd=round(sum(x["judge_cost"] for x in rows), 6),
                total_compressor_cost_usd=round(total_comp_cost if m == "HybridTree" else 0.0, 6),
            )

        failure_rates = {rid: round(cnt / n, 4) for rid, cnt in failure_counts.items()}
        return BatchReport(
            num_traces=total,
            budget_tokens=target_budget,
            method_summaries=summaries,
            failure_mode_counts=failure_counts,
            failure_mode_rates=failure_rates,
            rubric_names={r.id: r.name for r in rubrics},
            per_trace=per_trace_rows,
            compressor_usage=self.decision_backend.usage_stats(),
        )
