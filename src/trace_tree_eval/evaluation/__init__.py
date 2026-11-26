"""Evaluation judge, controlled baselines, and Decision Sufficiency metrics."""

from trace_tree_eval.evaluation.baselines import (
    FullTraceBaseline,
    HeadTailBaseline,
    LexicalRagBaseline,
)
from trace_tree_eval.evaluation.judge import EvidenceFirstJudge, JudgeConfig
from trace_tree_eval.evaluation.metrics import (
    binary_f1,
    compute_sufficiency_report,
    consistency_k,
    decision_flip_rate,
    false_fail_rate,
    false_pass_rate,
    js_divergence,
    kl_divergence,
)
from trace_tree_eval.evaluation.runner import BatchEvaluator, BatchReport, MethodAggregate

__all__ = [
    "EvidenceFirstJudge",
    "JudgeConfig",
    "BatchEvaluator",
    "BatchReport",
    "MethodAggregate",
    "FullTraceBaseline",
    "HeadTailBaseline",
    "LexicalRagBaseline",
    "js_divergence",
    "kl_divergence",
    "decision_flip_rate",
    "false_pass_rate",
    "false_fail_rate",
    "binary_f1",
    "consistency_k",
    "compute_sufficiency_report",
]
