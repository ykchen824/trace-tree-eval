"""trace-tree-eval: Rubric-conditioned hybrid tree compaction and Evidence-First evaluation."""

from trace_tree_eval.api import (
    build_runtime_backends,
    compress_trace,
    evaluate_trace,
    load_traces,
)
from trace_tree_eval.compression import CompressionConfig, HybridTreeCompressor
from trace_tree_eval.config import env_summary, load_dotenv
from trace_tree_eval.evaluation import (
    BatchEvaluator,
    BatchReport,
    EvidenceFirstJudge,
    JudgeConfig,
    compute_sufficiency_report,
)
from trace_tree_eval.presets import get_preset_rubric, list_preset_rubrics, resolve_rubrics
from trace_tree_eval.report_html import render_batch_report_html, render_dossier_html
from trace_tree_eval.schema import (
    CrossBranchEdge,
    EvalDossier,
    GlobalManifest,
    JudgeEvaluation,
    RubricItem,
    RubricJudgment,
    TraceStep,
    Trajectory,
    TreeNode,
)

__all__ = [
    "TraceStep",
    "Trajectory",
    "RubricItem",
    "CrossBranchEdge",
    "TreeNode",
    "GlobalManifest",
    "EvalDossier",
    "RubricJudgment",
    "JudgeEvaluation",
    "CompressionConfig",
    "HybridTreeCompressor",
    "JudgeConfig",
    "EvidenceFirstJudge",
    "BatchEvaluator",
    "BatchReport",
    "compute_sufficiency_report",
    "load_dotenv",
    "env_summary",
    "list_preset_rubrics",
    "get_preset_rubric",
    "resolve_rubrics",
    "load_traces",
    "build_runtime_backends",
    "compress_trace",
    "evaluate_trace",
    "render_dossier_html",
    "render_batch_report_html",
]
