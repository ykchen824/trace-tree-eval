"""Stage 3 compression: submodular coverage selection, dossier rendering, and HybridTreeCompressor."""

from trace_tree_eval.compression.compressor import CompressionConfig, HybridTreeCompressor
from trace_tree_eval.compression.dossier import render_eval_dossier
from trace_tree_eval.compression.submodular import compute_rubric_coverage, select_nodes_submodular

__all__ = [
    "CompressionConfig",
    "HybridTreeCompressor",
    "select_nodes_submodular",
    "compute_rubric_coverage",
    "render_eval_dossier",
]
