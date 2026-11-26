"""End-to-end tests for Evidence-First Judge, Tier-2 escalation, ATIF adapter, and metrics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from examples.data.generate_synthetic_trace import build_synthetic_180_step_trace
from trace_tree_eval.adapters import ATIFAdapter
from trace_tree_eval.backends.mock import (
    HashingEmbeddingBackend,
    MockDecisionBackend,
    MockTextBackend,
)
from trace_tree_eval.compression import CompressionConfig, HybridTreeCompressor
from trace_tree_eval.evaluation import (
    EvidenceFirstJudge,
    HeadTailBaseline,
    JudgeConfig,
    compute_sufficiency_report,
    consistency_k,
    false_pass_rate,
    js_divergence,
    kl_divergence,
)


class TestPipelineAndMetrics(unittest.TestCase):
    def test_metrics_math(self) -> None:
        self.assertAlmostEqual(js_divergence(0.9, 0.9), 0.0, places=5)
        self.assertGreater(js_divergence(0.95, 0.05), 0.5)
        self.assertGreater(kl_divergence(0.9, 0.1), 0.0)
        self.assertEqual(false_pass_rate([False, False, True], [True, False, True]), 0.5)
        self.assertEqual(consistency_k([[True, True, True], [False, True, False]]), 0.5)

    def test_atif_adapter_roundtrip(self) -> None:
        traj, _ = build_synthetic_180_step_trace()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "trace.json"
            p.write_text(json.dumps(traj.to_dict(), ensure_ascii=False), encoding="utf-8")
            loaded = ATIFAdapter().load(p)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].trace_id, traj.trace_id)
        self.assertEqual(len(loaded[0].steps), len(traj.steps))

    def test_hybrid_tree_outperforms_head_tail_on_false_pass_rate(self) -> None:
        traj, rubrics = build_synthetic_180_step_trace()
        gt = traj.ground_truth or {}
        embedder = HashingEmbeddingBackend()

        compressor = HybridTreeCompressor(
            decision_backend=MockDecisionBackend(),
            text_backend=MockTextBackend(),
            embedder=embedder,
            config=CompressionConfig(budget_tokens=3500),
        )
        judge = EvidenceFirstJudge(text_backend=MockTextBackend())

        head_ctx = HeadTailBaseline().build_context(traj, budget_tokens=3500)
        dossier = compressor.compress(traj, rubrics, budget_tokens=3500)

        res_head = judge.evaluate_context(traj.trace_id, head_ctx, rubrics, "HeadTail")
        res_tree = judge.evaluate_dossier(traj, dossier, rubrics, compressor=compressor)

        rep_head = compute_sufficiency_report(gt, res_head.verdicts_dict())
        rep_tree = compute_sufficiency_report(gt, res_tree.verdicts_dict())

        self.assertEqual(rep_tree["false_pass_rate"], 0.0)
        self.assertGreater(rep_head["false_pass_rate"], rep_tree["false_pass_rate"])
        self.assertGreaterEqual(rep_tree["agreement"], 0.83)

    def test_tier2_escalation_triggers_when_coverage_is_below_threshold(self) -> None:
        traj, rubrics = build_synthetic_180_step_trace()
        compressor = HybridTreeCompressor(
            decision_backend=MockDecisionBackend(),
            text_backend=MockTextBackend(),
            embedder=HashingEmbeddingBackend(),
            config=CompressionConfig(budget_tokens=2200),
        )
        dossier = compressor.compress(traj, rubrics, budget_tokens=2200)
        # Force high min_coverage_threshold to exercise Tier-2 unfolding path
        strict_judge = EvidenceFirstJudge(
            text_backend=MockTextBackend(),
            config=JudgeConfig(min_coverage_threshold=0.99, max_escalations=1, extra_unfold_budget_tokens=2000),
        )
        res = strict_judge.evaluate_dossier(traj, dossier, rubrics, compressor=compressor)
        self.assertTrue(res.escalated)
        self.assertTrue(len(res.unfolded_nodes) >= 1)


if __name__ == "__main__":
    unittest.main()
