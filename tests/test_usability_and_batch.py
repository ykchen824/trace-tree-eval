"""Tests for .env auto-loading, DiskCache, presets, high-level API, selective pruning, BatchEvaluator, HTML reports, and CLI."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from typer.testing import CliRunner

from examples.data.generate_synthetic_trace import build_synthetic_180_step_trace
from trace_tree_eval import (
    BatchEvaluator,
    CompressionConfig,
    HybridTreeCompressor,
    compress_trace,
    env_summary,
    evaluate_trace,
    list_preset_rubrics,
    load_dotenv,
    render_batch_report_html,
    render_dossier_html,
    resolve_rubrics,
)
from trace_tree_eval.backends import DiskCache, HashingEmbeddingBackend, MockDecisionBackend, MockTextBackend
from trace_tree_eval.cli import app
from trace_tree_eval.schema import RubricItem, TraceStep, Trajectory


class TestUsabilityAndBatch(unittest.TestCase):
    def test_dotenv_loader_and_env_summary(self) -> None:
        key_a = "TTE_TEST_VAR_ALPHA"
        key_b = "TTE_TEST_VAR_BETA"
        os.environ.pop(key_a, None)
        os.environ[key_b] = "existing_val"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                env_path = Path(tmpdir) / ".env"
                env_path.write_text(
                    f'# Comment line\n{key_a}="hello_world"\nexport {key_b}=\'should_not_override\'\n',
                    encoding="utf-8",
                )
                loaded = load_dotenv(env_path, override=False)
                self.assertEqual(loaded, env_path)
                self.assertEqual(os.environ.get(key_a), "hello_world")
                self.assertEqual(os.environ.get(key_b), "existing_val")

            summary = env_summary()
            self.assertIn("TYPESAFE_API_KEY", summary)
            self.assertIn("JUDGE_MODEL", summary)
        finally:
            os.environ.pop(key_a, None)
            os.environ.pop(key_b, None)

    def test_disk_cache_hits_and_misses(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache1 = DiskCache(tmpdir)
            key = DiskCache.make_key("jev_noul", "typesafe/jev-latest", {"q": "ok"})
            self.assertIsNone(cache1.get(key))
            cache1.set(key, {"ok": 0.92})

            # Second instance reading from same directory must hit
            cache2 = DiskCache(tmpdir)
            val = cache2.get(key)
            self.assertEqual(val, {"ok": 0.92})
            self.assertEqual(cache2.stats()["hits"], 1)

    def test_presets_and_high_level_api(self) -> None:
        presets = list_preset_rubrics()
        self.assertEqual(
            presets,
            ["coding_agent", "deep_research", "multi_turn_assistant", "web_agent"],
        )
        items = resolve_rubrics("preset:web_agent")
        self.assertTrue(len(items) >= 4)

        traj, _ = build_synthetic_180_step_trace()
        dossier = compress_trace(traj, rubrics="deep_research", budget_tokens=3500)
        self.assertLess(dossier.dossier_tokens, traj.total_tokens())

        dossier2, evaluation = evaluate_trace(traj, rubrics="preset:deep_research", budget_tokens=3500)
        self.assertEqual(dossier2.trace_id, traj.trace_id)
        self.assertIn("mid_task_constraint", evaluation.judgments)

    def test_selective_high_risk_pruning_reduces_calls_when_no_high_risk(self) -> None:
        steps = [
            TraceStep(step_id=i, role="assistant", action=f"browse_{i}", observation=f"neutral doc {i}")
            for i in range(1, 25)
        ]
        traj = Trajectory(trace_id="t-prune", task_instruction="Find apple price", steps=steps)

        no_hr_rubrics = [
            RubricItem(id="apple_price", name="Apple", question="Mentions apple price?", high_risk=False)
        ]
        hr_rubrics = [
            RubricItem(id="apple_price", name="Apple", question="Mentions apple price?", high_risk=True)
        ]

        jev_no_hr = MockDecisionBackend()
        comp_no_hr = HybridTreeCompressor(
            decision_backend=jev_no_hr,
            text_backend=MockTextBackend(),
            embedder=HashingEmbeddingBackend(),
            config=CompressionConfig(branch_prune_threshold=0.95),
        )
        comp_no_hr.compress(traj, no_hr_rubrics, budget_tokens=1500)

        jev_hr = MockDecisionBackend()
        comp_hr = HybridTreeCompressor(
            decision_backend=jev_hr,
            text_backend=MockTextBackend(),
            embedder=HashingEmbeddingBackend(),
            config=CompressionConfig(branch_prune_threshold=0.95),
        )
        comp_hr.compress(traj, hr_rubrics, budget_tokens=1500)

        self.assertLess(jev_no_hr.usage_stats()["calls"], jev_hr.usage_stats()["calls"])

    def test_batch_evaluator_and_html_reports(self) -> None:
        traj, rubrics = build_synthetic_180_step_trace()
        runner = BatchEvaluator()
        report = runner.run([traj], rubrics, budget_tokens=3500, include_baselines=True)

        self.assertEqual(report.num_traces, 1)
        self.assertIn("HybridTree", report.method_summaries)
        self.assertEqual(report.method_summaries["HybridTree"].mean_false_pass_rate, 0.0)
        self.assertGreater(report.failure_mode_counts["mid_task_constraint"], 0)

        batch_html = render_batch_report_html(report)
        self.assertIn("trace-tree-eval Batch Benchmark Report", batch_html)
        self.assertIn("Failure-Mode Clustering", batch_html)
        self.assertIn("synthetic-deep-research-180", batch_html)

        dossier, evaluation = evaluate_trace(traj, rubrics=rubrics, budget_tokens=3500)
        dossier_html = render_dossier_html(dossier, evaluation, rubrics)
        self.assertIn("EvalDossier: synthetic-deep-research-180", dossier_html)
        self.assertIn("Evidence-First Judge Verdicts", dossier_html)

    def test_cli_doctor_presets_and_benchmark(self) -> None:
        runner = CliRunner()
        res_doc = runner.invoke(app, ["doctor"])
        self.assertEqual(res_doc.exit_code, 0, res_doc.output)
        self.assertIn("TYPESAFE_API_KEY", res_doc.output)

        res_pre = runner.invoke(app, ["presets"])
        self.assertEqual(res_pre.exit_code, 0, res_pre.output)
        self.assertIn("preset:deep_research", res_pre.output)

        with tempfile.TemporaryDirectory() as tmpdir:
            html_file = Path(tmpdir) / "batch.html"
            res_bm = runner.invoke(
                app,
                [
                    "benchmark",
                    "--dataset",
                    "synthetic",
                    "--budget",
                    "3500",
                    "--html-out",
                    str(html_file),
                ],
            )
            self.assertEqual(res_bm.exit_code, 0, res_bm.output)
            self.assertTrue(html_file.is_file())
            self.assertIn("HybridTree", res_bm.output)


if __name__ == "__main__":
    unittest.main()
