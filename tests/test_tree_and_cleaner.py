"""Tests for Stage 0 cleaning, Global Manifest, semantic tree construction, risk bubbling & overview pruning."""

from __future__ import annotations

import unittest

from trace_tree_eval.backends.mock import (
    HashingEmbeddingBackend,
    MockDecisionBackend,
    MockTextBackend,
)
from trace_tree_eval.schema import TraceStep, Trajectory
from trace_tree_eval.tree.builder import TreeBuildConfig, build_hybrid_tree
from trace_tree_eval.tree.cleaner import clean_trajectory


class TestTreeAndCleaner(unittest.TestCase):
    def test_cleaner_folds_loops_and_extracts_manifest(self) -> None:
        steps = [
            TraceStep(step_id=1, role="user", action="Check deployment"),
            TraceStep(step_id=2, role="user", action="补充要求：不得跳过自我检查"),
        ]
        for sid in range(3, 9):
            steps.append(
                TraceStep(
                    step_id=sid,
                    role="assistant",
                    tool_name="fetch_status",
                    action="fetch_status('https://svc.local/health')",
                    observation="HTTP 503 timeout",
                    status="error",
                )
            )
        steps.append(
            TraceStep(
                step_id=9,
                role="assistant",
                tool_name="verify_output",
                action="verify_output()",
                observation="All checks passed.",
            )
        )
        traj = Trajectory(trace_id="t-clean", task_instruction="Deploy", steps=steps)

        cleaned, manifest, risk_tags = clean_trajectory(traj, min_loop_len=3)
        self.assertLess(len(cleaned), len(steps))
        self.assertEqual(manifest.total_steps, 9)
        self.assertEqual(len(manifest.loop_folds), 1)
        self.assertEqual(manifest.loop_folds[0]["count"], 6)
        self.assertTrue(manifest.verification_detected)
        self.assertIn(9, manifest.verification_steps)
        self.assertIn(2, manifest.mid_task_constraint_steps)
        self.assertIn("error", risk_tags[3])

    def test_tree_risk_bubbling_multi_agent_and_sjc_overview_pruning(self) -> None:
        steps = [
            TraceStep(step_id=1, role="assistant", agent_id="main", action="init", observation="setup workspace"),
            TraceStep(step_id=2, role="assistant", agent_id="main", action="read", observation="read public docs"),
            TraceStep(
                step_id=3,
                role="assistant",
                agent_id="sub_agent",
                tool_name="read_file",
                action="read_file('.env')",
                observation="SECRET_API_KEY=12345",
                status="error",
            ),
            TraceStep(step_id=4, role="assistant", agent_id="sub_agent", action="done", observation="sub finished"),
        ]
        traj = Trajectory(trace_id="t-tree", task_instruction="Inspect", steps=steps)
        cleaned, _, risk_tags = clean_trajectory(traj)

        # Inject hallucinated sentence and verify SJC-style Jev pruning removes it
        text_backend = MockTextBackend(inject_hallucination=True)
        jev_backend = MockDecisionBackend()
        root = build_hybrid_tree(
            trace_id=traj.trace_id,
            cleaned_steps=cleaned,
            step_risk_tags=risk_tags,
            embedder=HashingEmbeddingBackend(),
            text_backend=text_backend,
            decision_backend=jev_backend,
            config=TreeBuildConfig(min_branch_steps=2, generate_overviews=True),
        )

        # Should preserve multi-agent separation across Level-1 branches
        agent_ids = {b.agent_id for b in root.children}
        self.assertEqual(agent_ids, {"main", "sub_agent"})

        # Risk tags from step 3 must bubble up to its sub_agent branch and root
        self.assertIn("sensitive_data", root.risk_tags)
        self.assertIn("error", root.risk_tags)
        sub_branch = [b for b in root.children if b.agent_id == "sub_agent"][0]
        self.assertIn("sensitive_data", sub_branch.risk_tags)

        # Hallucinated sentence marked [UNGROUNDED] must be pruned from overview
        for b in root.children:
            self.assertTrue(b.overview_verified)
            self.assertNotIn("[UNGROUNDED]", b.overview)
            self.assertNotIn("虚构结论", b.overview)


if __name__ == "__main__":
    unittest.main()
