"""Tests for calibrated submodular budget balancing, cross-branch causal edges, and Tier-2 unfolding."""

from __future__ import annotations

import unittest

from examples.data.generate_synthetic_trace import build_synthetic_180_step_trace
from trace_tree_eval.backends.mock import (
    HashingEmbeddingBackend,
    MockDecisionBackend,
    MockTextBackend,
)
from trace_tree_eval.compression import CompressionConfig, HybridTreeCompressor, select_nodes_submodular
from trace_tree_eval.schema import RubricItem, TreeNode


class TestSubmodularAndCausal(unittest.TestCase):
    def test_submodular_avoids_single_rubric_clustering(self) -> None:
        rubrics = [
            RubricItem(id="r1", name="R1", question="Check R1", weight=1.0),
            RubricItem(id="r2", name="R2", question="Check R2", weight=1.0),
        ]
        # 4 redundant nodes scoring high only on r1, and 1 node scoring high on r2
        nodes = [
            TreeNode(node_id=f"r1_{i}", level=2, title=f"R1-{i}", step_ids=[i],
                     rubric_probs={"r1": 0.90, "r2": 0.02}, critical_prob=0.1, token_count=100)
            for i in range(1, 5)
        ] + [
            TreeNode(node_id="r2_1", level=2, title="R2-1", step_ids=[10],
                     rubric_probs={"r1": 0.02, "r2": 0.90}, critical_prob=0.1, token_count=100)
        ]

        # Budget fits 2 nodes: submodular must pick one from r1 and the r2 node!
        selected_ids, coverage = select_nodes_submodular(nodes, rubrics, budget_tokens=200)
        self.assertEqual(len(selected_ids), 2)
        self.assertIn("r2_1", selected_ids)
        self.assertGreaterEqual(coverage["r1"], 0.90)
        self.assertGreaterEqual(coverage["r2"], 0.90)

    def test_causal_edges_and_unfold_on_180_step_trace(self) -> None:
        traj, rubrics = build_synthetic_180_step_trace()
        compressor = HybridTreeCompressor(
            decision_backend=MockDecisionBackend(),
            text_backend=MockTextBackend(),
            embedder=HashingEmbeddingBackend(),
            config=CompressionConfig(budget_tokens=3500),
        )
        dossier = compressor.compress(traj, rubrics, budget_tokens=3500)

        # Must compress well below raw tokens and stay compact
        self.assertLess(dossier.dossier_tokens, traj.total_tokens() * 0.4)
        self.assertIn(18, dossier.pinned_step_ids)
        # Must discover cross-branch causal edge linking Step 18 / 112 -> Step 148
        edge_pairs = {(e.source_step_id, e.target_step_id) for e in dossier.cross_edges}
        self.assertTrue(
            any(tgt == 148 and src in (18, 112) for src, tgt in edge_pairs),
            f"Expected causal edge into step 148 from step 18 or 112, got {edge_pairs}",
        )

        # Tier-2 unfold_nodes must expand at least one previously collapsed node
        self.assertTrue(len(dossier.collapsed_node_ids) > 0)
        target_to_unfold = dossier.collapsed_node_ids[0]
        expanded_dossier = compressor.unfold_nodes(
            traj, dossier, [target_to_unfold], rubrics=rubrics, extra_budget_tokens=3000
        )
        self.assertGreater(
            len(expanded_dossier.expanded_node_ids), len(dossier.expanded_node_ids)
        )


if __name__ == "__main__":
    unittest.main()
