"""End-to-end HybridTreeCompressor orchestrating Stages 0 through 3 and Tier-2 unfolding."""

from __future__ import annotations

from dataclasses import dataclass, field

from trace_tree_eval.backends.base import DecisionBackend, EmbeddingBackend, TextModelBackend
from trace_tree_eval.backends.mock import (
    HashingEmbeddingBackend,
    MockDecisionBackend,
    MockTextBackend,
)
from trace_tree_eval.compression.dossier import render_eval_dossier
from trace_tree_eval.compression.submodular import compute_rubric_coverage, select_nodes_submodular
from trace_tree_eval.schema import EvalDossier, RubricItem, TraceStep, Trajectory, TreeNode
from trace_tree_eval.tokens import estimate_tokens
from trace_tree_eval.tree.builder import TreeBuildConfig, build_hybrid_tree
from trace_tree_eval.tree.causal import find_causal_edges
from trace_tree_eval.tree.cleaner import clean_trajectory


@dataclass
class CompressionConfig:
    """Hyperparameters for HybridTreeCompressor."""

    budget_tokens: int = 8000
    branch_prune_threshold: float = 0.10
    critical_lambda: float = 0.25
    causal_trigger_threshold: float = 0.65
    causal_confirm_threshold: float = 0.60
    dedup_jaccard_threshold: float = 0.85
    min_loop_len: int = 3
    tree_build: TreeBuildConfig = field(default_factory=TreeBuildConfig)


class HybridTreeCompressor:
    """Rubric-conditioned hybrid tree compressor for long-horizon agent traces."""

    def __init__(
        self,
        decision_backend: DecisionBackend | None = None,
        text_backend: TextModelBackend | None = None,
        embedder: EmbeddingBackend | None = None,
        config: CompressionConfig | None = None,
    ) -> None:
        self.decision_backend = decision_backend or MockDecisionBackend()
        self.text_backend = text_backend or MockTextBackend()
        self.embedder = embedder or HashingEmbeddingBackend()
        self.config = config or CompressionConfig()
        # Cache cleaned steps per trace_id to support fast Tier-2 node unfolding
        self._cleaned_cache: dict[str, list[TraceStep]] = {}

    def compress(
        self,
        trajectory: Trajectory,
        rubrics: list[RubricItem],
        budget_tokens: int | None = None,
    ) -> EvalDossier:
        target_budget = budget_tokens or self.config.budget_tokens

        # Stage 0: Deterministic cleaning, loop folding & Global Manifest
        cleaned_steps, manifest, step_risk_tags = clean_trajectory(
            trajectory,
            dedup_jaccard_threshold=self.config.dedup_jaccard_threshold,
            min_loop_len=self.config.min_loop_len,
        )
        self._cleaned_cache[trajectory.trace_id] = cleaned_steps
        step_map = {s.step_id: s for s in cleaned_steps}

        # Stage 1: Semantic topic tree + risk tag bubbling + Jev-verified branch overviews
        tree_root = build_hybrid_tree(
            trace_id=trajectory.trace_id,
            cleaned_steps=cleaned_steps,
            step_risk_tags=step_risk_tags,
            embedder=self.embedder,
            text_backend=self.text_backend,
            decision_backend=self.decision_backend,
            config=self.config.tree_build,
        )

        # Top-down Jev branch screening + safe-window episode scoring
        self._score_tree_with_jev(tree_root, step_map, rubrics)

        # Stage 2: Cross-branch backward causal/contradiction retrieval
        cross_edges = find_causal_edges(
            root=tree_root,
            cleaned_steps=cleaned_steps,
            step_risk_tags=step_risk_tags,
            embedder=self.embedder,
            decision_backend=self.decision_backend,
            trigger_threshold=self.config.causal_trigger_threshold,
            causal_confirm_threshold=self.config.causal_confirm_threshold,
        )

        # Pin mid-task constraints and cross-branch causal endpoint steps
        pinned_step_ids: set[int] = set(manifest.mid_task_constraint_steps)
        pinned_ep_ids: set[str] = set()
        for edge in cross_edges:
            pinned_step_ids.add(edge.source_step_id)
            pinned_step_ids.add(edge.target_step_id)
            if edge.source_node_id:
                pinned_ep_ids.add(edge.source_node_id)
            if edge.target_node_id:
                pinned_ep_ids.add(edge.target_node_id)

        # Reserve tokens for fixed scaffolding (Section 1, Section 2, collapsed overviews, Section 4)
        fixed_tokens = self._estimate_fixed_overhead(trajectory, tree_root, pinned_step_ids, step_map)
        evidence_budget = max(800, target_budget - fixed_tokens)

        # Stage 3: Budget-constrained submodular selection over L2 episodes
        l2_nodes = [n for n in tree_root.iter_nodes() if n.level == 2]
        selected_ep_ids, coverage = select_nodes_submodular(
            candidate_nodes=l2_nodes,
            rubrics=rubrics,
            budget_tokens=evidence_budget,
            pinned_node_ids=pinned_ep_ids,
            critical_lambda=self.config.critical_lambda,
        )

        return render_eval_dossier(
            trajectory=trajectory,
            cleaned_steps=cleaned_steps,
            tree_root=tree_root,
            manifest=manifest,
            expanded_node_ids=set(selected_ep_ids),
            pinned_step_ids=pinned_step_ids,
            cross_edges=cross_edges,
            rubric_coverage=coverage,
            budget_tokens=target_budget,
        )

    def unfold_nodes(
        self,
        trajectory: Trajectory,
        dossier: EvalDossier,
        node_ids: list[str],
        rubrics: list[RubricItem] | None = None,
        extra_budget_tokens: int = 4000,
    ) -> EvalDossier:
        """Tier-2 Escalation: expand specified collapsed L1/L2 nodes into verbatim steps."""
        if dossier.tree_root is None:
            return dossier

        cleaned_steps = self._cleaned_cache.get(trajectory.trace_id)
        if cleaned_steps is None:
            cleaned_steps, _, _ = clean_trajectory(trajectory)
            self._cleaned_cache[trajectory.trace_id] = cleaned_steps

        new_expanded = set(dossier.expanded_node_ids)
        added_tokens = 0
        for nid in node_ids:
            node = dossier.tree_root.find_node(nid)
            if node is None:
                continue
            targets = node.children if node.level == 1 else [node]
            for ep in targets:
                if ep.node_id in new_expanded:
                    continue
                if added_tokens + ep.token_count <= extra_budget_tokens or not added_tokens:
                    new_expanded.add(ep.node_id)
                    added_tokens += ep.token_count

        # If requested nodes didn't match or were already expanded, expand top-critical collapsed episodes
        if added_tokens == 0:
            collapsed_eps = [
                n for n in dossier.tree_root.iter_nodes()
                if n.level == 2 and n.node_id not in new_expanded
            ]
            collapsed_eps.sort(key=lambda n: (-n.critical_prob, -max(n.rubric_probs.values(), default=0.0)))
            for ep in collapsed_eps:
                if added_tokens + ep.token_count <= extra_budget_tokens or added_tokens == 0:
                    new_expanded.add(ep.node_id)
                    added_tokens += ep.token_count
                    if added_tokens >= extra_budget_tokens:
                        break

        expanded_objs = [
            n for n in dossier.tree_root.iter_nodes()
            if n.level == 2 and n.node_id in new_expanded
        ]
        coverage = (
            compute_rubric_coverage(expanded_objs, rubrics)
            if rubrics
            else dossier.rubric_coverage
        )
        return render_eval_dossier(
            trajectory=trajectory,
            cleaned_steps=cleaned_steps,
            tree_root=dossier.tree_root,
            manifest=dossier.manifest,
            expanded_node_ids=new_expanded,
            pinned_step_ids=set(dossier.pinned_step_ids),
            cross_edges=dossier.cross_edges,
            rubric_coverage=coverage,
            budget_tokens=dossier.budget_tokens + extra_budget_tokens,
        )

    def _score_tree_with_jev(
        self,
        tree_root: TreeNode,
        step_map: dict[int, TraceStep],
        rubrics: list[RubricItem],
    ) -> None:
        all_questions = {
            r.id: f"Is this segment directly relevant as evidence for evaluating: {r.question}"
            for r in rubrics
        }
        all_questions["critical_anomaly"] = (
            "Does this segment contain errors, constraint updates, contradictions, "
            "verification checks, or high-impact decisions?"
        )

        # Pass 1: Top-down L1 branch overview + metadata screening
        l1_nodes = tree_root.children
        if l1_nodes:
            l1_items: list[tuple[str, dict[str, str]]] = []
            for b in l1_nodes:
                state = (
                    f"{b.title}\nRisk tags: {', '.join(b.risk_tags) or 'none'}\n"
                    f"Tools: {b.tool_counts}\nOverview: {b.overview}"
                )
                l1_items.append((state, all_questions))
            l1_probs = self.decision_backend.batch_score_noul(l1_items)
            for b, probs in zip(l1_nodes, l1_probs):
                b.critical_prob = float(probs.get("critical_anomaly", 0.0))
                b.rubric_probs = {r.id: float(probs.get(r.id, 0.0)) for r in rubrics}

        # Pass 2: Safe-window L2 episode scoring for surviving branches;
        # pruned branches are only scanned against high-risk rubrics (not all rubrics).
        high_risk_questions = {
            r.id: all_questions[r.id] for r in rubrics if r.high_risk
        }
        if high_risk_questions:
            high_risk_questions["critical_anomaly"] = all_questions["critical_anomaly"]

        surviving_eps: list[TreeNode] = []
        pruned_eps: list[TreeNode] = []
        for b in l1_nodes:
            max_b = max(b.rubric_probs.values(), default=0.0)
            survives = (
                max_b >= self.config.branch_prune_threshold
                or b.critical_prob >= self.config.branch_prune_threshold
                or bool(b.risk_tags)
            )
            if survives:
                surviving_eps.extend(b.children)
            else:
                pruned_eps.extend(b.children)

        if surviving_eps:
            surv_items = [
                (
                    "\n".join(step_map[sid].render_text() for sid in ep.step_ids if sid in step_map),
                    all_questions,
                )
                for ep in surviving_eps
            ]
            surv_probs = self.decision_backend.batch_score_noul(surv_items)
            for ep, probs in zip(surviving_eps, surv_probs):
                ep.critical_prob = float(probs.get("critical_anomaly", 0.0))
                ep.rubric_probs = {r.id: float(probs.get(r.id, 0.0)) for r in rubrics}

        if pruned_eps:
            if high_risk_questions:
                pruned_items = [
                    (
                        "\n".join(step_map[sid].render_text() for sid in ep.step_ids if sid in step_map),
                        high_risk_questions,
                    )
                    for ep in pruned_eps
                ]
                pruned_probs = self.decision_backend.batch_score_noul(pruned_items)
                for ep, probs in zip(pruned_eps, pruned_probs):
                    ep.critical_prob = float(probs.get("critical_anomaly", 0.0))
                    ep.rubric_probs = {
                        r.id: float(probs.get(r.id, 0.0)) if r.high_risk else 0.0
                        for r in rubrics
                    }
            else:
                for ep in pruned_eps:
                    ep.critical_prob = 0.0
                    ep.rubric_probs = {r.id: 0.0 for r in rubrics}

    @staticmethod
    def _estimate_fixed_overhead(
        trajectory: Trajectory,
        tree_root: TreeNode,
        pinned_step_ids: set[int],
        step_map: dict[int, TraceStep],
    ) -> int:
        pinned_tok = sum(
            step_map[sid].token_count() for sid in pinned_step_ids if sid in step_map
        )
        overview_tok = sum(estimate_tokens(b.overview) + 25 for b in tree_root.children)
        scaffold_tok = (
            estimate_tokens(trajectory.task_instruction)
            + estimate_tokens(trajectory.final_output)
            + 300  # Global Manifest + headings
        )
        return pinned_tok + overview_tok + scaffold_tok
