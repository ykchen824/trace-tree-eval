"""Stage 2: Cross-branch backward semantic & causal dependency retrieval."""

from __future__ import annotations

import numpy as np

from trace_tree_eval.backends.base import DecisionBackend, EmbeddingBackend
from trace_tree_eval.schema import CrossBranchEdge, TraceStep, TreeNode


def _step_to_l2_map(root: TreeNode) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for node in root.iter_nodes():
        if node.level == 2:
            for sid in node.step_ids:
                mapping[sid] = node.node_id
    return mapping


def find_causal_edges(
    root: TreeNode,
    cleaned_steps: list[TraceStep],
    step_risk_tags: dict[int, list[str]],
    embedder: EmbeddingBackend,
    decision_backend: DecisionBackend,
    trigger_threshold: float = 0.65,
    top_k_candidates: int = 4,
    causal_confirm_threshold: float = 0.60,
) -> list[CrossBranchEdge]:
    """For high-scoring target episodes, retrieve earlier causal/contradiction steps across branches."""
    if not cleaned_steps:
        return []

    step_map = {s.step_id: s for s in cleaned_steps}
    step_to_ep = _step_to_l2_map(root)

    # Identify target steps inside high-scoring L2 episodes (or tagged anomaly/contradiction)
    target_steps: list[TraceStep] = []
    for node in root.iter_nodes():
        if node.level != 2:
            continue
        max_r = max(node.rubric_probs.values(), default=0.0)
        if max_r >= trigger_threshold or node.critical_prob >= trigger_threshold:
            # Pick representative anomalous / decision steps from this episode
            for sid in node.step_ids:
                tags = step_risk_tags.get(sid, [])
                if (
                    "contradiction_cue" in tags
                    or "error" in tags
                    or "sensitive_data" in tags
                    or sid == node.step_ids[-1]
                ):
                    if sid in step_map:
                        target_steps.append(step_map[sid])

    if not target_steps:
        return []

    # Embed all cleaned steps once for fast backward cosine similarity
    ordered_steps = sorted(cleaned_steps, key=lambda s: s.step_id)
    texts = [s.render_text(include_header=False) or s.role for s in ordered_steps]
    vecs = embedder.embed(texts)
    idx_by_id = {s.step_id: idx for idx, s in enumerate(ordered_steps)}

    candidate_pairs: list[tuple[TraceStep, TraceStep, float]] = []
    for tgt in target_steps:
        t_idx = idx_by_id[tgt.step_id]
        if t_idx == 0:
            continue
        t_vec = vecs[t_idx]
        earlier_vecs = vecs[:t_idx]
        sims = earlier_vecs @ t_vec

        # Boost mid-task user constraints & sensitive-data steps so cross-branch constraints surface
        scored_indices: list[tuple[float, int]] = []
        for e_idx in range(t_idx):
            e_step = ordered_steps[e_idx]
            # Focus on cross-branch antecedents (not immediately adjacent inside same episode)
            if step_to_ep.get(e_step.step_id) == step_to_ep.get(tgt.step_id):
                continue
            tags = step_risk_tags.get(e_step.step_id, [])
            boost = 0.25 if "mid_task_constraint" in tags else (0.12 if "sensitive_data" in tags else 0.0)
            scored_indices.append((float(sims[e_idx]) + boost, e_idx))

        scored_indices.sort(key=lambda x: -x[0])
        for score, e_idx in scored_indices[:top_k_candidates]:
            if score > 0.08:
                candidate_pairs.append((ordered_steps[e_idx], tgt, score))

    if not candidate_pairs:
        return []

    batch_items: list[tuple[str, dict[str, str]]] = []
    for ant, tgt, _ in candidate_pairs:
        state = (
            f"[CANDIDATE_ANTECEDENT]\n{ant.render_text()}\n\n"
            f"[TARGET_STEP]\n{tgt.render_text()}"
        )
        batch_items.append(
            (
                state,
                {
                    "causal_link": (
                        "Does the candidate antecedent establish a constraint, factual premise, or "
                        "cause that directly affects or contradicts the target step?"
                    )
                },
            )
        )

    probs = decision_backend.batch_score_noul(batch_items)
    edges: list[CrossBranchEdge] = []
    seen_pairs: set[tuple[int, int]] = set()
    for (ant, tgt, _), p in zip(candidate_pairs, probs):
        conf = float(p.get("causal_link", 0.0))
        if conf >= causal_confirm_threshold and (ant.step_id, tgt.step_id) not in seen_pairs:
            seen_pairs.add((ant.step_id, tgt.step_id))
            edges.append(
                CrossBranchEdge(
                    source_step_id=ant.step_id,
                    target_step_id=tgt.step_id,
                    source_node_id=step_to_ep.get(ant.step_id, ""),
                    target_node_id=step_to_ep.get(tgt.step_id, ""),
                    reason="Causal antecedent / cross-branch constraint link",
                    confidence=round(conf, 4),
                )
            )
    return edges
