"""Budget-constrained monotone submodular coverage over calibrated Jev probabilities."""

from __future__ import annotations

from trace_tree_eval.schema import RubricItem, TreeNode


def compute_rubric_coverage(
    selected_nodes: list[TreeNode],
    rubrics: list[RubricItem],
) -> dict[str, float]:
    """Compute joint probability 1 - prod(1 - p_rel(u | r_m)) for each rubric."""
    coverage: dict[str, float] = {}
    for r in rubrics:
        miss_prob = 1.0
        for node in selected_nodes:
            p = max(0.0, min(1.0, float(node.rubric_probs.get(r.id, 0.0))))
            miss_prob *= 1.0 - p
        coverage[r.id] = round(1.0 - miss_prob, 4)
    return coverage


def select_nodes_submodular(
    candidate_nodes: list[TreeNode],
    rubrics: list[RubricItem],
    budget_tokens: int,
    pinned_node_ids: set[str] | None = None,
    critical_lambda: float = 0.25,
) -> tuple[list[str], dict[str, float]]:
    """Greedy cost-normalized monotone submodular selection for balanced rubric coverage.

    Objective:
        F(S) = sum_m w_m * (1 - prod_{u in S} (1 - p_rel(u | r_m))) + lambda * sum_{u in S} p_crit(u)
    """
    pinned_ids = set(pinned_node_ids or set())
    node_by_id = {n.node_id: n for n in candidate_nodes}

    selected: list[TreeNode] = []
    selected_ids: set[str] = set()
    used_tokens = 0

    # Initialize miss probability per rubric: prod_{u in S} (1 - p(u | r_m))
    miss_probs: dict[str, float] = {r.id: 1.0 for r in rubrics}

    # First admit pinned nodes if budget allows (sorted by critical_prob desc)
    pinned_nodes = [node_by_id[nid] for nid in pinned_ids if nid in node_by_id]
    pinned_nodes.sort(key=lambda n: (-n.critical_prob, n.node_id))
    for node in pinned_nodes:
        tok = max(1, node.token_count)
        if used_tokens + tok <= budget_tokens or not selected:
            selected.append(node)
            selected_ids.add(node.node_id)
            used_tokens += tok
            for r in rubrics:
                p = max(0.0, min(1.0, float(node.rubric_probs.get(r.id, 0.0))))
                miss_probs[r.id] *= 1.0 - p

    remaining = [n for n in candidate_nodes if n.node_id not in selected_ids]

    while remaining and used_tokens < budget_tokens:
        best_node: TreeNode | None = None
        best_ratio = 0.0

        for node in remaining:
            tok = max(1, node.token_count)
            if used_tokens + tok > budget_tokens:
                continue

            # Marginal gain delta_F(node | S)
            delta = critical_lambda * float(node.critical_prob)
            for r in rubrics:
                p = max(0.0, min(1.0, float(node.rubric_probs.get(r.id, 0.0))))
                # (1 - miss*(1-p)) - (1 - miss) = miss * p
                delta += float(r.weight) * miss_probs[r.id] * p

            ratio = delta / tok
            if ratio > best_ratio:
                best_ratio = ratio
                best_node = node

        if best_node is None or best_ratio <= 0.0:
            break

        selected.append(best_node)
        selected_ids.add(best_node.node_id)
        used_tokens += max(1, best_node.token_count)
        for r in rubrics:
            p = max(0.0, min(1.0, float(best_node.rubric_probs.get(r.id, 0.0))))
            miss_probs[r.id] *= 1.0 - p
        remaining = [n for n in remaining if n.node_id != best_node.node_id]

    ordered_ids = [n.node_id for n in sorted(selected, key=lambda n: (n.step_ids[0] if n.step_ids else 0, n.node_id))]
    coverage = {r.id: round(1.0 - miss_probs[r.id], 4) for r in rubrics}
    return ordered_ids, coverage
