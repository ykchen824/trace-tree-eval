"""Stage 3: Structured 4-section Eval Evidence Dossier renderer with overview/verbatim isolation."""

from __future__ import annotations

from trace_tree_eval.schema import (
    CrossBranchEdge,
    EvalDossier,
    GlobalManifest,
    TraceStep,
    Trajectory,
    TreeNode,
)
from trace_tree_eval.tokens import estimate_tokens


def render_eval_dossier(
    trajectory: Trajectory,
    cleaned_steps: list[TraceStep],
    tree_root: TreeNode,
    manifest: GlobalManifest,
    expanded_node_ids: set[str],
    pinned_step_ids: set[int],
    cross_edges: list[CrossBranchEdge],
    rubric_coverage: dict[str, float],
    budget_tokens: int,
) -> EvalDossier:
    """Render a Partially Expanded Execution Tree Dossier with strict evidence isolation."""
    step_map = {s.step_id: s for s in cleaned_steps}
    raw_step_map = trajectory.step_map()

    # Map target_step_id -> list of causal source annotations
    causal_notes: dict[int, list[str]] = {}
    for edge in cross_edges:
        causal_notes.setdefault(edge.target_step_id, []).append(
            f"step {edge.source_step_id} (conf={edge.confidence:.2f})"
        )

    lines: list[str] = []

    # Section 1: Task Instruction, Deliverables & Pinned Mid-Task Constraints
    lines.append("## 1. Task Instruction, Deliverables & Pinned Mid-Task Constraints")
    lines.append(f"**Trace ID:** `{trajectory.trace_id}`")
    lines.append(f"**Task Instruction:**\n{trajectory.task_instruction}")
    if trajectory.deliverables:
        lines.append("**Required Deliverables:**")
        for d in trajectory.deliverables:
            lines.append(f"- {d}")
    if pinned_step_ids:
        lines.append("**Pinned Mid-Task Constraints & Key Causal Anchors (Verbatim):**")
        for sid in sorted(pinned_step_ids):
            step = step_map.get(sid) or raw_step_map.get(sid)
            if step is not None:
                lines.append(step.render_text())
    lines.append("")

    # Section 2: Global Execution Manifest (disambiguates Errors of Omission)
    lines.append("## 2. Global Execution Manifest")
    lines.append(
        f"- total_steps: {manifest.total_steps} | raw_tokens: {manifest.raw_tokens} | cleaned_tokens: {manifest.cleaned_tokens}"
    )
    lines.append(
        f"- verification_detected: {str(manifest.verification_detected).lower()} | "
        f"verification_steps: {manifest.verification_steps}"
    )
    lines.append(f"- mid_task_constraint_steps: {manifest.mid_task_constraint_steps}")
    if manifest.tool_stats:
        tool_summary = "; ".join(
            f"{name}(calls={st['calls']}, errors={st['errors']}, steps={st['first_step']}..{st['last_step']})"
            for name, st in sorted(manifest.tool_stats.items())
        )
        lines.append(f"- tool_stats: {tool_summary}")
    if manifest.loop_folds:
        fold_str = "; ".join(
            f"steps {f['start_step']}-{f['end_step']} ({f['count']}x {f['tool']} {f['status']})"
            for f in manifest.loop_folds
        )
        lines.append(f"- loop_folds: {fold_str}")
    lines.append("")

    # Section 3: Partially Expanded Execution Tree
    lines.append("## 3. Partially Expanded Execution Tree")
    expanded_ids_ordered: list[str] = []
    collapsed_ids_ordered: list[str] = []
    selected_step_ids: set[int] = set(pinned_step_ids)

    for b_node in tree_root.children:
        b_tags = f" | risks={','.join(b_node.risk_tags)}" if b_node.risk_tags else ""
        b_has_expanded_child = any(ep.node_id in expanded_node_ids for ep in b_node.children)

        if not b_has_expanded_child and b_node.node_id not in expanded_node_ids:
            b_node.expanded = False
            collapsed_ids_ordered.append(b_node.node_id)
            verified_lbl = "Verified Overview - Not Ground-Truth Evidence" if b_node.overview_verified else "Overview"
            lines.append(
                f"- [COLLAPSED | {verified_lbl} | node_id={b_node.node_id} | {b_node.title}{b_tags}] "
                f"{b_node.overview}"
            )
            continue

        b_node.expanded = True
        lines.append(f"### {b_node.title}{b_tags}")
        for ep in b_node.children:
            ep_tags = f" | risks={','.join(ep.risk_tags)}" if ep.risk_tags else ""
            if ep.node_id in expanded_node_ids or b_node.node_id in expanded_node_ids:
                ep.expanded = True
                expanded_ids_ordered.append(ep.node_id)
                lines.append(f"#### [EXPANDED | node_id={ep.node_id} | {ep.title}{ep_tags}]")
                for sid in ep.step_ids:
                    selected_step_ids.add(sid)
                    step = step_map.get(sid) or raw_step_map.get(sid)
                    if step is None:
                        continue
                    if sid in causal_notes:
                        lines.append(f"[Causal Parents: {', '.join(causal_notes[sid])}]")
                    lines.append(step.render_text())
            else:
                ep.expanded = False
                collapsed_ids_ordered.append(ep.node_id)
                lines.append(
                    f"- [COLLAPSED | Verified Overview - Not Ground-Truth Evidence | node_id={ep.node_id} | "
                    f"{ep.title}{ep_tags}] {ep.overview}"
                )
    lines.append("")

    # Section 4: Final Output & Deliverable Check
    lines.append("## 4. Final Output & Deliverable Check")
    if trajectory.deliverables:
        lines.append("Required Deliverables: " + " | ".join(trajectory.deliverables))
    lines.append(f"Final Output:\n{trajectory.final_output}")

    rendered_markdown = "\n".join(lines)
    dossier_tokens = estimate_tokens(rendered_markdown)
    raw_tokens = max(1, manifest.raw_tokens)
    compression_ratio = round(dossier_tokens / raw_tokens, 4)

    return EvalDossier(
        trace_id=trajectory.trace_id,
        budget_tokens=budget_tokens,
        dossier_tokens=dossier_tokens,
        compression_ratio=compression_ratio,
        rendered_markdown=rendered_markdown,
        manifest=manifest,
        expanded_node_ids=expanded_ids_ordered,
        collapsed_node_ids=collapsed_ids_ordered,
        selected_step_ids=sorted(selected_step_ids),
        pinned_step_ids=sorted(pinned_step_ids),
        cross_edges=cross_edges,
        rubric_coverage=rubric_coverage,
        tree_root=tree_root,
    )
