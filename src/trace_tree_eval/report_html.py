"""Self-contained interactive HTML report renderers for EvalDossier and BatchReport."""

from __future__ import annotations

import html
from typing import Any

from trace_tree_eval.evaluation.runner import BatchReport
from trace_tree_eval.schema import EvalDossier, JudgeEvaluation, RubricItem, TreeNode

_CSS = """
:root {
  --bg: #0f172a; --card: #1e293b; --border: #334155; --text: #f1f5f9;
  --muted: #94a3b8; --accent: #38bdf8; --ok: #22c55e; --warn: #f59e0b; --err: #ef4444;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg); color: var(--text); margin: 0; padding: 24px; line-height: 1.5; }
.wrap { max-width: 1180px; margin: 0 auto; }
h1, h2, h3 { margin: 0 0 12px 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 12px; margin-bottom: 20px; }
.kpi { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
.kpi .label { font-size: 12px; color: var(--muted); }
.kpi .val { font-size: 22px; font-weight: 700; margin-top: 4px; color: var(--accent); }
.panel { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 18px; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; margin-right: 6px; }
.pill-ok { background: rgba(34,197,94,.18); color: #86efac; border: 1px solid rgba(34,197,94,.4); }
.pill-err { background: rgba(239,68,68,.18); color: #fca5a5; border: 1px solid rgba(239,68,68,.4); }
.pill-warn { background: rgba(245,158,11,.18); color: #fde68a; border: 1px solid rgba(245,158,11,.4); }
.pill-muted { background: rgba(148,163,184,.15); color: #cbd5e1; border: 1px solid rgba(148,163,184,.3); }
details { border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; background: rgba(15,23,42,.45); }
summary { cursor: pointer; font-weight: 600; }
pre { background: #020617; border: 1px solid var(--border); padding: 10px; border-radius: 6px;
  overflow-x: auto; white-space: pre-wrap; word-break: break-word; font-size: 12px; margin: 8px 0 0 0; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 600; }
.bar-bg { background: #020617; border-radius: 6px; height: 14px; width: 100%; overflow: hidden; }
.bar-fill { background: var(--err); height: 100%; }
"""


def _esc(val: Any) -> str:
    return html.escape(str(val), quote=True)


def render_dossier_html(
    dossier: EvalDossier,
    evaluation: JudgeEvaluation | None = None,
    rubrics: list[RubricItem] | None = None,
) -> str:
    """Render an interactive, offline-safe HTML viewer for a compressed EvalDossier."""
    savings_pct = max(0.0, (1.0 - dossier.dossier_tokens / max(1, dossier.manifest.raw_tokens)) * 100.0)
    verif_badge = (
        '<span class="pill pill-ok">Verification Detected</span>'
        if dossier.manifest.verification_detected
        else '<span class="pill pill-err">No Verification Detected</span>'
    )
    esc_badge = ""
    if evaluation is not None:
        esc_badge = (
            '<span class="pill pill-warn">Tier-2 Escalated</span>'
            if evaluation.escalated
            else '<span class="pill pill-ok">Tier-1 Sufficient</span>'
        )

    # Render Judgments table if evaluation is provided
    eval_html = ""
    if evaluation is not None:
        name_map = {r.id: r.name for r in (rubrics or [])}
        rows = []
        for rid, j in evaluation.judgments.items():
            badge = '<span class="pill pill-ok">PASS</span>' if j.verdict else '<span class="pill pill-err">FAIL</span>'
            rows.append(
                f"<tr><td>{_esc(name_map.get(rid, rid))}</td><td>{badge}</td>"
                f"<td>{j.prob_true:.2f} (conf {j.confidence:.2f})</td>"
                f"<td>{_esc(j.cited_step_ids)}</td><td>{_esc(j.rationale)}</td></tr>"
            )
        eval_html = (
            '<div class="panel"><h2>Evidence-First Judge Verdicts</h2>'
            "<table><thead><tr><th>Rubric</th><th>Verdict</th><th>Prob / Conf</th>"
            "<th>Cited Steps</th><th>Rationale</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )

    # Render Cross-Branch Edges
    edges_html = "<p class='muted'>None detected.</p>"
    if dossier.cross_edges:
        edge_items = [
            f"<li><span class='pill pill-warn'>step {_esc(e.source_step_id)} &rarr; step {_esc(e.target_step_id)}</span> "
            f"conf={e.confidence:.2f} ({_esc(e.source_node_id)} &rarr; {_esc(e.target_node_id)})</li>"
            for e in dossier.cross_edges
        ]
        edges_html = f"<ul>{''.join(edge_items)}</ul>"

    # Render Hybrid Tree structure
    tree_blocks: list[str] = []
    if dossier.tree_root is not None:
        for b in dossier.tree_root.children:
            tree_blocks.append(_render_branch_html(b))

    # Render raw markdown preview inside collapsible section
    md_preview = _esc(dossier.rendered_markdown)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EvalDossier: {_esc(dossier.trace_id)}</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>EvalDossier: {_esc(dossier.trace_id)} {verif_badge} {esc_badge}</h1>
  <div class="grid">
    <div class="kpi"><div class="label">Raw &rarr; Dossier Tokens</div><div class="val">{dossier.manifest.raw_tokens} &rarr; {dossier.dossier_tokens}</div></div>
    <div class="kpi"><div class="label">Token Savings</div><div class="val">{savings_pct:.1f}%</div></div>
    <div class="kpi"><div class="label">Expanded / Collapsed</div><div class="val">{len(dossier.expanded_node_ids)} / {len(dossier.collapsed_node_ids)}</div></div>
    <div class="kpi"><div class="label">Cross-Branch Edges</div><div class="val">{len(dossier.cross_edges)}</div></div>
    <div class="kpi"><div class="label">Pinned Steps</div><div class="val">{len(dossier.pinned_step_ids)}</div></div>
  </div>

  {eval_html}

  <div class="panel">
    <h2>Global Manifest &amp; Cross-Branch Causal Links</h2>
    <p><strong>Total Steps:</strong> {dossier.manifest.total_steps} |
       <strong>Mid-Task Constraint Steps:</strong> {_esc(dossier.manifest.mid_task_constraint_steps)} |
       <strong>Verification Steps:</strong> {_esc(dossier.manifest.verification_steps)} |
       <strong>Loop Folds:</strong> {len(dossier.manifest.loop_folds)}</p>
    {edges_html}
  </div>

  <div class="panel">
    <h2>Hybrid Execution Tree (Collapsible)</h2>
    {''.join(tree_blocks) if tree_blocks else '<p>No tree metadata available.</p>'}
  </div>

  <div class="panel">
    <details>
      <summary>Full Rendered Dossier Markdown ({dossier.dossier_tokens} tokens)</summary>
      <pre>{md_preview}</pre>
    </details>
  </div>
</div>
</body>
</html>"""


def _render_branch_html(branch: TreeNode) -> str:
    state = '<span class="pill pill-ok">EXPANDED</span>' if branch.expanded else '<span class="pill pill-muted">COLLAPSED</span>'
    risks = "".join(f'<span class="pill pill-err">{_esc(t)}</span>' for t in branch.risk_tags)
    children_html = []
    for ep in branch.children:
        ep_state = '<span class="pill pill-ok">EXPANDED</span>' if ep.expanded else '<span class="pill pill-muted">COLLAPSED</span>'
        ep_risks = "".join(f'<span class="pill pill-warn">{_esc(t)}</span>' for t in ep.risk_tags)
        children_html.append(
            f"<div style='margin:6px 0 6px 14px;'>{ep_state} <strong>{_esc(ep.node_id)}</strong> "
            f"{_esc(ep.title)} ({ep.token_count} tok) {ep_risks}</div>"
        )
    open_attr = " open" if branch.expanded else ""
    return (
        f"<details{open_attr}><summary>{state} {_esc(branch.node_id)}: {_esc(branch.title)} "
        f"({branch.token_count} tok) {risks}</summary>"
        f"<p style='margin:8px 0 4px 0;color:#cbd5e1;'><strong>Verified Overview:</strong> {_esc(branch.overview)}</p>"
        f"{''.join(children_html)}</details>"
    )


def render_batch_report_html(report: BatchReport) -> str:
    """Render an interactive HTML report comparing compression methods and failure-mode clusters."""
    method_rows = []
    for name, agg in report.method_summaries.items():
        method_rows.append(
            f"<tr><td><strong>{_esc(name)}</strong></td>"
            f"<td>{agg.mean_tokens:.0f}</td>"
            f"<td>{agg.mean_savings_pct:.1f}%</td>"
            f"<td>{agg.mean_agreement * 100:.1f}%</td>"
            f"<td>{agg.mean_f1:.3f}</td>"
            f"<td>{agg.mean_false_pass_rate * 100:.1f}%</td>"
            f"<td>{agg.mean_js_vs_full:.4f}</td>"
            f"<td>${agg.total_judge_cost_usd:.5f}</td>"
            f"<td>${agg.total_compressor_cost_usd:.5f}</td></tr>"
        )

    failure_rows = []
    sorted_failures = sorted(report.failure_mode_counts.items(), key=lambda x: -x[1])
    for rid, cnt in sorted_failures:
        rate = report.failure_mode_rates.get(rid, 0.0) * 100.0
        rname = report.rubric_names.get(rid, rid)
        failure_rows.append(
            f"<tr><td>{_esc(rname)} (<code>{_esc(rid)}</code>)</td>"
            f"<td>{cnt} / {report.num_traces} ({rate:.1f}%)</td>"
            f"<td style='width:42%'><div class='bar-bg'><div class='bar-fill' style='width:{min(100.0, rate):.1f}%'></div></div></td></tr>"
        )

    trace_rows = []
    for row in report.per_trace:
        failed = ", ".join(row.get("failed_rubrics", [])) or "none"
        trace_rows.append(
            f"<tr><td><code>{_esc(row['trace_id'])}</code></td>"
            f"<td>{row['steps']}</td>"
            f"<td>{row['raw_tokens']} &rarr; {row['dossier_tokens']}</td>"
            f"<td>{row['cross_edges']}</td>"
            f"<td>{'Yes' if row['escalated'] else 'No'}</td>"
            f"<td>{_esc(failed)}</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>trace-tree-eval Batch Report ({report.num_traces} traces)</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>trace-tree-eval Batch Benchmark Report</h1>
  <div class="grid">
    <div class="kpi"><div class="label">Traces Evaluated</div><div class="val">{report.num_traces}</div></div>
    <div class="kpi"><div class="label">Target Budget</div><div class="val">{report.budget_tokens} tok</div></div>
    <div class="kpi"><div class="label">Jev Calls</div><div class="val">{report.compressor_usage.get('calls', 0)}</div></div>
    <div class="kpi"><div class="label">Jev Total Cost</div><div class="val">${float(report.compressor_usage.get('estimated_cost_usd', 0.0)):.5f}</div></div>
  </div>

  <div class="panel">
    <h2>Method Comparison</h2>
    <table>
      <thead>
        <tr><th>Method</th><th>Mean Tokens</th><th>Savings</th><th>Agreement</th><th>F1</th>
        <th>False-Pass Rate</th><th>JS Div vs Full</th><th>Judge Cost</th><th>Compressor Cost</th></tr>
      </thead>
      <tbody>{''.join(method_rows)}</tbody>
    </table>
  </div>

  <div class="panel">
    <h2>Failure-Mode Clustering (Rubric Failure Distribution)</h2>
    <table>
      <thead><tr><th>Rubric Dimension</th><th>Failures</th><th>Distribution</th></tr></thead>
      <tbody>{''.join(failure_rows)}</tbody>
    </table>
  </div>

  <div class="panel">
    <h2>Per-Trace Summary</h2>
    <table>
      <thead><tr><th>Trace ID</th><th>Steps</th><th>Raw &rarr; Dossier</th><th>Causal Edges</th><th>Escalated</th><th>Failed Rubrics</th></tr></thead>
      <tbody>{''.join(trace_rows)}</tbody>
    </table>
  </div>
</div>
</body>
</html>"""
