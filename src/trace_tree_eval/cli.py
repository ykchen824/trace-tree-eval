"""Typer + Rich CLI for trace-tree-eval."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from trace_tree_eval.api import build_runtime_backends, load_traces
from trace_tree_eval.compression import CompressionConfig, HybridTreeCompressor
from trace_tree_eval.config import env_summary, load_dotenv
from trace_tree_eval.evaluation import (
    BatchEvaluator,
    EvidenceFirstJudge,
    FullTraceBaseline,
    HeadTailBaseline,
    LexicalRagBaseline,
    compute_sufficiency_report,
)
from trace_tree_eval.presets import get_preset_rubric, list_preset_rubrics, resolve_rubrics
from trace_tree_eval.report_html import render_batch_report_html, render_dossier_html
from trace_tree_eval.schema import EvalDossier, Trajectory, TreeNode

app = typer.Typer(
    name="trace-tree-eval",
    help="Hybrid tree compaction and Evidence-First evaluation for long-horizon agent traces.",
    no_args_is_help=True,
)
console = Console()


def _load_synthetic_trace() -> tuple[Trajectory, list]:
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from examples.data.generate_synthetic_trace import build_synthetic_180_step_trace

    return build_synthetic_180_step_trace()


def _select_trace(trajectories: list[Trajectory], trace_id: str | None) -> Trajectory:
    if not trajectories:
        console.print("[red]No trajectories found in source.[/red]")
        raise typer.Exit(code=1)
    if trace_id is None:
        return trajectories[0]
    for t in trajectories:
        if t.trace_id == trace_id:
            return t
    console.print(f"[red]Trace ID '{trace_id}' not found.[/red]")
    raise typer.Exit(code=1)


@app.callback()
def _startup() -> None:
    """Automatically load .env when running any CLI command."""
    load_dotenv()


@app.command()
def doctor(
    ping: bool = typer.Option(False, "--ping", help="Also run a live API connectivity probe."),
) -> None:
    """Check Python environment, optional packages, and masked .env configuration."""
    load_dotenv()
    pkg_table = Table(title="Environment & Optional Dependencies")
    pkg_table.add_column("Component")
    pkg_table.add_column("Status")
    pkg_table.add_column("Note")

    pkg_table.add_row("Python", sys.version.split()[0], "Requires >= 3.10")
    for pkg, purpose in [
        ("numpy", "Core math & feature hashing"),
        ("httpx", "Core HTTP client for Jev / OpenAI-compat"),
        ("typer", "Core CLI"),
        ("rich", "Core terminal formatting"),
        ("datasets", "Optional: load Hugging Face datasets"),
        ("typesafe_sdk", "Optional: official TypeSafe SDK (httpx fallback ready)"),
        ("tiktoken", "Optional: exact cl100k_base token counting"),
        ("sentence_transformers", "Optional: local dense embeddings"),
    ]:
        ok = importlib.util.find_spec(pkg) is not None
        status = "[green]installed[/green]" if ok else "[yellow]missing[/yellow]"
        pkg_table.add_row(pkg, status, purpose)
    console.print(pkg_table)

    env_table = Table(title=".env & API Configuration (Secrets Masked)")
    env_table.add_column("Key")
    env_table.add_column("Value")
    for k, v in env_summary().items():
        env_table.add_row(k, str(v))
    console.print(env_table)

    if ping:
        try:
            jev, ov_llm, judge_llm, _ = build_runtime_backends(live=True)
            for label, backend in [("Jev", jev), ("Overview LLM", ov_llm), ("Judge LLM", judge_llm)]:
                ok, msg = backend.ping()
                tag = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
                console.print(f"{tag} {label}: {msg}")
        except Exception as exc:
            console.print(f"[red]Live ping failed:[/red] {exc}")
            raise typer.Exit(code=1)


@app.command()
def presets() -> None:
    """List built-in strict boolean rubric presets."""
    table = Table(title="Built-in Boolean Rubric Presets")
    table.add_column("Preset Name")
    table.add_column("Criteria Count")
    table.add_column("Dimensions")
    for name in list_preset_rubrics():
        items = get_preset_rubric(name)
        dims = ", ".join(f"{it.id}{'*' if it.high_risk else ''}" for it in items)
        table.add_row(f"preset:{name}", str(len(items)), dims)
    console.print(table)
    console.print("[dim]* indicates high_risk=True (always scanned at episode level).[/dim]")


@app.command()
def demo(
    budget: int = typer.Option(3500, help="Target dossier token budget."),
    live: bool = typer.Option(False, "--live", help="Use live Jev & OpenAI-compatible APIs from .env."),
    html_out: Path | None = typer.Option(None, "--html-out", help="Optional path to save interactive HTML viewer."),
    cache_dir: Path | None = typer.Option(None, "--cache-dir", help="Optional disk cache directory."),
) -> None:
    """Run the built-in 180-step unstructured trace benchmark comparing baselines vs HybridTree."""
    traj, rubrics = _load_synthetic_trace()
    jev, overview_llm, judge_llm, embedder = build_runtime_backends(live=live, cache_dir=cache_dir)
    compressor = HybridTreeCompressor(
        decision_backend=jev,
        text_backend=overview_llm,
        embedder=embedder,
        config=CompressionConfig(budget_tokens=budget),
    )
    judge = EvidenceFirstJudge(text_backend=judge_llm)

    gt = traj.ground_truth or {}
    raw_tokens = traj.total_tokens()

    full_ctx = FullTraceBaseline().build_context(traj)
    head_ctx = HeadTailBaseline().build_context(traj, budget_tokens=budget)
    rag_ctx = LexicalRagBaseline(embedder=embedder).build_context(traj, rubrics, budget_tokens=budget)
    dossier = compressor.compress(traj, rubrics, budget_tokens=budget)

    res_full = judge.evaluate_context(traj.trace_id, full_ctx, rubrics, "FullTrace")
    res_head = judge.evaluate_context(traj.trace_id, head_ctx, rubrics, "HeadTail")
    res_rag = judge.evaluate_context(traj.trace_id, rag_ctx, rubrics, "LexicalRAG")
    res_tree = judge.evaluate_dossier(traj, dossier, rubrics, compressor=compressor)

    table = Table(
        title=f"trace-tree-eval Demo (trace={traj.trace_id}, steps={len(traj.steps)}, raw_tokens={raw_tokens})"
    )
    table.add_column("Method", no_wrap=True)
    for col in ("Input Tokens", "Compression", "Agreement vs GT", "False-Pass Rate", "JS Div vs Full", "Est. Cost ($)"):
        table.add_column(col)

    full_probs = res_full.probs_dict()
    for label, tok_cnt, eval_res in [
        ("FullTrace", raw_tokens, res_full),
        ("HeadTail", budget, res_head),
        ("LexicalRAG", budget, res_rag),
        ("HybridTree (Ours)", dossier.dossier_tokens, res_tree),
    ]:
        rep_gt = compute_sufficiency_report(gt, eval_res.verdicts_dict())
        rep_full = compute_sufficiency_report(
            res_full.verdicts_dict(), eval_res.verdicts_dict(), full_probs, eval_res.probs_dict()
        )
        comp_pct = f"{(1.0 - tok_cnt / max(1, raw_tokens)) * 100:.1f}%"
        table.add_row(
            label,
            str(tok_cnt),
            comp_pct,
            f"{rep_gt['agreement'] * 100:.1f}%",
            f"{rep_gt['false_pass_rate'] * 100:.1f}%",
            f"{rep_full['mean_js_divergence']:.4f}",
            f"${eval_res.estimated_cost_usd:.5f}",
        )
    console.print(table)
    console.print(f"[dim]Jev usage: {jev.usage_stats()}[/dim]")

    if html_out is not None:
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.write_text(render_dossier_html(dossier, res_tree, rubrics), encoding="utf-8")
        console.print(f"Saved interactive HTML viewer -> {html_out}")


@app.command()
def compress(
    trace_path: Path = typer.Argument(..., exists=True, help="Path to ATIF/JSON/JSONL trace file."),
    rubric: str = typer.Option("preset:deep_research", "--rubric", "-r", help="Rubric JSON path or preset name."),
    budget: int = typer.Option(6000, "--budget", "-b", help="Target token budget."),
    out: Path = typer.Option(Path("dossier.md"), "--out", "-o", help="Output Markdown file (or directory for batch JSONL)."),
    json_out: Path | None = typer.Option(None, "--json-out", help="Optional JSON dossier metadata path."),
    html_out: Path | None = typer.Option(None, "--html-out", help="Optional interactive HTML viewer path."),
    trace_id: str | None = typer.Option(None, "--trace-id", help="Specific trace_id to compress from multi-trace file."),
    cache_dir: Path | None = typer.Option(None, "--cache-dir", help="Optional disk cache directory for live APIs."),
    live: bool = typer.Option(False, "--live", help="Use live Jev & OpenAI-compatible APIs."),
) -> None:
    """Compress one or more traces into structured EvalDossier files."""
    trajectories = load_traces(trace_path)
    if not trajectories:
        raise typer.Exit(code=1)

    rubrics = resolve_rubrics(rubric)
    jev, overview_llm, _, embedder = build_runtime_backends(live=live, cache_dir=cache_dir)
    compressor = HybridTreeCompressor(
        decision_backend=jev,
        text_backend=overview_llm,
        embedder=embedder,
        config=CompressionConfig(budget_tokens=budget),
    )

    batch_mode = len(trajectories) > 1 and trace_id is None and (out.is_dir() or out.suffix == "")
    if batch_mode:
        out.mkdir(parents=True, exist_ok=True)
        for traj in trajectories:
            dossier = compressor.compress(traj, rubrics, budget_tokens=budget)
            md_file = out / f"{traj.trace_id}.md"
            md_file.write_text(dossier.rendered_markdown, encoding="utf-8")
            if html_out is not None:
                html_out.mkdir(parents=True, exist_ok=True)
                (html_out / f"{traj.trace_id}.html").write_text(
                    render_dossier_html(dossier, rubrics=rubrics), encoding="utf-8"
                )
        console.print(f"Batch compressed {len(trajectories)} traces into directory: {out}")
        return

    traj = _select_trace(trajectories, trace_id)
    dossier = compressor.compress(traj, rubrics, budget_tokens=budget)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dossier.rendered_markdown, encoding="utf-8")
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(dossier.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    if html_out is not None:
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.write_text(render_dossier_html(dossier, rubrics=rubrics), encoding="utf-8")
    console.print(
        f"Compressed [bold]{traj.trace_id}[/bold]: "
        f"{dossier.manifest.raw_tokens} -> {dossier.dossier_tokens} tokens "
        f"(ratio={dossier.compression_ratio:.2%}) -> {out}"
    )


@app.command()
def evaluate(
    trace_path: Path = typer.Argument(..., exists=True, help="Path to ATIF/JSON/JSONL trace file."),
    rubric: str = typer.Option("preset:deep_research", "--rubric", "-r", help="Rubric JSON path or preset name."),
    budget: int = typer.Option(6000, "--budget", "-b", help="Target token budget."),
    trace_id: str | None = typer.Option(None, "--trace-id", help="Specific trace_id to evaluate from multi-trace file."),
    json_out: Path | None = typer.Option(None, "--json-out", help="Optional JSON output path."),
    html_out: Path | None = typer.Option(None, "--html-out", help="Optional interactive HTML viewer path."),
    cache_dir: Path | None = typer.Option(None, "--cache-dir", help="Optional disk cache directory for live APIs."),
    live: bool = typer.Option(False, "--live", help="Use live Jev & OpenAI-compatible APIs."),
) -> None:
    """Compress and run Evidence-First Judge (with Tier-2 unfolding)."""
    trajectories = load_traces(trace_path)
    traj = _select_trace(trajectories, trace_id)
    rubrics = resolve_rubrics(rubric)

    jev, overview_llm, judge_llm, embedder = build_runtime_backends(live=live, cache_dir=cache_dir)
    compressor = HybridTreeCompressor(
        decision_backend=jev,
        text_backend=overview_llm,
        embedder=embedder,
        config=CompressionConfig(budget_tokens=budget),
    )
    dossier = compressor.compress(traj, rubrics, budget_tokens=budget)
    judge = EvidenceFirstJudge(text_backend=judge_llm)
    res = judge.evaluate_dossier(traj, dossier, rubrics, compressor=compressor)

    table = Table(title=f"Evaluation Result: {traj.trace_id} (escalated={res.escalated})")
    table.add_column("Rubric")
    table.add_column("Verdict")
    table.add_column("Prob")
    table.add_column("Cited Steps")
    table.add_column("Rationale")
    for r in rubrics:
        j = res.judgments[r.id]
        table.add_row(
            r.name,
            "[green]PASS[/green]" if j.verdict else "[red]FAIL[/red]",
            f"{j.prob_true:.2f}",
            str(j.cited_step_ids),
            j.rationale,
        )
    console.print(table)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps({"dossier": dossier.to_dict(), "evaluation": res.to_dict()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if html_out is not None:
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.write_text(render_dossier_html(dossier, res, rubrics), encoding="utf-8")
        console.print(f"Saved interactive HTML report -> {html_out}")


@app.command()
def benchmark(
    source: str | None = typer.Argument(None, help="Local JSON/JSONL path or Hugging Face dataset repo (omit for synthetic)."),
    dataset: str = typer.Option("auto", "--dataset", "-d", help="Dataset format: auto | atif | agent-reward-bench | deep-research | synthetic."),
    rubric: str = typer.Option("preset:deep_research", "--rubric", "-r", help="Rubric JSON path or preset name."),
    budget: int = typer.Option(3500, "--budget", "-b", help="Target dossier token budget."),
    split: str = typer.Option("test", help="Dataset split when loading from Hugging Face."),
    min_steps: int = typer.Option(1, "--min-steps", help="Filter out traces shorter than N steps."),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum number of traces to evaluate."),
    no_baselines: bool = typer.Option(False, "--no-baselines", help="Run only HybridTree without baselines."),
    html_out: Path | None = typer.Option(None, "--html-out", help="Save interactive HTML batch report."),
    json_out: Path | None = typer.Option(None, "--json-out", help="Save JSON batch report."),
    cache_dir: Path | None = typer.Option(None, "--cache-dir", help="Optional disk cache directory for live APIs."),
    live: bool = typer.Option(False, "--live", help="Use live Jev & OpenAI-compatible APIs."),
) -> None:
    """Run batch evaluation & failure-mode clustering over local/HF datasets or synthetic trace."""
    if dataset == "synthetic" or source is None:
        traj, default_rubrics = _load_synthetic_trace()
        trajectories = [traj]
        rubrics = default_rubrics if rubric == "preset:deep_research" else resolve_rubrics(rubric)
    else:
        trajectories = load_traces(
            source, dataset_format=dataset, split=split, min_steps=min_steps, limit=limit
        )
        rubrics = resolve_rubrics(rubric)

    if not trajectories:
        console.print("[red]No trajectories matched the dataset/filter.[/red]")
        raise typer.Exit(code=1)

    jev, overview_llm, judge_llm, embedder = build_runtime_backends(live=live, cache_dir=cache_dir)
    runner = BatchEvaluator(
        decision_backend=jev,
        overview_backend=overview_llm,
        judge_backend=judge_llm,
        embedder=embedder,
        compression_config=CompressionConfig(budget_tokens=budget),
    )

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task(f"Evaluating {len(trajectories)} trace(s)...", total=len(trajectories))

        def _on_prog(idx: int, total: int, tid: str) -> None:
            progress.update(task, completed=idx - 1, description=f"[{idx}/{total}] {tid}")

        report = runner.run(
            trajectories=trajectories,
            rubrics=rubrics,
            budget_tokens=budget,
            include_baselines=not no_baselines,
            on_progress=_on_prog,
        )
        progress.update(task, completed=len(trajectories), description="Done")

    table = Table(title=f"Batch Benchmark Summary (traces={report.num_traces}, budget={budget})")
    table.add_column("Method", no_wrap=True)
    for col in ("Mean Tokens", "Savings", "Agreement", "False-Pass Rate", "JS Div vs Full", "Judge Cost ($)", "Jev Cost ($)"):
        table.add_column(col)
    for name, agg in report.method_summaries.items():
        table.add_row(
            name,
            f"{agg.mean_tokens:.0f}",
            f"{agg.mean_savings_pct:.1f}%",
            f"{agg.mean_agreement * 100:.1f}%",
            f"{agg.mean_false_pass_rate * 100:.1f}%",
            f"{agg.mean_js_vs_full:.4f}",
            f"${agg.total_judge_cost_usd:.5f}",
            f"${agg.total_compressor_cost_usd:.5f}",
        )
    console.print(table)

    fail_table = Table(title="Failure-Mode Clustering (Rubric Failures Across Batch)")
    fail_table.add_column("Rubric")
    fail_table.add_column("Failed Count")
    fail_table.add_column("Failure Rate")
    for rid, cnt in sorted(report.failure_mode_counts.items(), key=lambda x: -x[1]):
        fail_table.add_row(
            f"{report.rubric_names.get(rid, rid)} ({rid})",
            f"{cnt}/{report.num_traces}",
            f"{report.failure_mode_rates.get(rid, 0.0) * 100:.1f}%",
        )
    console.print(fail_table)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"Saved JSON report -> {json_out}")
    if html_out is not None:
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.write_text(render_batch_report_html(report), encoding="utf-8")
        console.print(f"Saved interactive HTML report -> {html_out}")


@app.command()
def inspect(
    dossier_json: Path = typer.Argument(..., exists=True, help="Path to saved dossier JSON."),
) -> None:
    """Visualize the collapsed/expanded hybrid tree structure of a compressed dossier."""
    data = json.loads(dossier_json.read_text(encoding="utf-8"))
    if "dossier" in data and isinstance(data["dossier"], dict):
        data = data["dossier"]
    dossier = EvalDossier.from_dict(data)
    if dossier.tree_root is None:
        console.print("[yellow]No tree_root stored in dossier JSON.[/yellow]")
        raise typer.Exit()

    def _add_children(rich_node: Tree, t_node: TreeNode) -> None:
        state_tag = "[green]EXPANDED[/green]" if t_node.expanded else "[dim]COLLAPSED[/dim]"
        risk = f" risks={t_node.risk_tags}" if t_node.risk_tags else ""
        label = f"{state_tag} {t_node.node_id}: {t_node.title} ({t_node.token_count} tok){risk}"
        branch = rich_node.add(label)
        if t_node.level < 2:
            for ch in t_node.children:
                _add_children(branch, ch)

    root_tree = Tree(f"[bold]{dossier.trace_id}[/bold] ({dossier.dossier_tokens}/{dossier.manifest.raw_tokens} tokens)")
    for b in dossier.tree_root.children:
        _add_children(root_tree, b)
    console.print(root_tree)


if __name__ == "__main__":
    app()
