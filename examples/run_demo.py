"""End-to-end demo comparing FullTrace, HeadTail, LexicalRAG, and HybridTreeCompressor."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Ensure src/ and repo root are importable when run directly as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from examples.data.generate_synthetic_trace import build_synthetic_180_step_trace
from trace_tree_eval.backends import (
    HashingEmbeddingBackend,
    JevBackend,
    MockDecisionBackend,
    MockTextBackend,
    OpenAICompatTextBackend,
)
from trace_tree_eval.compression import CompressionConfig, HybridTreeCompressor
from trace_tree_eval.evaluation import (
    EvidenceFirstJudge,
    FullTraceBaseline,
    HeadTailBaseline,
    LexicalRagBaseline,
    compute_sufficiency_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trace-tree-eval 180-step comparison demo.")
    parser.add_argument("--budget", type=int, default=3500, help="Target compressed token budget.")
    parser.add_argument("--live", action="store_true", help="Use live Jev & OpenAI-compatible APIs.")
    args = parser.parse_args()

    console = Console()
    traj, rubrics = build_synthetic_180_step_trace()
    embedder = HashingEmbeddingBackend()

    if args.live:
        jev = JevBackend()
        overview_llm = OpenAICompatTextBackend(
            model=os.getenv("OVERVIEW_MODEL", "google/gemini-2.5-flash-lite"),
            price_in_per_1m=0.10,
            price_out_per_1m=0.40,
        )
        judge_llm = OpenAICompatTextBackend(
            model=os.getenv("JUDGE_MODEL", "anthropic/claude-sonnet-4"),
            price_in_per_1m=3.0,
            price_out_per_1m=15.0,
        )
    else:
        jev = MockDecisionBackend()
        overview_llm = MockTextBackend(price_in_per_1m=0.10, price_out_per_1m=0.40)
        judge_llm = MockTextBackend(price_in_per_1m=3.0, price_out_per_1m=15.0)

    compressor = HybridTreeCompressor(
        decision_backend=jev,
        text_backend=overview_llm,
        embedder=embedder,
        config=CompressionConfig(budget_tokens=args.budget),
    )
    judge = EvidenceFirstJudge(text_backend=judge_llm)

    raw_tokens = traj.total_tokens()
    gt = traj.ground_truth or {}

    full_ctx = FullTraceBaseline().build_context(traj)
    head_ctx = HeadTailBaseline().build_context(traj, budget_tokens=args.budget)
    rag_ctx = LexicalRagBaseline(embedder=embedder).build_context(traj, rubrics, budget_tokens=args.budget)
    dossier = compressor.compress(traj, rubrics, budget_tokens=args.budget)

    res_full = judge.evaluate_context(traj.trace_id, full_ctx, rubrics, "FullTrace")
    res_head = judge.evaluate_context(traj.trace_id, head_ctx, rubrics, "HeadTail")
    res_rag = judge.evaluate_context(traj.trace_id, rag_ctx, rubrics, "LexicalRAG")
    res_tree = judge.evaluate_dossier(traj, dossier, rubrics, compressor=compressor)

    table = Table(
        title=f"trace-tree-eval Benchmark (steps={len(traj.steps)}, raw_tokens={raw_tokens}, target_budget={args.budget})"
    )
    for col in (
        "Method",
        "Tokens",
        "Token Savings",
        "Agreement vs GT",
        "False-Pass Rate",
        "JS Div vs Full",
        "Judge Cost ($)",
    ):
        table.add_column(col)

    full_probs = res_full.probs_dict()
    for name, tok, ev in [
        ("FullTrace", raw_tokens, res_full),
        ("HeadTail", args.budget, res_head),
        ("LexicalRAG", args.budget, res_rag),
        ("HybridTree (Ours)", dossier.dossier_tokens, res_tree),
    ]:
        rep_gt = compute_sufficiency_report(gt, ev.verdicts_dict())
        rep_full = compute_sufficiency_report(
            res_full.verdicts_dict(), ev.verdicts_dict(), full_probs, ev.probs_dict()
        )
        savings = f"{(1.0 - tok / max(1, raw_tokens)) * 100:.1f}%"
        table.add_row(
            name,
            str(tok),
            savings,
            f"{rep_gt['agreement'] * 100:.1f}%",
            f"{rep_gt['false_pass_rate'] * 100:.1f}%",
            f"{rep_full['mean_js_divergence']:.4f}",
            f"${ev.estimated_cost_usd:.5f}",
        )

    console.print(table)
    console.print(f"\n[bold]HybridTree Cross-Branch Causal Edges:[/bold] {len(dossier.cross_edges)}")
    for e in dossier.cross_edges[:5]:
        console.print(f"  - step {e.source_step_id} -> step {e.target_step_id} (conf={e.confidence:.2f})")
    console.print(f"[bold]Expanded Nodes:[/bold] {dossier.expanded_node_ids}")
    console.print(f"[bold]Collapsed Nodes:[/bold] {len(dossier.collapsed_node_ids)}")
    console.print(f"[bold]Tier-2 Escalated:[/bold] {res_tree.escalated} (unfolded={res_tree.unfolded_nodes})")
    console.print(f"[bold]Jev Compressor Usage:[/bold] {jev.usage_stats()}")


if __name__ == "__main__":
    main()
