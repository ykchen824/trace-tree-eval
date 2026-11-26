"""Stage 1: Semantic topic segmentation tree builder with risk bubbling and SJC overview verification."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from trace_tree_eval.backends.base import DecisionBackend, EmbeddingBackend, TextModelBackend
from trace_tree_eval.schema import TraceStep, TreeNode
from trace_tree_eval.tokens import estimate_tokens

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+|\n+")


@dataclass
class TreeBuildConfig:
    """Configuration for deterministic/semantic hybrid tree construction."""

    min_branch_steps: int = 4
    max_branch_tokens: int = 12000  # Keep safely under Jev's 32K/64K Danger-Zone
    max_episode_tokens: int = 3500
    boundary_std_factor: float = 0.85
    overview_grounding_threshold: float = 0.65
    generate_overviews: bool = True


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _group_by_agent_runs(steps: list[TraceStep]) -> list[tuple[str, list[TraceStep]]]:
    """Preserve temporal order while separating contiguous sub-agent runs (Era 3 support)."""
    if not steps:
        return []
    runs: list[tuple[str, list[TraceStep]]] = []
    cur_agent = steps[0].agent_id
    cur_buf: list[TraceStep] = [steps[0]]
    for s in steps[1:]:
        if s.agent_id != cur_agent:
            runs.append((cur_agent, cur_buf))
            cur_agent = s.agent_id
            cur_buf = [s]
        else:
            cur_buf.append(s)
    runs.append((cur_agent, cur_buf))
    return runs


def _segment_run_semantically(
    steps: list[TraceStep],
    embedder: EmbeddingBackend,
    min_steps: int,
    max_tokens: int,
    std_factor: float,
) -> list[list[TraceStep]]:
    """Split a contiguous step sequence into coherent topic branches using cosine similarity drops."""
    if len(steps) <= min_steps:
        return [steps]

    texts = [s.render_text(include_header=False) or s.role for s in steps]
    vecs = embedder.embed(texts)
    adj_sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
    threshold = float(np.mean(adj_sims) - std_factor * np.std(adj_sims))

    branches: list[list[TraceStep]] = []
    cur_group: list[TraceStep] = [steps[0]]
    cur_tokens = steps[0].token_count()

    for idx in range(1, len(steps)):
        s = steps[idx]
        st = s.token_count()
        sim = float(adj_sims[idx - 1])
        exceeds_budget = (cur_tokens + st > max_tokens) and len(cur_group) >= 1
        topic_shift = (sim < threshold) and (len(cur_group) >= min_steps)
        if exceeds_budget or topic_shift:
            branches.append(cur_group)
            cur_group = [s]
            cur_tokens = st
        else:
            cur_group.append(s)
            cur_tokens += st

    if cur_group:
        branches.append(cur_group)
    return branches


def _chunk_episodes(steps: list[TraceStep], max_episode_tokens: int) -> list[list[TraceStep]]:
    episodes: list[list[TraceStep]] = []
    buf: list[TraceStep] = []
    tok = 0
    for s in steps:
        st = s.token_count()
        if buf and (tok + st > max_episode_tokens or len(buf) >= 6):
            episodes.append(buf)
            buf = [s]
            tok = st
        else:
            buf.append(s)
            tok += st
    if buf:
        episodes.append(buf)
    return episodes


def _bubble_metadata_upward(node: TreeNode) -> None:
    """Propagate descendant risk_tags and tool_counts upward so high-risk leaves cannot hide."""
    if not node.children:
        return
    merged_tags = set(node.risk_tags)
    merged_tools = dict(node.tool_counts)
    merged_steps: list[int] = []
    total_tok = 0
    for child in node.children:
        _bubble_metadata_upward(child)
        merged_tags.update(child.risk_tags)
        for k, v in child.tool_counts.items():
            merged_tools[k] = merged_tools.get(k, 0) + v
        merged_steps.extend(child.step_ids)
        total_tok += child.token_count

    node.risk_tags = sorted(merged_tags)
    node.tool_counts = merged_tools
    if not node.step_ids:
        node.step_ids = merged_steps
    node.token_count = total_tok


def _summarize_tools(tool_counts: dict[str, int]) -> str:
    if not tool_counts:
        return "no tool calls"
    top = sorted(tool_counts.items(), key=lambda x: -x[1])[:4]
    return ", ".join(f"{k}:{v}" for k, v in top)


def _verify_and_attach_overviews(
    branches: list[TreeNode],
    step_lookup: dict[int, TraceStep],
    text_backend: TextModelBackend | None,
    decision_backend: DecisionBackend | None,
    threshold: float,
) -> None:
    """Generate short branch overviews and apply SJC-style sentence grounding pruning via Jev."""
    system_prompt = (
        "You are a Branch Overview Generator ([OVERVIEW_TASK]). "
        "Write 2 concise factual sentences summarizing what happened in these agent steps. "
        "Only state facts directly observed in the text."
    )

    verify_batch: list[tuple[str, dict[str, str]]] = []
    branch_drafts: list[tuple[TreeNode, list[str]]] = []

    for branch in branches:
        branch_text = "\n".join(step_lookup[sid].render_text() for sid in branch.step_ids if sid in step_lookup)
        if text_backend is None:
            branch.overview = (
                f"Steps {branch.step_ids[0]}-{branch.step_ids[-1]} ({branch.agent_id}): "
                f"{_summarize_tools(branch.tool_counts)}."
            )
            branch.overview_verified = True
            continue

        raw_overview = text_backend.complete(system_prompt, branch_text[:12000], json_mode=False)
        sentences = _split_sentences(raw_overview)
        if decision_backend is None or not sentences:
            branch.overview = raw_overview.strip()
            branch.overview_verified = False
            continue

        questions = {
            f"grounding_{idx}": (
                "Is the following summary claim strictly supported by facts in the branch log "
                f"without hallucination? Claim: {sent}"
            )
            for idx, sent in enumerate(sentences)
        }
        verify_batch.append((branch_text, questions))
        branch_drafts.append((branch, sentences))

    if verify_batch and decision_backend is not None:
        results = decision_backend.batch_score_noul(verify_batch)
        for (branch, sentences), probs in zip(branch_drafts, results):
            kept = [
                sent
                for idx, sent in enumerate(sentences)
                if probs.get(f"grounding_{idx}", 0.0) >= threshold
            ]
            if kept:
                branch.overview = " ".join(kept)
            else:
                branch.overview = (
                    f"Steps {branch.step_ids[0]}-{branch.step_ids[-1]} ({branch.agent_id}): "
                    f"{_summarize_tools(branch.tool_counts)} [unverified prose pruned]."
                )
            branch.overview_verified = True


def build_hybrid_tree(
    trace_id: str,
    cleaned_steps: list[TraceStep],
    step_risk_tags: dict[int, list[str]],
    embedder: EmbeddingBackend,
    text_backend: TextModelBackend | None = None,
    decision_backend: DecisionBackend | None = None,
    config: TreeBuildConfig | None = None,
) -> TreeNode:
    """Build a 3-level hybrid execution tree with risk bubbling and Jev-verified overviews."""
    cfg = config or TreeBuildConfig()
    step_lookup = {s.step_id: s for s in cleaned_steps}

    root = TreeNode(
        node_id="root",
        level=0,
        title=f"Trajectory {trace_id}",
        agent_id="main",
        step_ids=[s.step_id for s in cleaned_steps],
    )

    if not cleaned_steps:
        return root

    agent_runs = _group_by_agent_runs(cleaned_steps)
    l1_nodes: list[TreeNode] = []
    l2_nodes: list[TreeNode] = []

    b_idx = 0
    for agent_id, run_steps in agent_runs:
        phase_groups = _segment_run_semantically(
            run_steps,
            embedder=embedder,
            min_steps=cfg.min_branch_steps,
            max_tokens=cfg.max_branch_tokens,
            std_factor=cfg.boundary_std_factor,
        )
        for p_steps in phase_groups:
            b_idx += 1
            b_id = f"b{b_idx}"
            episodes = _chunk_episodes(p_steps, cfg.max_episode_tokens)
            ep_nodes: list[TreeNode] = []

            for e_idx, ep_steps in enumerate(episodes, start=1):
                e_id = f"{b_id}_e{e_idx}"
                leaf_nodes: list[TreeNode] = []
                for s in ep_steps:
                    tname = s.tool_name or (f"role:{s.role}" if s.role != "assistant" else "reasoning")
                    l_node = TreeNode(
                        node_id=f"step_{s.step_id}",
                        level=3,
                        title=f"Step {s.step_id} ({tname})",
                        agent_id=s.agent_id,
                        step_ids=[s.step_id],
                        risk_tags=list(step_risk_tags.get(s.step_id, [])),
                        tool_counts={tname: 1},
                        token_count=s.token_count(),
                    )
                    leaf_nodes.append(l_node)

                ep_node = TreeNode(
                    node_id=e_id,
                    level=2,
                    title=f"Episode {e_id} (steps {ep_steps[0].step_id}-{ep_steps[-1].step_id})",
                    agent_id=agent_id,
                    step_ids=[s.step_id for s in ep_steps],
                    children=leaf_nodes,
                )
                ep_nodes.append(ep_node)
                l2_nodes.append(ep_node)

            b_node = TreeNode(
                node_id=b_id,
                level=1,
                title=f"Branch {b_idx} [{agent_id}] (steps {p_steps[0].step_id}-{p_steps[-1].step_id})",
                agent_id=agent_id,
                step_ids=[s.step_id for s in p_steps],
                children=ep_nodes,
            )
            l1_nodes.append(b_node)

    root.children = l1_nodes
    _bubble_metadata_upward(root)

    if cfg.generate_overviews:
        _verify_and_attach_overviews(
            branches=l1_nodes,
            step_lookup=step_lookup,
            text_backend=text_backend,
            decision_backend=decision_backend,
            threshold=cfg.overview_grounding_threshold,
        )
        for ep in l2_nodes:
            ep.overview = (
                f"Steps {ep.step_ids[0]}-{ep.step_ids[-1]} ({ep.agent_id}): "
                f"{_summarize_tools(ep.tool_counts)}."
            )
            ep.overview_verified = True

    root.overview = (
        f"Total {len(cleaned_steps)} cleaned steps across {len(l1_nodes)} branches; "
        f"tokens={estimate_tokens(' '.join(s.render_text() for s in cleaned_steps))}."
    )
    root.overview_verified = True
    return root
