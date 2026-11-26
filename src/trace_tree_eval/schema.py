"""Core data schema for long-horizon trajectories, hybrid trees, dossiers, and judgments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from trace_tree_eval.tokens import estimate_tokens


@dataclass
class TraceStep:
    """Single step in an agent execution trajectory."""

    step_id: int
    role: str = "assistant"  # user / assistant / tool / system
    agent_id: str = "main"
    thought: str = ""
    action: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    observation: str = ""
    status: str = "ok"  # ok / error / warning
    metadata: dict[str, Any] = field(default_factory=dict)

    def render_text(self, include_header: bool = True) -> str:
        """Render the step into human- and judge-readable text."""
        parts: list[str] = []
        if include_header:
            tool_part = f" tool={self.tool_name}" if self.tool_name else ""
            parts.append(
                f"[step {self.step_id} | agent={self.agent_id} | role={self.role} | status={self.status}{tool_part}]"
            )
        if self.thought:
            parts.append(f"Thought: {self.thought}")
        if self.action:
            parts.append(f"Action: {self.action}")
        if self.tool_args:
            args_str = json.dumps(self.tool_args, ensure_ascii=False)
            parts.append(f"ToolArgs: {args_str}")
        if self.observation:
            parts.append(f"Observation: {self.observation}")
        return "\n".join(parts)

    def token_count(self) -> int:
        return estimate_tokens(self.render_text())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceStep:
        return cls(
            step_id=int(data["step_id"]),
            role=str(data.get("role", "assistant")),
            agent_id=str(data.get("agent_id", "main")),
            thought=str(data.get("thought", "")),
            action=str(data.get("action", "")),
            tool_name=data.get("tool_name"),
            tool_args=data.get("tool_args"),
            observation=str(data.get("observation", "")),
            status=str(data.get("status", "ok")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Trajectory:
    """Normalized long-horizon agent execution trajectory."""

    trace_id: str
    task_instruction: str
    deliverables: list[str] = field(default_factory=list)
    final_output: str = ""
    steps: list[TraceStep] = field(default_factory=list)
    ground_truth: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def total_tokens(self) -> int:
        base = estimate_tokens(self.task_instruction) + estimate_tokens(self.final_output)
        return base + sum(step.token_count() for step in self.steps)

    def step_map(self) -> dict[int, TraceStep]:
        return {step.step_id: step for step in self.steps}

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_instruction": self.task_instruction,
            "deliverables": list(self.deliverables),
            "final_output": self.final_output,
            "steps": [s.to_dict() for s in self.steps],
            "ground_truth": self.ground_truth,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        return cls(
            trace_id=str(data.get("trace_id", "unknown")),
            task_instruction=str(data.get("task_instruction", "")),
            deliverables=[str(x) for x in data.get("deliverables", [])],
            final_output=str(data.get("final_output", "")),
            steps=[TraceStep.from_dict(s) for s in data.get("steps", [])],
            ground_truth=data.get("ground_truth"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class RubricItem:
    """Strict boolean evaluation criterion aligned with Eugene Yan / Jev Noul primitives."""

    id: str
    name: str
    question: str
    weight: float = 1.0
    high_risk: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RubricItem:
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            question=str(data["question"]),
            weight=float(data.get("weight", 1.0)),
            high_risk=bool(data.get("high_risk", False)),
        )


@dataclass
class CrossBranchEdge:
    """Directed cross-branch causal/contradiction link from antecedent step to target step."""

    source_step_id: int
    target_step_id: int
    source_node_id: str
    target_node_id: str
    reason: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrossBranchEdge:
        return cls(
            source_step_id=int(data["source_step_id"]),
            target_step_id=int(data["target_step_id"]),
            source_node_id=str(data.get("source_node_id", "")),
            target_node_id=str(data.get("target_node_id", "")),
            reason=str(data.get("reason", "")),
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass
class TreeNode:
    """Hierarchical node in the hybrid execution tree."""

    node_id: str
    level: int  # 0=root, 1=phase/agent, 2=episode, 3=leaf
    title: str
    agent_id: str = "main"
    step_ids: list[int] = field(default_factory=list)
    children: list[TreeNode] = field(default_factory=list)
    overview: str = ""
    overview_verified: bool = False
    risk_tags: list[str] = field(default_factory=list)
    tool_counts: dict[str, int] = field(default_factory=dict)
    rubric_probs: dict[str, float] = field(default_factory=dict)
    critical_prob: float = 0.0
    expanded: bool = False
    pinned: bool = False
    token_count: int = 0

    def iter_nodes(self) -> list[TreeNode]:
        result = [self]
        for child in self.children:
            result.extend(child.iter_nodes())
        return result

    def find_node(self, node_id: str) -> TreeNode | None:
        if self.node_id == node_id:
            return self
        for child in self.children:
            found = child.find_node(node_id)
            if found is not None:
                return found
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "level": self.level,
            "title": self.title,
            "agent_id": self.agent_id,
            "step_ids": list(self.step_ids),
            "children": [c.to_dict() for c in self.children],
            "overview": self.overview,
            "overview_verified": self.overview_verified,
            "risk_tags": list(self.risk_tags),
            "tool_counts": dict(self.tool_counts),
            "rubric_probs": dict(self.rubric_probs),
            "critical_prob": self.critical_prob,
            "expanded": self.expanded,
            "pinned": self.pinned,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TreeNode:
        return cls(
            node_id=str(data["node_id"]),
            level=int(data.get("level", 0)),
            title=str(data.get("title", "")),
            agent_id=str(data.get("agent_id", "main")),
            step_ids=[int(x) for x in data.get("step_ids", [])],
            children=[cls.from_dict(c) for c in data.get("children", [])],
            overview=str(data.get("overview", "")),
            overview_verified=bool(data.get("overview_verified", False)),
            risk_tags=[str(x) for x in data.get("risk_tags", [])],
            tool_counts={str(k): int(v) for k, v in data.get("tool_counts", {}).items()},
            rubric_probs={str(k): float(v) for k, v in data.get("rubric_probs", {}).items()},
            critical_prob=float(data.get("critical_prob", 0.0)),
            expanded=bool(data.get("expanded", False)),
            pinned=bool(data.get("pinned", False)),
            token_count=int(data.get("token_count", 0)),
        )


@dataclass
class GlobalManifest:
    """Deterministic global execution summary to disambiguate errors of omission."""

    total_steps: int
    raw_tokens: int
    cleaned_tokens: int
    tool_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    verification_detected: bool = False
    verification_steps: list[int] = field(default_factory=list)
    mid_task_constraint_steps: list[int] = field(default_factory=list)
    loop_folds: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GlobalManifest:
        return cls(
            total_steps=int(data.get("total_steps", 0)),
            raw_tokens=int(data.get("raw_tokens", 0)),
            cleaned_tokens=int(data.get("cleaned_tokens", 0)),
            tool_stats=dict(data.get("tool_stats", {})),
            verification_detected=bool(data.get("verification_detected", False)),
            verification_steps=[int(x) for x in data.get("verification_steps", [])],
            mid_task_constraint_steps=[int(x) for x in data.get("mid_task_constraint_steps", [])],
            loop_folds=list(data.get("loop_folds", [])),
        )


@dataclass
class EvalDossier:
    """Structured evaluation dossier produced by HybridTreeCompressor."""

    trace_id: str
    budget_tokens: int
    dossier_tokens: int
    compression_ratio: float
    rendered_markdown: str
    manifest: GlobalManifest
    expanded_node_ids: list[str] = field(default_factory=list)
    collapsed_node_ids: list[str] = field(default_factory=list)
    selected_step_ids: list[int] = field(default_factory=list)
    pinned_step_ids: list[int] = field(default_factory=list)
    cross_edges: list[CrossBranchEdge] = field(default_factory=list)
    rubric_coverage: dict[str, float] = field(default_factory=dict)
    tree_root: TreeNode | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "budget_tokens": self.budget_tokens,
            "dossier_tokens": self.dossier_tokens,
            "compression_ratio": self.compression_ratio,
            "rendered_markdown": self.rendered_markdown,
            "manifest": self.manifest.to_dict(),
            "expanded_node_ids": list(self.expanded_node_ids),
            "collapsed_node_ids": list(self.collapsed_node_ids),
            "selected_step_ids": list(self.selected_step_ids),
            "pinned_step_ids": list(self.pinned_step_ids),
            "cross_edges": [e.to_dict() for e in self.cross_edges],
            "rubric_coverage": dict(self.rubric_coverage),
            "tree_root": self.tree_root.to_dict() if self.tree_root else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalDossier:
        tree_data = data.get("tree_root")
        return cls(
            trace_id=str(data["trace_id"]),
            budget_tokens=int(data.get("budget_tokens", 0)),
            dossier_tokens=int(data.get("dossier_tokens", 0)),
            compression_ratio=float(data.get("compression_ratio", 0.0)),
            rendered_markdown=str(data.get("rendered_markdown", "")),
            manifest=GlobalManifest.from_dict(data.get("manifest", {})),
            expanded_node_ids=[str(x) for x in data.get("expanded_node_ids", [])],
            collapsed_node_ids=[str(x) for x in data.get("collapsed_node_ids", [])],
            selected_step_ids=[int(x) for x in data.get("selected_step_ids", [])],
            pinned_step_ids=[int(x) for x in data.get("pinned_step_ids", [])],
            cross_edges=[CrossBranchEdge.from_dict(e) for e in data.get("cross_edges", [])],
            rubric_coverage={str(k): float(v) for k, v in data.get("rubric_coverage", {}).items()},
            tree_root=TreeNode.from_dict(tree_data) if tree_data else None,
        )


@dataclass
class RubricJudgment:
    """Evidence-first judgment for a single boolean rubric item."""

    rubric_id: str
    verdict: bool
    prob_true: float
    confidence: float
    cited_step_ids: list[int] = field(default_factory=list)
    evidence_sufficient: bool = True
    requested_unfold_nodes: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeEvaluation:
    """Aggregated evaluation output for a single trajectory."""

    trace_id: str
    method_name: str
    judgments: dict[str, RubricJudgment] = field(default_factory=dict)
    escalated: bool = False
    unfolded_nodes: list[str] = field(default_factory=list)
    total_input_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0

    def verdicts_dict(self) -> dict[str, bool]:
        return {rid: j.verdict for rid, j in self.judgments.items()}

    def probs_dict(self) -> dict[str, float]:
        return {rid: j.prob_true for rid, j in self.judgments.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "method_name": self.method_name,
            "judgments": {k: v.to_dict() for k, v in self.judgments.items()},
            "escalated": self.escalated,
            "unfolded_nodes": list(self.unfolded_nodes),
            "total_input_tokens": self.total_input_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "latency_ms": self.latency_ms,
        }
