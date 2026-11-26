"""Adapters for public long-horizon evaluation datasets (AgentRewardBench, DeepResearch-traj)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trace_tree_eval.adapters.atif import ATIFAdapter
from trace_tree_eval.schema import TraceStep, Trajectory

try:
    from datasets import load_dataset  # type: ignore
except ImportError:  # pragma: no cover
    load_dataset = None


def _load_records(
    source: str | Path | list[dict[str, Any]] | dict[str, Any],
    split: str = "train",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if isinstance(source, dict):
        return [source]
    if isinstance(source, list):
        return list(source[:limit] if limit else source)

    path = Path(source)
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            parsed = json.loads(text)
            rows = parsed if isinstance(parsed, list) else [parsed]
        return rows[:limit] if limit else rows

    if load_dataset is None:
        raise ImportError(
            "Loading remote Hugging Face datasets requires `datasets` (`pip install datasets`)."
        )
    ds = load_dataset(str(source), split=split)
    rows: list[dict[str, Any]] = []
    for row in ds:
        rows.append(dict(row))
        if limit is not None and len(rows) >= limit:
            break
    return rows


class AgentRewardBenchAdapter:
    """Adapter for McGill-NLP/agent-reward-bench expert-annotated web agent trajectories."""

    def __init__(self, split: str = "test", min_steps: int = 1, limit: int | None = None) -> None:
        self.split = split
        self.min_steps = min_steps
        self.limit = limit
        self._atif = ATIFAdapter()

    def load(
        self, source: str | Path | list[dict[str, Any]] | dict[str, Any] = "McGill-NLP/agent-reward-bench"
    ) -> list[Trajectory]:
        records = _load_records(source, split=self.split, limit=self.limit)
        trajectories: list[Trajectory] = []
        for idx, rec in enumerate(records):
            traj = self._parse_record(rec, idx)
            if len(traj.steps) >= self.min_steps:
                trajectories.append(traj)
        return trajectories

    def _parse_record(self, rec: dict[str, Any], idx: int) -> Trajectory:
        if "steps" in rec and isinstance(rec["steps"], list) and rec["steps"] and isinstance(rec["steps"][0], dict):
            return self._atif.parse_record(rec, idx)

        trace_id = str(rec.get("id") or rec.get("task_id") or f"arb-{idx}")
        goal = str(rec.get("goal") or rec.get("task") or rec.get("instruction") or "")
        raw_actions = rec.get("actions") or rec.get("trajectory") or []
        raw_obs = rec.get("observations") or rec.get("axtrees") or []
        raw_thoughts = rec.get("thoughts") or []

        steps: list[TraceStep] = []
        total = max(len(raw_actions), len(raw_obs))
        for i in range(total):
            act = str(raw_actions[i]) if i < len(raw_actions) else ""
            obs = str(raw_obs[i]) if i < len(raw_obs) else ""
            thought = str(raw_thoughts[i]) if i < len(raw_thoughts) else ""
            steps.append(
                TraceStep(
                    step_id=i + 1,
                    role="assistant",
                    agent_id="web_agent",
                    thought=thought,
                    action=act,
                    tool_name="browser_action",
                    observation=obs,
                )
            )

        gt = None
        for key in ("human_label", "expert_reward", "reward", "success", "ground_truth"):
            if key in rec and rec[key] is not None:
                gt = {"label": rec[key]}
                break

        return Trajectory(
            trace_id=trace_id,
            task_instruction=goal,
            final_output=str(rec.get("final_answer") or rec.get("response") or (steps[-1].action if steps else "")),
            steps=steps,
            ground_truth=gt,
            metadata={"benchmark": rec.get("benchmark", "agent-reward-bench")},
        )


class DeepResearchTrajAdapter:
    """Adapter for IPF/DeepResearch-traj and OpenResearcher long-horizon web-search traces."""

    def __init__(self, split: str = "train", min_steps: int = 1, limit: int | None = None) -> None:
        self.split = split
        self.min_steps = min_steps
        self.limit = limit
        self._atif = ATIFAdapter()

    def load(
        self, source: str | Path | list[dict[str, Any]] | dict[str, Any] = "IPF/DeepResearch-traj"
    ) -> list[Trajectory]:
        records = _load_records(source, split=self.split, limit=self.limit)
        trajectories: list[Trajectory] = []
        for idx, rec in enumerate(records):
            traj = self._parse_record(rec, idx)
            if len(traj.steps) >= self.min_steps:
                trajectories.append(traj)
        return trajectories

    def _parse_record(self, rec: dict[str, Any], idx: int) -> Trajectory:
        if "steps" in rec or "messages" in rec:
            traj = self._atif.parse_record(rec, idx)
            if "pass_rate" in rec and traj.ground_truth is None:
                traj.ground_truth = {"pass_rate": rec["pass_rate"]}
            return traj

        trace_id = str(rec.get("id") or rec.get("question_id") or f"dr-{idx}")
        question = str(rec.get("question") or rec.get("query") or rec.get("prompt") or "")
        raw_traj = rec.get("trajectory") or rec.get("rollout") or []
        if isinstance(raw_traj, str):
            try:
                raw_traj = json.loads(raw_traj)
            except json.JSONDecodeError:
                raw_traj = [{"content": raw_traj}]

        steps: list[TraceStep] = []
        for i, item in enumerate(raw_traj):
            if isinstance(item, dict):
                role = str(item.get("role", "assistant"))
                thought = str(item.get("thought", item.get("reasoning", "")))
                action = str(item.get("action", item.get("content", "")))
                obs = str(item.get("observation", item.get("search_result", "")))
                tool_name = item.get("tool_name") or ("web_search" if "search" in action.lower() else None)
            else:
                role, thought, action, obs, tool_name = "assistant", "", str(item), "", None

            steps.append(
                TraceStep(
                    step_id=i + 1,
                    role=role,
                    agent_id="researcher",
                    thought=thought,
                    action=action,
                    tool_name=tool_name,
                    observation=obs,
                )
            )

        gt = None
        if "pass_rate" in rec or "correct" in rec or "answer" in rec:
            gt = {
                k: rec[k]
                for k in ("pass_rate", "correct", "answer", "ground_truth")
                if k in rec
            }

        return Trajectory(
            trace_id=trace_id,
            task_instruction=question,
            final_output=str(rec.get("final_answer") or rec.get("prediction") or (steps[-1].action if steps else "")),
            steps=steps,
            ground_truth=gt,
            metadata={"dataset": "deep_research_traj"},
        )
