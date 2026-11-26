"""ATIF / Harbor / Hugging Face Hub Trace JSON & JSONL adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trace_tree_eval.schema import TraceStep, Trajectory


class ATIFAdapter:
    """Load traces in ATIF, Harbor, or generic message/step JSON/JSONL formats."""

    def load(self, source: str | Path | list[dict[str, Any]] | dict[str, Any]) -> list[Trajectory]:
        records: list[dict[str, Any]]
        if isinstance(source, (str, Path)):
            path = Path(source)
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                return []
            if path.suffix.lower() == ".jsonl" or ("\n" in text and not text.startswith("[") and not text.startswith("{")):
                records = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                parsed = json.loads(text)
                records = parsed if isinstance(parsed, list) else [parsed]
        elif isinstance(source, dict):
            records = [source]
        else:
            records = list(source)

        return [self.parse_record(rec, idx) for idx, rec in enumerate(records)]

    def parse_record(self, rec: dict[str, Any], idx: int = 0) -> Trajectory:
        trace_id = str(
            rec.get("trace_id")
            or rec.get("session_id")
            or rec.get("trajectory_id")
            or rec.get("id")
            or f"trace-{idx}"
        )
        task_instruction = str(
            rec.get("task_instruction")
            or rec.get("instruction")
            or rec.get("prompt")
            or rec.get("query")
            or ""
        )
        deliverables = [str(d) for d in rec.get("deliverables", [])]
        raw_steps = rec.get("steps") or rec.get("trajectory") or rec.get("messages") or []

        steps: list[TraceStep] = []
        for s_idx, item in enumerate(raw_steps):
            if not isinstance(item, dict):
                steps.append(TraceStep(step_id=s_idx + 1, role="assistant", observation=str(item)))
                continue

            step_id = int(item.get("step_id", item.get("step", item.get("turn", s_idx + 1))))
            role = str(item.get("role") or item.get("source") or "assistant")
            agent_id = str(item.get("agent_id") or item.get("agent_name") or "main")
            thought = str(item.get("thought") or item.get("reasoning") or "")
            action = str(item.get("action") or item.get("content") or item.get("message") or "")
            tool_name = item.get("tool_name") or item.get("tool")
            tool_args = item.get("tool_args") or item.get("arguments")
            if isinstance(item.get("tool_calls"), list) and item["tool_calls"]:
                tc0 = item["tool_calls"][0]
                if isinstance(tc0, dict):
                    fn = tc0.get("function", tc0)
                    tool_name = tool_name or fn.get("name")
                    raw_args = fn.get("arguments")
                    if isinstance(raw_args, str):
                        try:
                            tool_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            tool_args = {"raw": raw_args}
                    elif isinstance(raw_args, dict):
                        tool_args = raw_args

            observation = str(
                item.get("observation")
                or item.get("observation_text")
                or item.get("tool_output")
                or item.get("output")
                or ""
            )
            status = str(item.get("status") or ("error" if item.get("error") else "ok"))
            if not task_instruction and role == "user" and s_idx == 0:
                task_instruction = action or observation

            steps.append(
                TraceStep(
                    step_id=step_id,
                    role=role,
                    agent_id=agent_id,
                    thought=thought,
                    action=action,
                    tool_name=str(tool_name) if tool_name else None,
                    tool_args=tool_args if isinstance(tool_args, dict) else None,
                    observation=observation,
                    status=status,
                    metadata={k: v for k, v in item.items() if k not in {
                        "step_id", "step", "turn", "role", "source", "agent_id", "agent_name",
                        "thought", "reasoning", "action", "content", "message", "tool_name",
                        "tool", "tool_args", "arguments", "tool_calls", "observation",
                        "observation_text", "tool_output", "output", "status",
                    }},
                )
            )

        final_output = str(
            rec.get("final_output")
            or rec.get("final_answer")
            or rec.get("answer")
            or (steps[-1].action or steps[-1].observation if steps else "")
        )
        ground_truth = rec.get("ground_truth")
        if ground_truth is None and "reward" in rec:
            ground_truth = {"reward": rec["reward"]}

        return Trajectory(
            trace_id=trace_id,
            task_instruction=task_instruction,
            deliverables=deliverables,
            final_output=final_output,
            steps=steps,
            ground_truth=ground_truth,
            metadata=dict(rec.get("metadata", {})),
        )
