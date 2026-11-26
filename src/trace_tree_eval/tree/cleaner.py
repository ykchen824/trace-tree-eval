"""Stage 0: Deterministic trajectory cleaning, loop RLE, and Global Manifest extraction."""

from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy
from typing import Any

from trace_tree_eval.schema import GlobalManifest, TraceStep, Trajectory
from trace_tree_eval.tokens import estimate_tokens

_SHINGLE_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]")

_VERIFICATION_PATTERNS = re.compile(
    r"(self[-_ ]?check|verify_output|cross[-_ ]?check|run_tests|pytest|unittest|核对需求|自我检查|验证结果|复核)",
    re.IGNORECASE,
)
_SENSITIVE_PATTERNS = re.compile(
    r"(api[_-]?key|secret|password|token|\.env|private[_-]?key|信用卡|credit card|closed[-_ ]?source|闭源)",
    re.IGNORECASE,
)
_CONSTRAINT_PATTERNS = re.compile(
    r"(补充要求|修改需求|注意[:：]|必须|不得|不要|禁止|must not|do not|requirement update|constraint)",
    re.IGNORECASE,
)
_CONTRADICTION_PATTERNS = re.compile(
    r"(recommend|建议选用|最终推荐|choose|selected)",
    re.IGNORECASE,
)


def _shingle_set(text: str) -> set[str]:
    tokens = _SHINGLE_RE.findall(text.lower())
    if len(tokens) < 3:
        return set(tokens)
    return {f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _strip_repeated_boilerplate(steps: list[TraceStep]) -> list[TraceStep]:
    """Remove boilerplate lines that appear verbatim across >= 4 distinct steps."""
    line_freq: Counter[str] = Counter()
    for step in steps:
        if not step.observation:
            continue
        unique_lines = {ln.strip() for ln in step.observation.splitlines() if len(ln.strip()) >= 25}
        for ln in unique_lines:
            line_freq[ln] += 1

    boilerplate = {ln for ln, cnt in line_freq.items() if cnt >= 4}
    cleaned: list[TraceStep] = []
    for step in steps:
        s = deepcopy(step)
        if boilerplate and s.observation:
            kept_lines = [
                ln for ln in s.observation.splitlines() if ln.strip() not in boilerplate
            ]
            if len(kept_lines) < len(s.observation.splitlines()):
                s.observation = "\n".join(kept_lines).strip()
        cleaned.append(s)
    return cleaned


def _detect_step_risk_tags(step: TraceStep, is_first_step: bool) -> list[str]:
    tags: list[str] = []
    text = step.render_text(include_header=False)
    if step.status == "error" or "error" in text.lower() or "timeout" in text.lower():
        tags.append("error")
    if step.role == "user" and not is_first_step:
        tags.append("mid_task_constraint")
    elif _CONSTRAINT_PATTERNS.search(text) and step.role in ("user", "system"):
        tags.append("mid_task_constraint")
    if _VERIFICATION_PATTERNS.search(text) or (
        step.tool_name and _VERIFICATION_PATTERNS.search(step.tool_name)
    ):
        tags.append("self_verification")
    if _SENSITIVE_PATTERNS.search(text):
        tags.append("sensitive_data")
    if _CONTRADICTION_PATTERNS.search(text) and _SENSITIVE_PATTERNS.search(text):
        tags.append("contradiction_cue")
    return tags


def clean_trajectory(
    trajectory: Trajectory,
    dedup_jaccard_threshold: float = 0.85,
    min_loop_len: int = 3,
) -> tuple[list[TraceStep], GlobalManifest, dict[int, list[str]]]:
    """Run Stage 0 deterministic reduction, loop RLE folding, and Global Manifest extraction."""
    raw_steps = trajectory.steps
    raw_tokens = trajectory.total_tokens()

    # 1. Extract exact tool statistics, verification steps, and mid-task constraints before folding
    tool_stats: dict[str, dict[str, int]] = {}
    verification_steps: list[int] = []
    mid_task_constraint_steps: list[int] = []
    step_risk_tags: dict[int, list[str]] = {}

    for idx, step in enumerate(raw_steps):
        tags = _detect_step_risk_tags(step, is_first_step=(idx == 0))
        if tags:
            step_risk_tags[step.step_id] = tags
        if "self_verification" in tags:
            verification_steps.append(step.step_id)
        if "mid_task_constraint" in tags:
            mid_task_constraint_steps.append(step.step_id)

        tname = step.tool_name or (f"role:{step.role}" if step.role != "assistant" else "reasoning")
        if tname not in tool_stats:
            tool_stats[tname] = {
                "calls": 0,
                "errors": 0,
                "first_step": step.step_id,
                "last_step": step.step_id,
            }
        tool_stats[tname]["calls"] += 1
        if step.status == "error":
            tool_stats[tname]["errors"] += 1
        tool_stats[tname]["last_step"] = step.step_id

    # 2. Strip repeated boilerplate across observations
    deboilerplated = _strip_repeated_boilerplate(raw_steps)

    # 3. Fold consecutive retry loops (RLE)
    rle_steps: list[TraceStep] = []
    loop_folds: list[dict[str, Any]] = []
    i = 0
    while i < len(deboilerplated):
        cur = deboilerplated[i]
        cur_sig = (cur.agent_id, cur.tool_name, cur.status, cur.action[:80].strip())
        # Never fold user constraint turns or self-verification steps
        cur_tags = step_risk_tags.get(cur.step_id, [])
        if "mid_task_constraint" in cur_tags or "self_verification" in cur_tags:
            rle_steps.append(cur)
            i += 1
            continue

        j = i + 1
        while j < len(deboilerplated):
            nxt = deboilerplated[j]
            nxt_tags = step_risk_tags.get(nxt.step_id, [])
            if "mid_task_constraint" in nxt_tags or "self_verification" in nxt_tags:
                break
            nxt_sig = (nxt.agent_id, nxt.tool_name, nxt.status, nxt.action[:80].strip())
            if nxt_sig != cur_sig:
                break
            j += 1

        run_len = j - i
        if run_len >= min_loop_len:
            first_s = deboilerplated[i]
            last_s = deboilerplated[j - 1]
            folded = deepcopy(first_s)
            folded.observation = (
                f"[LOOP_FOLD: Folded {run_len} repeated steps "
                f"(steps {first_s.step_id}-{last_s.step_id}, tool={first_s.tool_name}, status={first_s.status}). "
                f"Sample output: {first_s.observation[:140]}]"
            )
            folded.metadata["folded_step_ids"] = [s.step_id for s in deboilerplated[i:j]]
            rle_steps.append(folded)
            tags = step_risk_tags.setdefault(folded.step_id, [])
            if "loop_fold" not in tags:
                tags.append("loop_fold")
            loop_folds.append(
                {
                    "anchor_step_id": first_s.step_id,
                    "start_step": first_s.step_id,
                    "end_step": last_s.step_id,
                    "count": run_len,
                    "tool": first_s.tool_name,
                    "status": first_s.status,
                }
            )
            i = j
        else:
            rle_steps.append(cur)
            i += 1

    # 4. Near-duplicate observation folding against earlier observations
    seen_obs: list[tuple[int, set[str]]] = []
    final_steps: list[TraceStep] = []
    for step in rle_steps:
        s = deepcopy(step)
        tags = step_risk_tags.get(s.step_id, [])
        if (
            s.observation
            and len(s.observation) >= 120
            and "mid_task_constraint" not in tags
            and "loop_fold" not in tags
        ):
            shingles = _shingle_set(s.observation)
            matched_ref: int | None = None
            matched_sim = 0.0
            for prev_id, prev_shingles in seen_obs[-20:]:
                sim = _jaccard(shingles, prev_shingles)
                if sim >= dedup_jaccard_threshold:
                    matched_ref = prev_id
                    matched_sim = sim
                    break
            if matched_ref is not None:
                s.observation = (
                    f"[DEDUP: {int(matched_sim * 100)}% overlap with step {matched_ref}; "
                    f"prefix={s.observation[:90]}...]"
                )
            else:
                seen_obs.append((s.step_id, shingles))
        final_steps.append(s)

    cleaned_tokens = (
        estimate_tokens(trajectory.task_instruction)
        + estimate_tokens(trajectory.final_output)
        + sum(s.token_count() for s in final_steps)
    )
    manifest = GlobalManifest(
        total_steps=len(raw_steps),
        raw_tokens=raw_tokens,
        cleaned_tokens=cleaned_tokens,
        tool_stats=tool_stats,
        verification_detected=len(verification_steps) > 0,
        verification_steps=verification_steps,
        mid_task_constraint_steps=mid_task_constraint_steps,
        loop_folds=loop_folds,
    )
    return final_steps, manifest, step_risk_tags
