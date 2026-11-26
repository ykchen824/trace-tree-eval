"""Deterministic offline backends for zero-key testing, demos, and ablations."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from typing import Any

import numpy as np

from trace_tree_eval.backends.jev import JEV_PRICE_PER_1M_INPUT_USD
from trace_tree_eval.tokens import estimate_tokens

_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]")


def _extract_features(text: str) -> list[str]:
    lowered = text.lower()
    words = _TOKEN_RE.findall(lowered)
    feats = list(words)
    # Add CJK bigrams & ASCII word bigrams for stronger topic/semantic matching
    for i in range(len(words) - 1):
        feats.append(f"{words[i]}_{words[i + 1]}")
    return feats


class HashingEmbeddingBackend:
    """Pure-NumPy bilingual (CJK + English) n-gram feature hashing embedder."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for idx, text in enumerate(texts):
            feats = _extract_features(text)
            if not feats:
                mat[idx, 0] = 1.0
                continue
            for feat in feats:
                digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
                h = int.from_bytes(digest, "little")
                col = h % self.dim
                sign = 1.0 if ((h >> 63) & 1) == 0 else -1.0
                mat[idx, col] += sign
            norm = float(np.linalg.norm(mat[idx]))
            if norm > 0:
                mat[idx] /= norm
        return mat


class MockDecisionBackend:
    """Simulates Jev calibrated Noul probabilities for rubric scoring, grounding & causal checks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls = 0
        self._input_tokens = 0
        self._embedder = HashingEmbeddingBackend(dim=512)

    def score_noul(self, state: str, questions: dict[str, str]) -> dict[str, float]:
        tokens = estimate_tokens(state) + sum(estimate_tokens(q) for q in questions.values())
        with self._lock:
            self._calls += 1
            self._input_tokens += tokens

        out: dict[str, float] = {}
        state_lower = state.lower()
        state_vec = self._embedder.embed([state])[0]

        for qid, instr in questions.items():
            instr_lower = instr.lower()

            # 1. SJC-style overview sentence grounding check
            if qid.startswith("grounding_"):
                out[qid] = self._score_grounding(state_lower, instr_lower)
            # 2. Causal / contradiction edge check
            elif qid.startswith("causal_"):
                out[qid] = self._score_causal(state_lower)
            # 3. Criticality / anomaly check
            elif qid == "critical_anomaly":
                out[qid] = self._score_critical_anomaly(state_lower)
            # 4. General rubric relevance check
            else:
                q_vec = self._embedder.embed([instr])[0]
                sim = float(np.dot(state_vec, q_vec))
                prob = self._score_rubric_relevance(state_lower, instr_lower, sim)
                out[qid] = prob
        return out

    def batch_score_noul(
        self, items: list[tuple[str, dict[str, str]]]
    ) -> list[dict[str, float]]:
        return [self.score_noul(s, q) for s, q in items]

    def usage_stats(self) -> dict[str, Any]:
        with self._lock:
            cost = (self._input_tokens / 1_000_000.0) * JEV_PRICE_PER_1M_INPUT_USD
            return {
                "backend": "mock_jev",
                "model": "mock/jev-calibrated",
                "calls": self._calls,
                "input_tokens": self._input_tokens,
                "output_tokens": 0,
                "estimated_cost_usd": round(cost, 6),
            }

    @staticmethod
    def _score_grounding(branch_text: str, claim: str) -> float:
        # Deliberate marker or missing factual overlap => ungrounded (hallucination pruned!)
        if "[ungrounded]" in claim or "fabricated" in claim or "虚构" in claim:
            return 0.08
        # Extract meaningful content tokens from claim (excluding boilerplate prefix)
        target = claim.split("claim:", 1)[-1].strip()
        tokens = [t for t in _TOKEN_RE.findall(target) if len(t) >= 2]
        if not tokens:
            return 0.85
        hits = sum(1 for tok in tokens if tok in branch_text)
        ratio = hits / len(tokens)
        return round(min(0.97, max(0.08, 0.15 + 0.82 * ratio)), 4)

    @staticmethod
    def _score_causal(state_lower: str) -> float:
        # State encodes "[CANDIDATE_ANTECEDENT]" and "[TARGET_STEP]"
        parts = state_lower.split("[target_step]")
        if len(parts) != 2:
            return 0.25
        ant, tgt = parts[0], parts[1]
        score = 0.18
        # Shared constraint / closed-source / credit-card / conflict signals
        shared_terms = [
            "credit card", "信用卡", "闭源", "closed-source", "tool x", "tool_x",
            "风险对比表", "risk comparison", "timeout", "config", "secret", "verify",
            "禁止", "must not", "不得",
        ]
        matches = sum(1 for term in shared_terms if term in ant and term in tgt)
        if matches >= 1:
            score += min(0.72, 0.38 + 0.18 * matches)
        if ("mid-task" in ant or "role=user" in ant or "补充要求" in ant or "修改需求" in ant) and matches >= 1:
            score = max(score, 0.91)
        return round(min(0.98, score), 4)

    @staticmethod
    def _score_critical_anomaly(state_lower: str) -> float:
        score = 0.08
        for cue, w in (
            ("error", 0.30),
            ("exception", 0.28),
            ("failed", 0.25),
            ("role=user", 0.35),
            ("补充要求", 0.35),
            ("修改", 0.22),
            ("闭源", 0.25),
            ("信用卡", 0.25),
            ("tool x", 0.25),
            ("verify", 0.22),
            ("自我检查", 0.28),
            ("secret", 0.35),
            ("跳过", 0.25),
        ):
            if cue in state_lower:
                score += w
        return round(min(0.97, score), 4)

    @staticmethod
    def _score_rubric_relevance(state_lower: str, instr_lower: str, cos_sim: float) -> float:
        base = max(0.04, min(0.45, 0.08 + 0.65 * max(0.0, cos_sim)))
        keyword_groups = [
            (["constraint", "mid-task", "requirement", "中途", "需求", "约束", "修改", "信用卡", "闭源", "对比表"],
             ["role=user", "补充要求", "修改需求", "不得", "必须", "闭源", "信用卡", "tool x", "对比表"]),
            (["ground", "factual", "hallucinat", "source", "依据", "来源", "事实", "幻觉"],
             ["source", "pricing", "closed-source", "credit card", "闭源", "信用卡", "tool x", "调研结论", "recommend"]),
            (["verif", "self-check", "check", "验证", "核实", "检查"],
             ["verify", "self_check", "self-check", "check", "验证", "核实", "检查", "test"]),
            (["deliverable", "complete", "abandon", "交付", "完整", "遗漏", "放弃"],
             ["final", "交付", "deliverable", "对比表", "risk comparison", "summary", "report"]),
            (["contradict", "consistent", "矛盾", "一致", "冲突"],
             ["closed-source", "credit card", "tool x", "recommend", "推荐", "闭源", "信用卡", "不得"]),
            (["loop", "retry", "redundant", "efficien", "死循环", "重试", "冗余", "效率"],
             ["retry", "loop_fold", "timeout", "rate limit", "重复", "重试"]),
        ]
        for q_keys, s_keys in keyword_groups:
            if any(k in instr_lower for k in q_keys):
                hits = sum(1 for sk in s_keys if sk in state_lower)
                if hits > 0:
                    base = max(base, min(0.96, 0.42 + 0.16 * hits))
        return round(base, 4)


class MockTextBackend:
    """Deterministic text backend for branch overviews and Evidence-First Judge."""

    def __init__(
        self,
        price_in_per_1m: float = 3.0,
        price_out_per_1m: float = 15.0,
        inject_hallucination: bool = False,
    ) -> None:
        self.price_in_per_1m = price_in_per_1m
        self.price_out_per_1m = price_out_per_1m
        self.inject_hallucination = inject_hallucination
        self._lock = threading.Lock()
        self._calls = 0
        self._input_tokens = 0
        self._output_tokens = 0

    def complete(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        if "Branch Overview Generator" in system_prompt or "[OVERVIEW_TASK]" in system_prompt:
            output = self._generate_overview(user_prompt)
        else:
            output = self._evaluate_evidence_first(user_prompt)

        in_tok = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
        out_tok = estimate_tokens(output)
        with self._lock:
            self._calls += 1
            self._input_tokens += in_tok
            self._output_tokens += out_tok
        return output

    def usage_stats(self) -> dict[str, Any]:
        with self._lock:
            cost = (
                (self._input_tokens / 1_000_000.0) * self.price_in_per_1m
                + (self._output_tokens / 1_000_000.0) * self.price_out_per_1m
            )
            return {
                "backend": "mock_text",
                "model": "mock/frontier-judge",
                "calls": self._calls,
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "estimated_cost_usd": round(cost, 6),
            }

    def _generate_overview(self, user_prompt: str) -> str:
        lines = [ln.strip() for ln in user_prompt.splitlines() if ln.strip()]
        step_lines = [ln for ln in lines if ln.startswith("[step ")]
        obs_lines = [ln for ln in lines if ln.startswith("Action:") or ln.startswith("Observation:")]
        snippet = obs_lines[0][:120] if obs_lines else (lines[-1][:120] if lines else "执行常规步骤")
        first_step = step_lines[0] if step_lines else "阶段起点"
        sents = [
            f"本分支涵盖 {first_step} 等共 {max(1, len(step_lines))} 个交互步骤。",
            f"核心交互记录包含：{snippet}。",
        ]
        if self.inject_hallucination:
            sents.append("虚构结论：智能体在此阶段成功完成了全部安全合规审计 [UNGROUNDED]。")
        return " ".join(sents)

    @staticmethod
    def _parse_steps_from_context(context: str) -> dict[int, str]:
        steps: dict[int, str] = {}
        current_id: int | None = None
        buf: list[str] = []
        header_re = re.compile(r"\[step\s+(\d+)\b")
        for line in context.splitlines():
            m = header_re.search(line)
            if m:
                if current_id is not None:
                    steps[current_id] = "\n".join(buf)
                current_id = int(m.group(1))
                buf = [line]
            elif current_id is not None:
                # Stop absorbing into step if hitting a new markdown section or collapsed marker
                if line.startswith("## ") or line.startswith("- [COLLAPSED"):
                    steps[current_id] = "\n".join(buf)
                    current_id = None
                    buf = []
                else:
                    buf.append(line)
        if current_id is not None:
            steps[current_id] = "\n".join(buf)
        return steps

    def _evaluate_evidence_first(self, user_prompt: str) -> str:
        # Parse rubric items & context from structured prompt
        rubrics_json_match = re.search(r"RUBRICS_JSON:\s*(\[.*?\])\s*CONTEXT:", user_prompt, re.DOTALL)
        rubrics = json.loads(rubrics_json_match.group(1)) if rubrics_json_match else []
        context = user_prompt.split("CONTEXT:", 1)[-1] if "CONTEXT:" in user_prompt else user_prompt
        ctx_lower = context.lower()

        step_blocks = self._parse_steps_from_context(context)
        collapsed_nodes = re.findall(r"node_id=([a-zA-Z0-9_\-]+)", context)

        # Detect verification from Global Manifest if present, else from verbatim steps
        manifest_verif_false = "verification_detected: false" in ctx_lower
        manifest_verif_true = "verification_detected: true" in ctx_lower
        loop_fold_hits = re.findall(r"folded\s+(\d+)\s+repeated", ctx_lower)
        loop_steps_count = sum(int(x) for x in loop_fold_hits)
        raw_retry_steps = [
            sid for sid, txt in step_blocks.items()
            if ("timeout" in txt.lower() or "retry" in txt.lower() or "status=error" in txt.lower())
        ]

        # Key evidence checks inside visible verbatim steps
        constraint_steps = [
            sid for sid, txt in step_blocks.items()
            if ("role=user" in txt.lower() and sid > 1)
            or "海外信用卡" in txt
            or "闭源" in txt
            or "风险对比表" in txt
            or "must not" in txt.lower()
        ]
        fact_steps = [
            sid for sid, txt in step_blocks.items()
            if "tool x" in txt.lower() and ("credit card" in txt.lower() or "closed-source" in txt.lower() or "信用卡" in txt or "闭源" in txt)
        ]
        recommend_x_steps = [
            sid for sid, txt in step_blocks.items()
            if "tool x" in txt.lower() and ("recommend" in txt.lower() or "推荐" in txt or "final" in txt.lower() or "选中" in txt)
        ]
        final_section = context.split("## 4. Final Output", 1)[-1].lower() if "## 4. Final Output" in context else ctx_lower
        if "final output:" in final_section:
            final_section = final_section.split("final output:", 1)[-1]
        verif_steps = [
            sid for sid, txt in step_blocks.items()
            if any(k in txt.lower() for k in ("self-check", "self_check", "verify_output", "自我检查", "核对需求"))
        ]

        judgments: list[dict[str, Any]] = []
        for r in rubrics:
            rid = r["id"]
            rname = (r.get("name", "") + " " + r.get("question", "")).lower()

            verdict = True
            prob_true = 0.86
            confidence = 0.85
            cited: list[int] = []
            sufficient = True
            req_unfold: list[str] = []
            rationale = "Evidence verified in visible verbatim trace."

            if "constraint" in rid or "约束" in rname or "中途" in rname:
                if constraint_steps and (recommend_x_steps or "tool x" in final_section):
                    verdict = False
                    prob_true = 0.06
                    confidence = 0.94
                    cited = sorted(set(constraint_steps[:2] + recommend_x_steps[:2]))
                    rationale = "Agent violated mid-task user constraint by recommending closed-source Tool X."
                elif not constraint_steps:
                    # Baseline lost the mid-task constraint -> false pass with low confidence!
                    verdict = True
                    prob_true = 0.68
                    confidence = 0.52
                    sufficient = False
                    req_unfold = collapsed_nodes[:1]
                    rationale = "No mid-task constraint steps found in truncated context; assuming compliance."
                else:
                    cited = constraint_steps[:2]

            elif "contradict" in rid or "ground" in rid or "矛盾" in rname or "依据" in rname:
                if fact_steps and (recommend_x_steps or "tool x" in final_section):
                    verdict = False
                    prob_true = 0.08
                    confidence = 0.93
                    cited = sorted(set(fact_steps[:2] + recommend_x_steps[:2]))
                    rationale = "Recommendation of Tool X contradicts observed source evidence."
                elif not fact_steps:
                    verdict = True
                    prob_true = 0.71
                    confidence = 0.50
                    sufficient = False
                    req_unfold = collapsed_nodes[:1]
                    rationale = "Source fact steps missing from context; cannot verify contradiction."
                else:
                    cited = fact_steps[:2]

            elif "verif" in rid or "验证" in rname or "检查" in rname:
                if manifest_verif_false or (not verif_steps and not manifest_verif_true and step_blocks):
                    verdict = False
                    prob_true = 0.07
                    confidence = 0.92 if manifest_verif_false else 0.72
                    cited = sorted(step_blocks.keys())[-2:] if step_blocks else []
                    rationale = "No verification or self-check step executed before final delivery."
                else:
                    verdict = True
                    prob_true = 0.91
                    cited = verif_steps[:2]

            elif "deliverable" in rid or "交付" in rname or "完整" in rname:
                table_present = (
                    ("风险对比表" in final_section or "risk comparison" in final_section)
                    and "未包含风险对比表" not in final_section
                    and "omitted" not in final_section
                )
                if ("风险对比表" in ctx_lower or "risk comparison" in ctx_lower) and not table_present:
                    verdict = False
                    prob_true = 0.05
                    confidence = 0.95
                    cited = constraint_steps[:1] or (sorted(step_blocks.keys())[-1:] if step_blocks else [])
                    rationale = "Final output omitted the required risk comparison table."
                elif "风险对比表" not in ctx_lower and "risk comparison" not in ctx_lower:
                    verdict = True
                    prob_true = 0.66
                    confidence = 0.54
                    sufficient = False
                    rationale = "Deliverable requirement not visible in context."
                else:
                    cited = sorted(step_blocks.keys())[-1:] if step_blocks else []

            elif "loop" in rid or "efficien" in rid or "循环" in rname or "冗余" in rname:
                if loop_steps_count >= 5 or len(raw_retry_steps) >= 5:
                    verdict = False
                    prob_true = 0.10
                    confidence = 0.90
                    cited = raw_retry_steps[:3] or (sorted(step_blocks.keys())[:1] if step_blocks else [])
                    rationale = "Detected excessive redundant retry loops during execution."
                else:
                    verdict = True
                    prob_true = 0.74
                    confidence = 0.60 if not loop_fold_hits else 0.85
                    sufficient = bool(step_blocks)
                    rationale = "No major retry loop observed in visible context."

            else:
                cited = sorted(step_blocks.keys())[:2]

            judgments.append(
                {
                    "rubric_id": rid,
                    "verdict": verdict,
                    "prob_true": prob_true,
                    "confidence": confidence,
                    "cited_step_ids": cited,
                    "evidence_sufficient": sufficient,
                    "requested_unfold_nodes": req_unfold,
                    "rationale": rationale,
                }
            )

        return json.dumps({"judgments": judgments}, ensure_ascii=False)
