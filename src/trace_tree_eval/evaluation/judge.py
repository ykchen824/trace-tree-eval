"""Evidence-First Judge with Tier-2 node unfolding escalation."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from trace_tree_eval.backends.base import TextModelBackend
from trace_tree_eval.backends.mock import MockTextBackend
from trace_tree_eval.compression.compressor import HybridTreeCompressor
from trace_tree_eval.schema import (
    EvalDossier,
    JudgeEvaluation,
    RubricItem,
    RubricJudgment,
    Trajectory,
)
from trace_tree_eval.tokens import estimate_tokens

_JUDGE_SYSTEM_PROMPT = """You are an Evidence-First AI Agent Trajectory Judge.
Evaluate the trajectory context strictly against each boolean rubric item.
Rules:
1. Base verdicts ONLY on verbatim steps or deterministic Global Execution Manifest facts.
2. NEVER treat '[COLLAPSED | Verified Overview - Not Ground-Truth Evidence]' as decisive proof for passing or failing; if decisive evidence is missing or hidden in a collapsed node, set evidence_sufficient=false, lower confidence (<0.65), and list node_id(s) in requested_unfold_nodes.
3. Return valid JSON matching:
{
  "judgments": [
    {
      "rubric_id": "...",
      "verdict": true/false,
      "prob_true": 0.0-1.0,
      "confidence": 0.0-1.0,
      "cited_step_ids": [1, 2],
      "evidence_sufficient": true/false,
      "requested_unfold_nodes": ["b2_e1"],
      "rationale": "..."
    }
  ]
}"""


@dataclass
class JudgeConfig:
    """Configuration for Evidence-First evaluation and Tier-2 escalation."""

    escalation_confidence_threshold: float = 0.65
    min_coverage_threshold: float = 0.45
    max_escalations: int = 1
    extra_unfold_budget_tokens: int = 4000


class EvidenceFirstJudge:
    """Controlled-harness Evidence-First Judge supporting Tier-2 tree unfolding."""

    def __init__(
        self,
        text_backend: TextModelBackend | None = None,
        config: JudgeConfig | None = None,
    ) -> None:
        self.text_backend = text_backend or MockTextBackend()
        self.config = config or JudgeConfig()

    def evaluate_context(
        self,
        trace_id: str,
        context_text: str,
        rubrics: list[RubricItem],
        method_name: str = "baseline",
    ) -> JudgeEvaluation:
        """Evaluate an arbitrary rendered trace/baseline string with the fixed judge prompt."""
        t0 = time.perf_counter()
        before_stats = self.text_backend.usage_stats()
        judgments = self._run_single_pass(context_text, rubrics)
        after_stats = self.text_backend.usage_stats()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        delta_tokens = max(
            estimate_tokens(context_text),
            int(after_stats.get("input_tokens", 0)) - int(before_stats.get("input_tokens", 0)),
        )
        delta_cost = max(
            0.0,
            float(after_stats.get("estimated_cost_usd", 0.0))
            - float(before_stats.get("estimated_cost_usd", 0.0)),
        )
        return JudgeEvaluation(
            trace_id=trace_id,
            method_name=method_name,
            judgments=judgments,
            escalated=False,
            unfolded_nodes=[],
            total_input_tokens=delta_tokens,
            estimated_cost_usd=round(delta_cost, 6),
            latency_ms=round(latency_ms, 2),
        )

    def evaluate_dossier(
        self,
        trajectory: Trajectory,
        dossier: EvalDossier,
        rubrics: list[RubricItem],
        compressor: HybridTreeCompressor | None = None,
        method_name: str = "HybridTree",
    ) -> JudgeEvaluation:
        """Evaluate an EvalDossier and trigger Tier-2 unfolding if evidence is insufficient."""
        t0 = time.perf_counter()
        before_stats = self.text_backend.usage_stats()

        current_dossier = dossier
        judgments = self._run_single_pass(current_dossier.rendered_markdown, rubrics)
        escalated = False
        unfolded_nodes: list[str] = []

        for _ in range(self.config.max_escalations):
            need_unfold, targets = self._needs_escalation(judgments, current_dossier, rubrics)
            if not need_unfold or compressor is None:
                break
            prev_expanded = set(current_dossier.expanded_node_ids)
            current_dossier = compressor.unfold_nodes(
                trajectory=trajectory,
                dossier=current_dossier,
                node_ids=targets,
                rubrics=rubrics,
                extra_budget_tokens=self.config.extra_unfold_budget_tokens,
            )
            newly_added = [
                nid for nid in current_dossier.expanded_node_ids if nid not in prev_expanded
            ]
            if not newly_added:
                break
            escalated = True
            unfolded_nodes.extend(newly_added)
            judgments = self._run_single_pass(current_dossier.rendered_markdown, rubrics)

        after_stats = self.text_backend.usage_stats()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        delta_tokens = max(
            current_dossier.dossier_tokens,
            int(after_stats.get("input_tokens", 0)) - int(before_stats.get("input_tokens", 0)),
        )
        delta_cost = max(
            0.0,
            float(after_stats.get("estimated_cost_usd", 0.0))
            - float(before_stats.get("estimated_cost_usd", 0.0)),
        )
        return JudgeEvaluation(
            trace_id=trajectory.trace_id,
            method_name=method_name,
            judgments=judgments,
            escalated=escalated,
            unfolded_nodes=unfolded_nodes,
            total_input_tokens=delta_tokens,
            estimated_cost_usd=round(delta_cost, 6),
            latency_ms=round(latency_ms, 2),
        )

    def _run_single_pass(
        self, context_text: str, rubrics: list[RubricItem]
    ) -> dict[str, RubricJudgment]:
        rubric_payload = json.dumps([r.to_dict() for r in rubrics], ensure_ascii=False)
        user_prompt = f"RUBRICS_JSON:\n{rubric_payload}\n\nCONTEXT:\n{context_text}"
        raw = self.text_backend.complete(_JUDGE_SYSTEM_PROMPT, user_prompt, json_mode=True)
        parsed = self._parse_json_response(raw)

        out: dict[str, RubricJudgment] = {}
        items = parsed.get("judgments", []) if isinstance(parsed, dict) else []
        by_id = {str(it.get("rubric_id")): it for it in items if isinstance(it, dict)}

        for r in rubrics:
            it = by_id.get(r.id, {})
            prob_true = max(0.0, min(1.0, float(it.get("prob_true", 0.5))))
            verdict = bool(it.get("verdict", prob_true >= 0.5))
            conf = max(0.0, min(1.0, float(it.get("confidence", abs(prob_true - 0.5) * 2.0))))
            cited = [int(x) for x in it.get("cited_step_ids", []) if str(x).isdigit()]
            sufficient = bool(it.get("evidence_sufficient", True))
            req_unfold = [str(x) for x in it.get("requested_unfold_nodes", [])]
            rationale = str(it.get("rationale", ""))
            out[r.id] = RubricJudgment(
                rubric_id=r.id,
                verdict=verdict,
                prob_true=round(prob_true, 4),
                confidence=round(conf, 4),
                cited_step_ids=cited,
                evidence_sufficient=sufficient,
                requested_unfold_nodes=req_unfold,
                rationale=rationale,
            )
        return out

    def _needs_escalation(
        self,
        judgments: dict[str, RubricJudgment],
        dossier: EvalDossier,
        rubrics: list[RubricItem],
    ) -> tuple[bool, list[str]]:
        targets: list[str] = []
        need = False
        selected_set = set(dossier.selected_step_ids)

        for r in rubrics:
            j = judgments.get(r.id)
            if j is None:
                continue
            cov = dossier.rubric_coverage.get(r.id, 1.0)
            invalid_cite = bool(j.cited_step_ids) and not any(
                sid in selected_set for sid in j.cited_step_ids
            )
            if (
                not j.evidence_sufficient
                or j.confidence < self.config.escalation_confidence_threshold
                or cov < self.config.min_coverage_threshold
                or invalid_cite
            ):
                need = True
                targets.extend(j.requested_unfold_nodes)

        if need and not targets:
            targets = list(dossier.collapsed_node_ids[:2])
        return need, targets

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return {}
