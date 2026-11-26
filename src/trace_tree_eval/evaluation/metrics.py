"""Decision Sufficiency & reliability metrics (JS/KL divergence, FPR, F1, consistency@k)."""

from __future__ import annotations

import math
from typing import Any


def _clip_prob(p: float, eps: float = 1e-6) -> float:
    return max(eps, min(1.0 - eps, float(p)))


def kl_divergence(p: float, q: float, eps: float = 1e-6) -> float:
    """Binary smoothed KL divergence D_KL(P || Q) in bits."""
    p_c = _clip_prob(p, eps)
    q_c = _clip_prob(q, eps)
    return p_c * math.log2(p_c / q_c) + (1.0 - p_c) * math.log2((1.0 - p_c) / (1.0 - q_c))


def js_divergence(p: float, q: float, eps: float = 1e-6) -> float:
    """Symmetric binary Jensen-Shannon divergence D_JS(P || Q) in bits, bounded in [0, 1]."""
    p_c = _clip_prob(p, eps)
    q_c = _clip_prob(q, eps)
    m = 0.5 * (p_c + q_c)
    return max(0.0, min(1.0, 0.5 * kl_divergence(p_c, m, eps) + 0.5 * kl_divergence(q_c, m, eps)))


def decision_flip_rate(ref_verdicts: list[bool], test_verdicts: list[bool]) -> float:
    """Fraction of rubric verdicts that flipped relative to the reference."""
    if not ref_verdicts:
        return 0.0
    flips = sum(1 for r, t in zip(ref_verdicts, test_verdicts) if bool(r) != bool(t))
    return round(flips / len(ref_verdicts), 4)


def false_pass_rate(ref_verdicts: list[bool], test_verdicts: list[bool]) -> float:
    """Fraction of reference-failed items (False) misclassified as Pass (True)."""
    negatives = [(r, t) for r, t in zip(ref_verdicts, test_verdicts) if not r]
    if not negatives:
        return 0.0
    fp = sum(1 for _, t in negatives if t)
    return round(fp / len(negatives), 4)


def false_fail_rate(ref_verdicts: list[bool], test_verdicts: list[bool]) -> float:
    """Fraction of reference-passed items (True) misclassified as Fail (False)."""
    positives = [(r, t) for r, t in zip(ref_verdicts, test_verdicts) if r]
    if not positives:
        return 0.0
    ff = sum(1 for _, t in positives if not t)
    return round(ff / len(positives), 4)


def binary_f1(ref_verdicts: list[bool], test_verdicts: list[bool]) -> float:
    """F1 score treating True as positive class (or macro-like fallback when all False)."""
    if not ref_verdicts:
        return 0.0
    tp = sum(1 for r, t in zip(ref_verdicts, test_verdicts) if r and t)
    fp = sum(1 for r, t in zip(ref_verdicts, test_verdicts) if (not r) and t)
    fn = sum(1 for r, t in zip(ref_verdicts, test_verdicts) if r and (not t))
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    denom = 2 * tp + fp + fn
    return round((2 * tp / denom) if denom > 0 else 0.0, 4)


def consistency_k(repeated_verdicts: list[list[bool]]) -> float:
    """Eugene Yan / tau-bench inspired pass^k consistency across k repeated judge runs.

    `repeated_verdicts` is a list of items, each holding k boolean verdicts across runs.
    """
    if not repeated_verdicts:
        return 0.0
    unanimous = sum(1 for runs in repeated_verdicts if runs and len(set(runs)) == 1)
    return round(unanimous / len(repeated_verdicts), 4)


def compute_sufficiency_report(
    ref_verdicts: dict[str, bool],
    test_verdicts: dict[str, bool],
    ref_probs: dict[str, float] | None = None,
    test_probs: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute full Decision Sufficiency report across aligned rubric IDs."""
    rids = sorted(ref_verdicts.keys())
    rv = [bool(ref_verdicts[k]) for k in rids]
    tv = [bool(test_verdicts.get(k, False)) for k in rids]

    agreement = round(sum(1 for r, t in zip(rv, tv) if r == t) / max(1, len(rids)), 4)
    mean_js = 0.0
    mean_kl = 0.0
    if ref_probs and test_probs and rids:
        mean_js = round(
            sum(js_divergence(ref_probs[k], test_probs.get(k, 0.5)) for k in rids) / len(rids),
            4,
        )
        mean_kl = round(
            sum(kl_divergence(ref_probs[k], test_probs.get(k, 0.5)) for k in rids) / len(rids),
            4,
        )

    return {
        "agreement": agreement,
        "f1": binary_f1(rv, tv),
        "decision_flip_rate": decision_flip_rate(rv, tv),
        "false_pass_rate": false_pass_rate(rv, tv),
        "false_fail_rate": false_fail_rate(rv, tv),
        "mean_js_divergence": mean_js,
        "mean_kl_divergence": mean_kl,
    }
