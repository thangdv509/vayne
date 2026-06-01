"""
Fairness metrics for VAYNE.

Total Unfairness = Direct Bias + Proxy-mediated Bias

Metrics:
  DU_A  — Direct Unfairness: E_x[ |s(x) - s(x_{-A})| ]
            measures how much the decision changes when the protected
            attribute is masked → captures direct discrimination.

  DP    — Demographic Parity gap: |P(ŷ=1|A=0) - P(ŷ=1|A=1)|

  EO    — Equal Opportunity gap: |TPR_{A=0} - TPR_{A=1}|
            (True Positive Rate for the positive class)
"""

import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class FairnessReport:
    du:                      float   # Direct Unfairness
    dp_gap:                  float   # Demographic Parity gap
    eo_gap:                  float   # Equal Opportunity gap
    mean_score:              float   # Overall mean approval probability
    mean_score_protected:    float   # Mean score for protected group
    mean_score_non_protected: float  # Mean score for non-protected group
    n_protected:             int
    n_non_protected:         int

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in asdict(self).items()}

    def improvement_over(self, baseline: "FairnessReport") -> dict[str, float]:
        """Positive value = improvement (reduction in unfairness)."""
        return {
            "Δdu":     round(baseline.du     - self.du,     4),
            "Δdp_gap": round(baseline.dp_gap - self.dp_gap, 4),
            "Δeo_gap": round(baseline.eo_gap - self.eo_gap, 4),
        }


def _tpr(preds: list[int], labels: list[int]) -> float:
    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    if not pos_idx:
        return 0.0
    return float(np.mean([preds[i] for i in pos_idx]))


def compute_fairness(
    scores:           list[float],
    scores_masked_A:  list[float],
    labels:           list[int],
    is_protected_arr: list[bool],
    threshold:        float = 0.5,
) -> FairnessReport:
    """
    Parameters
    ----------
    scores           : s(x) baseline scores for each sample
    scores_masked_A  : s(x_{-A}) scores with sensitive attr masked
    labels           : ground-truth binary labels
    is_protected_arr : True if the sample belongs to the protected group
    threshold        : score → binary decision cutoff
    """
    n = len(scores)
    assert n == len(scores_masked_A) == len(labels) == len(is_protected_arr)

    # ── Direct Unfairness ──────────────────────────────────────────────────
    du = float(np.mean([abs(s - sm) for s, sm in zip(scores, scores_masked_A)]))

    # ── Split by group ─────────────────────────────────────────────────────
    prot_s     = [s for s, p in zip(scores, is_protected_arr) if p]
    non_prot_s = [s for s, p in zip(scores, is_protected_arr) if not p]
    prot_l     = [l for l, p in zip(labels, is_protected_arr) if p]
    non_prot_l = [l for l, p in zip(labels, is_protected_arr) if not p]

    prot_pred     = [1 if s >= threshold else 0 for s in prot_s]
    non_prot_pred = [1 if s >= threshold else 0 for s in non_prot_s]

    # ── Demographic Parity ─────────────────────────────────────────────────
    p_prot     = float(np.mean(prot_pred))     if prot_pred     else 0.0
    p_non_prot = float(np.mean(non_prot_pred)) if non_prot_pred else 0.0
    dp_gap = abs(p_prot - p_non_prot)

    # ── Equal Opportunity ──────────────────────────────────────────────────
    eo_gap = abs(_tpr(prot_pred, prot_l) - _tpr(non_prot_pred, non_prot_l))

    return FairnessReport(
        du=du,
        dp_gap=float(dp_gap),
        eo_gap=float(eo_gap),
        mean_score=float(np.mean(scores)),
        mean_score_protected=float(np.mean(prot_s)) if prot_s else 0.0,
        mean_score_non_protected=float(np.mean(non_prot_s)) if non_prot_s else 0.0,
        n_protected=len(prot_s),
        n_non_protected=len(non_prot_s),
    )
