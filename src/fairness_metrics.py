"""
Fairness metrics for VAYNE — aligned with paper Section 3.1.2.

Metrics:
  DU     — Direct Unfairness: E_x[ |s(x) - s(x_{-A})| ]
  SP     — Statistical Parity gap: |P(ŷ=+|ā) - P(ŷ=+|a)|   (Eq. 5)
  EOD    — Equalized Odds Difference: Σ_{y∈{+,-}} |P(ŷ=+|A=a,Y=y) - P(ŷ=+|A=ā,Y=y)|  (Eq. 6)
  PP     — Predictive Parity: |P(Y=+|ŷ=+,A=a) - P(Y=+|ŷ=+,A=ā)|  (Eq. 7)
  ABROCA — Area Between ROC Curves: ∫|ROC_ā(t) - ROC_a(t)| dt  (Eq. 8)
  TI     — Theil Index: 1/n Σ (b_i/μ) ln(b_i/μ)  (Eq. 9)
"""

import numpy as np
from dataclasses import dataclass, asdict

from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d


@dataclass
class FairnessReport:
    du:                       float   # Direct Unfairness
    sp:                       float   # Statistical Parity gap
    eod:                      float   # Equalized Odds Difference
    pp:                       float   # Predictive Parity
    abroca:                   float   # Area Between ROC Curves
    ti:                       float   # Theil Index (individual fairness)
    mean_score:               float
    mean_score_protected:     float
    mean_score_non_protected: float
    n_protected:              int
    n_non_protected:          int

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in asdict(self).items()}

    def improvement_over(self, baseline: "FairnessReport") -> dict[str, float]:
        """Positive value = improvement (reduction in unfairness)."""
        return {
            "Δdu":     round(baseline.du     - self.du,     4),
            "Δsp":     round(baseline.sp     - self.sp,     4),
            "Δeod":    round(baseline.eod    - self.eod,    4),
            "Δpp":     round(baseline.pp     - self.pp,     4),
            "Δabroca": round(baseline.abroca - self.abroca, 4),
            "Δti":     round(baseline.ti     - self.ti,     4),
        }


# ── Internal helpers ─────────────────────────────────────────────────────────

def _tpr(preds: list[int], labels: list[int]) -> float:
    """True Positive Rate = P(ŷ=+|Y=+)."""
    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    if not pos_idx:
        return 0.0
    return float(np.mean([preds[i] for i in pos_idx]))


def _fpr(preds: list[int], labels: list[int]) -> float:
    """False Positive Rate = P(ŷ=+|Y=-)."""
    neg_idx = [i for i, l in enumerate(labels) if l == 0]
    if not neg_idx:
        return 0.0
    return float(np.mean([preds[i] for i in neg_idx]))


def _precision(preds: list[int], labels: list[int]) -> float:
    """Precision = P(Y=+|ŷ=+)."""
    pos_pred_idx = [i for i, p in enumerate(preds) if p == 1]
    if not pos_pred_idx:
        return 0.0
    return float(np.mean([labels[i] for i in pos_pred_idx]))


def _abroca(
    scores_a:  list[float], labels_a:  list[int],
    scores_b:  list[float], labels_b:  list[int],
) -> float:
    """Area Between ROC Curves (Eq. 8). Returns 0.0 on any error."""
    try:
        # sklearn roc_curve requires both classes present
        if len(set(labels_a)) < 2 or len(set(labels_b)) < 2:
            return 0.0
        fpr_a, tpr_a, _ = roc_curve(labels_a, scores_a)
        fpr_b, tpr_b, _ = roc_curve(labels_b, scores_b)
        common_fpr = np.linspace(0.0, 1.0, 1000)
        f_a = interp1d(fpr_a, tpr_a, kind="linear",
                       bounds_error=False, fill_value=(0.0, 1.0))
        f_b = interp1d(fpr_b, tpr_b, kind="linear",
                       bounds_error=False, fill_value=(0.0, 1.0))
        return float(np.trapz(np.abs(f_a(common_fpr) - f_b(common_fpr)), common_fpr))
    except Exception:
        return 0.0


def _theil_index(scores: list[float]) -> float:
    """Theil Index for individual benefit inequality (Eq. 9). Clips scores to (0, 1]."""
    eps = 1e-6
    b   = np.clip(np.array(scores, dtype=float), eps, None)
    mu  = float(np.mean(b))
    if mu <= 0:
        return 0.0
    ratios = b / mu
    return float(np.mean(ratios * np.log(ratios)))


# ── Main metric computation ──────────────────────────────────────────────────

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
    scores_masked_A  : s(x_{-A}) scores with sensitive attr masked (for DU)
    labels           : ground-truth binary labels
    is_protected_arr : True if the sample belongs to the protected group (a)
    threshold        : score → binary decision cutoff
    """
    n = len(scores)
    assert n == len(scores_masked_A) == len(labels) == len(is_protected_arr)

    # ── Direct Unfairness ────────────────────────────────────────────────────
    du = float(np.mean([abs(s - sm) for s, sm in zip(scores, scores_masked_A)]))

    # ── Split by group ───────────────────────────────────────────────────────
    prot_s     = [s for s, p in zip(scores, is_protected_arr) if p]
    non_prot_s = [s for s, p in zip(scores, is_protected_arr) if not p]
    prot_l     = [l for l, p in zip(labels, is_protected_arr) if p]
    non_prot_l = [l for l, p in zip(labels, is_protected_arr) if not p]

    prot_pred     = [1 if s >= threshold else 0 for s in prot_s]
    non_prot_pred = [1 if s >= threshold else 0 for s in non_prot_s]

    # ── SP: |P(ŷ=+|ā) - P(ŷ=+|a)| (Eq. 5) ─────────────────────────────────
    p_prot     = float(np.mean(prot_pred))     if prot_pred     else 0.0
    p_non_prot = float(np.mean(non_prot_pred)) if non_prot_pred else 0.0
    sp = abs(p_non_prot - p_prot)

    # ── EOD: |ΔTPR| + |ΔFPR| (Eq. 6) ───────────────────────────────────────
    tpr_prot     = _tpr(prot_pred, prot_l)
    tpr_non_prot = _tpr(non_prot_pred, non_prot_l)
    fpr_prot     = _fpr(prot_pred, prot_l)
    fpr_non_prot = _fpr(non_prot_pred, non_prot_l)
    eod = abs(tpr_prot - tpr_non_prot) + abs(fpr_prot - fpr_non_prot)

    # ── PP: |precision(ā) - precision(a)| (Eq. 7) ───────────────────────────
    prec_prot     = _precision(prot_pred, prot_l)
    prec_non_prot = _precision(non_prot_pred, non_prot_l)
    pp = abs(prec_non_prot - prec_prot)

    # ── ABROCA (Eq. 8) ──────────────────────────────────────────────────────
    abroca = _abroca(prot_s, prot_l, non_prot_s, non_prot_l)

    # ── Theil Index (Eq. 9) ─────────────────────────────────────────────────
    ti = _theil_index(scores)

    return FairnessReport(
        du=du,
        sp=sp,
        eod=eod,
        pp=pp,
        abroca=abroca,
        ti=ti,
        mean_score=float(np.mean(scores)),
        mean_score_protected=float(np.mean(prot_s))     if prot_s     else 0.0,
        mean_score_non_protected=float(np.mean(non_prot_s)) if non_prot_s else 0.0,
        n_protected=len(prot_s),
        n_non_protected=len(non_prot_s),
    )
