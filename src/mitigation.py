"""
VAYNE Mitigation Strategies

Four strategies for reducing proxy-mediated unfairness:

  1. Proxy Masking    — set high-risk features to "UNKNOWN"
  2. Proxy Abstraction — replace precise values with coarser categories
  3. Prompt Debiasing  — inject a fairness instruction into the system prompt
  4. Proxy Filtering   — drop features whose PRS exceeds a threshold

Each strategy returns (modified_row: pd.Series, system_suffix: str, exclude_cols: list[str])
which can be passed directly to serializer.serialize().
"""

import pandas as pd

DEBIASING_INSTRUCTION = (
    "IMPORTANT: Do not infer demographic characteristics "
    "(such as gender, race, or ethnicity) from geographic, occupational, "
    "educational, or other potentially correlated information. "
    "Evaluate this application solely on its direct financial and professional merits."
)

STRATEGY_NAMES = ["masking", "abstraction", "debiasing", "filtering"]


# ── Strategy 1: Proxy Masking ────────────────────────────────────────────────

def apply_masking(
    row: pd.Series,
    high_risk_features: list[str],
) -> tuple[pd.Series, str, list[str]]:
    """Mask high-risk proxy features to 'UNKNOWN'."""
    masked = row.copy()
    for feat in high_risk_features:
        if feat in masked.index:
            masked[feat] = "UNKNOWN"
    return masked, "", []


# ── Strategy 2: Proxy Abstraction ───────────────────────────────────────────

def apply_abstraction(
    row: pd.Series,
    features_to_abstract: list[str],
    abstraction_map: dict[str, dict[str, str]],
) -> tuple[pd.Series, str, list[str]]:
    """Replace precise feature values with coarser, less discriminating categories."""
    abstracted = row.copy()
    for feat in features_to_abstract:
        if feat in abstracted.index:
            feat_key = feat.lower()
            feat_map = abstraction_map.get(feat_key, abstraction_map.get(feat, {}))
            val = str(abstracted[feat])
            abstracted[feat] = feat_map.get(val, val)
    return abstracted, "", []


# ── Strategy 3: Prompt Debiasing ─────────────────────────────────────────────

def apply_debiasing(
    row: pd.Series,
) -> tuple[pd.Series, str, list[str]]:
    """Append a fairness instruction to the system prompt; leave row unchanged."""
    return row.copy(), DEBIASING_INSTRUCTION, []


# ── Strategy 4: Proxy Filtering ──────────────────────────────────────────────

def apply_filtering(
    row: pd.Series,
    proxy_risk: dict[str, float],
    prs_threshold: float,
) -> tuple[pd.Series, str, list[str]]:
    """
    Exclude columns whose proxy risk score exceeds prs_threshold.
    The row itself is not modified; instead the excluded column list is returned
    and passed to the serializer so those columns are omitted from the prompt.
    """
    exclude = [feat for feat, prs in proxy_risk.items() if prs >= prs_threshold]
    return row.copy(), "", exclude


# ── Unified dispatcher ───────────────────────────────────────────────────────

def apply_strategy(
    strategy: str,
    row: pd.Series,
    config: dict,
    proxy_risk: dict[str, float],
    top_k_features: list[str],
    prs_threshold: float = 0.05,
) -> tuple[pd.Series, str, list[str]]:
    """
    Dispatch to the requested strategy.

    Returns (modified_row, system_suffix, exclude_cols).
    """
    abstraction_map = config.get("abstraction_map", {})

    if strategy == "masking":
        return apply_masking(row, top_k_features)

    elif strategy == "abstraction":
        abstractable = [f for f in top_k_features if f.lower() in abstraction_map
                        or f in abstraction_map]
        return apply_abstraction(row, abstractable, abstraction_map)

    elif strategy == "debiasing":
        return apply_debiasing(row)

    elif strategy == "filtering":
        return apply_filtering(row, proxy_risk, prs_threshold)

    else:
        raise ValueError(f"Unknown mitigation strategy: '{strategy}'")
