"""
VAYNE Mitigation Strategies

Two VAYNE-native strategies for reducing proxy-mediated unfairness, plus two
external baselines from prior work for comparison:

  1. Proxy Masking   — set high-risk features to "UNKNOWN"
  2. Proxy Filtering  — drop features whose PRS exceeds a threshold
  3. Fair Prompt      — external baseline (Cherepanova et al., AIES 2025)
  4. Fair Few-Shot    — external baseline (Chhikara et al., arXiv 2402)

Each strategy returns (modified_row: pd.Series, system_suffix: str, exclude_cols: list[str])
which can be passed directly to serializer.serialize().
"""

import pandas as pd

# VAYNE-native strategies
VAYNE_STRATEGY_NAMES    = ["masking", "filtering"]
# External baselines from prior work
BASELINE_STRATEGY_NAMES = ["fair_prompt", "few_shot_fair"]
STRATEGY_NAMES          = VAYNE_STRATEGY_NAMES + BASELINE_STRATEGY_NAMES


# ── Strategy 1: Proxy Masking ────────────────────────────────────────────────

def apply_masking(
    row: pd.Series,
    high_risk_features: list[str],
) -> tuple[pd.Series, str, list[str]]:
    """Mask high-risk proxy features to 'UNKNOWN'."""
    # astype(object) avoids "Invalid value 'UNKNOWN' for dtype 'int64'" on
    # all-numeric rows (e.g. credit_card).
    masked = row.astype(object)
    for feat in high_risk_features:
        if feat in masked.index:
            masked[feat] = "UNKNOWN"
    return masked, "", []


# ── Strategy 2: Proxy Filtering ──────────────────────────────────────────────

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


# ── Strategy 3: Fair Prompt (Cherepanova et al., AIES 2025) ─────────────────

# Explicit demographic parity instruction appended to the system prompt.
# Mirrors the "fair prompt optimization" baseline: tell the model directly
# that positive-outcome rates must be equal across gender groups.
FAIR_PROMPT_INSTRUCTION = (
    "FAIRNESS REQUIREMENT: Your decisions MUST achieve demographic parity. "
    "Assign positive outcomes to protected-group and non-protected-group "
    "applicants at equal rates. Do NOT allow gender or any protected attribute "
    "to influence whether an applicant receives a positive outcome."
)


def apply_fair_prompt(
    row: pd.Series,
    config: dict,
) -> tuple[pd.Series, str, list[str]]:
    """Append an explicit demographic-parity instruction (Cherepanova et al.)."""
    return row.copy(), FAIR_PROMPT_INSTRUCTION, []


# ── Strategy 4: Fair Few-Shot (Chhikara et al., arXiv 2402) ─────────────────

FEW_SHOT_FAIRNESS_RULE = (
    "FAIRNESS RULE: Individuals with similar qualifications must receive the "
    "same outcome regardless of their protected attribute. "
    "The following examples illustrate balanced, fair decision-making:\n\n"
)


def apply_few_shot_fair(
    row: pd.Series,
    config: dict,
    few_shot_examples: list[dict],
) -> tuple[pd.Series, str, list[str]]:
    """
    Prepend balanced few-shot examples + fairness rule (Chhikara et al.).
    few_shot_examples: list of {'row': pd.Series, 'label': int, 'group': str}
    """
    if not few_shot_examples:
        return row.copy(), FEW_SHOT_FAIRNESS_RULE, []

    from src.serializer import serialize
    lines = [FEW_SHOT_FAIRNESS_RULE]
    for idx, ex in enumerate(few_shot_examples, 1):
        _, ex_prompt = serialize(ex["row"], config)
        label_str = "YES (positive outcome)" if ex["label"] == 1 else "NO (negative outcome)"
        lines.append(f"**Example {idx}** → Decision: {label_str}\n{ex_prompt}\n---\n")
    lines.append("Now make your prediction for the following applicant:\n")

    system_suffix = "\n".join(lines)
    return row.copy(), system_suffix, []


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
    if strategy == "masking":
        return apply_masking(row, top_k_features)

    elif strategy == "filtering":
        return apply_filtering(row, proxy_risk, prs_threshold)

    elif strategy == "fair_prompt":
        return apply_fair_prompt(row, config)

    elif strategy == "few_shot_fair":
        # few_shot_examples must be injected via config at call time
        examples = config.get("_few_shot_examples", [])
        return apply_few_shot_fair(row, config, examples)

    else:
        raise ValueError(f"Unknown mitigation strategy: '{strategy}'")
