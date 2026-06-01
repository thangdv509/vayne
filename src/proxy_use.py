"""
Proxy Use: PU_j = E_x [ |s(x) - s(x_{-j})| ]

Measures how much the LLM's decision changes when feature X_j is masked.
High PU_j → the LLM heavily relies on this feature.

Also computes pairwise PU_ij = E_x [ |s(x) - s(x_{-i,-j})| ]
needed for interaction analysis.
"""

import logging
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Optional

from src.llm_client import score_row, extract_p_yes
from src.serializer import serialize

logger = logging.getLogger(__name__)


def _score_masked(
    row: pd.Series,
    config: dict,
    model: str,
    mask_cols: list[str],
) -> Optional[float]:
    """Score a row with mask_cols set to 'UNKNOWN'."""
    masked_row = row.copy()
    for col in mask_cols:
        if col in masked_row.index:
            masked_row[col] = "UNKNOWN"

    system, prompt = serialize(masked_row, config)
    return score_row(prompt, model, system)


def compute_single_ablations(
    df: pd.DataFrame,
    config: dict,
    model: str,
    feature_cols: list[str],
    baseline_scores: list[float],
    max_workers: int = 8,
) -> dict[str, list[float]]:
    """
    For each feature j, compute s(x_{-j}) for all rows.

    Returns dict: feature → list of masked scores (one per row).
    """
    n = len(df)
    masked_scores: dict[str, list[Optional[float]]] = {f: [None] * n for f in feature_cols}

    tasks = [(i, row, feat) for feat in feature_cols for i, row in df.iterrows()]

    def _run(args):
        i, row, feat = args
        return feat, i, _score_masked(row, config, model, [feat])

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks),
                        desc="Single ablations", leave=False):
            try:
                feat, i, score = fut.result()
                if score is not None:
                    masked_scores[feat][i] = score
                else:
                    masked_scores[feat][i] = baseline_scores[i]
            except Exception as exc:
                logger.warning("Ablation failed: %s", exc)

    # Fill any remaining None with baseline score
    for feat in feature_cols:
        for idx in range(n):
            if masked_scores[feat][idx] is None:
                masked_scores[feat][idx] = baseline_scores[idx]

    return {f: [v for v in masked_scores[f]] for f in feature_cols}


def compute_pairwise_ablations(
    df: pd.DataFrame,
    config: dict,
    model: str,
    top_features: list[str],
    baseline_scores: list[float],
    max_workers: int = 8,
) -> dict[tuple[str, str], list[float]]:
    """
    For each pair (i, j) in top_features, compute s(x_{-i,-j}) for all rows.

    Returns dict: (feat_i, feat_j) → list of masked scores.
    """
    pairs = [(top_features[i], top_features[j])
             for i in range(len(top_features))
             for j in range(i + 1, len(top_features))]

    n = len(df)
    pairwise_scores: dict[tuple, list[Optional[float]]] = {p: [None] * n for p in pairs}

    tasks = [(row_idx, row, pair) for pair in pairs for row_idx, row in df.iterrows()]

    def _run(args):
        row_idx, row, pair = args
        return pair, row_idx, _score_masked(row, config, model, list(pair))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks),
                        desc="Pairwise ablations", leave=False):
            try:
                pair, row_idx, score = fut.result()
                if score is not None:
                    pairwise_scores[pair][row_idx] = score
                else:
                    pairwise_scores[pair][row_idx] = baseline_scores[row_idx]
            except Exception as exc:
                logger.warning("Pairwise ablation failed: %s", exc)

    for pair in pairs:
        for idx in range(n):
            if pairwise_scores[pair][idx] is None:
                pairwise_scores[pair][idx] = baseline_scores[idx]

    return {p: [v for v in pairwise_scores[p]] for p in pairs}


def compute_proxy_use(
    baseline_scores: list[float],
    single_masked_scores: dict[str, list[float]],
) -> dict[str, float]:
    """PU_j = E_x[ |s(x) - s(x_{-j})| ]"""
    result = {}
    for feat, masked in single_masked_scores.items():
        diffs = [abs(b - m) for b, m in zip(baseline_scores, masked)]
        result[feat] = float(np.mean(diffs))
    return result


def compute_pairwise_proxy_use(
    baseline_scores: list[float],
    pairwise_masked_scores: dict[tuple[str, str], list[float]],
) -> dict[tuple[str, str], float]:
    """PU_ij = E_x[ |s(x) - s(x_{-i,-j})| ]"""
    result = {}
    for pair, masked in pairwise_masked_scores.items():
        diffs = [abs(b - m) for b, m in zip(baseline_scores, masked)]
        result[pair] = float(np.mean(diffs))
    return result
