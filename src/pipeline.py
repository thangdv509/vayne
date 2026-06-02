"""
VAYNE Pipeline — end-to-end orchestration.

Flow per experiment (dataset × model):
  1. Load N balanced samples
  2. Baseline inference  → s(x) for all samples
  3. Sensitive-attr masked inference → s(x_{-A}) for direct-unfairness
  4. Compute Proxy Capacity  (statistical, no LLM)
  5. Single-feature ablations → s(x_{-j}) for each feature j
  6. Compute Proxy Use, Proxy Risk
  7. Select top-K proxies; compute pairwise ablations
  8. Compute Proxy Interaction
  9. Build Proxy Path Graph
 10. Compute baseline fairness metrics
 11. For each of 4 mitigation strategies:
       a. Apply strategy to each row
       b. Re-run LLM inference on modified prompts
       c. Compute post-mitigation fairness metrics
 12. Return VAYNEResult
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.config import (
    DECISION_THRESHOLD, MAX_WORKERS, TOP_K_PROXY, PRS_FILTER_THRESHOLD,
    TAU_CAPACITY, TAU_USE, TAU_INTERACTION, GRAPH_THRESHOLD_PERCENTILE,
)
from src.llm_client import score_row
from src.data_loader import load_dataset, get_feature_cols, get_all_feature_cols, is_protected, get_label
from src.serializer import serialize
from src.proxy_capacity import compute_proxy_capacity, rank_by_capacity
from src.proxy_use import (
    compute_single_ablations, compute_pairwise_ablations,
    compute_proxy_use, compute_pairwise_proxy_use,
)
from src.proxy_risk import compute_proxy_risk, top_k_proxy_risk, proxy_risk_summary
from src.proxy_interaction import compute_proxy_interaction, interaction_summary
from src.proxy_graph import build_proxy_graph, get_proxy_paths, graph_to_dict, graph_to_ascii
from src.fairness_metrics import compute_fairness, FairnessReport
from src.mitigation import apply_strategy, STRATEGY_NAMES

logger = logging.getLogger(__name__)


@dataclass
class MitigationResult:
    strategy:    str
    scores:      list[float]
    report:      FairnessReport
    improvement: dict[str, float]


@dataclass
class VAYNEResult:
    dataset:  str
    model:    str
    n_samples: int

    # ── Proxy Analysis ──────────────────────────────────────────────────────
    proxy_capacity:   dict[str, float]     = field(default_factory=dict)
    proxy_use:        dict[str, float]     = field(default_factory=dict)
    proxy_risk:       dict[str, float]     = field(default_factory=dict)
    proxy_interaction: dict               = field(default_factory=dict)
    proxy_graph_dict: dict                = field(default_factory=dict)
    proxy_paths:      list                = field(default_factory=list)
    proxy_ascii:      str                 = ""
    risk_summary:     list[dict]          = field(default_factory=list)
    interaction_summary: list[dict]       = field(default_factory=list)

    # ── Adaptive thresholds used for Proxy Path Graph ───────────────────────
    tau_capacity:    float                     = 0.0
    tau_use:         float                     = 0.0
    tau_interaction: float                     = 0.0

    # ── Proxy Path Graph object (NetworkX DiGraph) ──────────────────────────
    proxy_graph_obj:  Optional[object]         = None

    # ── Baseline Fairness ───────────────────────────────────────────────────
    baseline_report:  Optional[FairnessReport] = None
    baseline_scores:  list[float]          = field(default_factory=list)

    # ── Per-strategy Results ────────────────────────────────────────────────
    mitigation_results: list[MitigationResult] = field(default_factory=list)

    # ── Raw sample metadata ─────────────────────────────────────────────────
    labels:           list[int]           = field(default_factory=list)
    is_protected_arr: list[bool]          = field(default_factory=list)


def _adaptive_thresholds(
    proxy_capacity:   dict[str, float],
    proxy_use:        dict[str, float],
    proxy_interaction: dict,
) -> tuple[float, float, float]:
    """
    Compute adaptive percentile thresholds for the Proxy Path Graph edges,
    as described in paper Section 4.4.

    Falls back to config defaults when fewer than 4 values are available.
    """
    import numpy as np

    pc_vals  = list(proxy_capacity.values())
    pu_vals  = list(proxy_use.values())
    int_vals = [v for v in proxy_interaction.values() if v > 0]

    pct  = GRAPH_THRESHOLD_PERCENTILE
    tau1 = float(np.percentile(pc_vals,  pct)) if len(pc_vals)  >= 4 else TAU_CAPACITY
    tau2 = float(np.percentile(pu_vals,  pct)) if len(pu_vals)  >= 4 else TAU_USE
    tau3 = float(np.percentile(int_vals, pct)) if len(int_vals) >= 4 else TAU_INTERACTION

    # Guard against zero thresholds (edge case when all values are identical)
    tau1 = max(tau1, 1e-6)
    tau2 = max(tau2, 1e-6)
    tau3 = max(tau3, 1e-6)

    logger.info("Adaptive thresholds (%dth pct) — τ_PC=%.4f  τ_PU=%.4f  τ_INT=%.4f",
                pct, tau1, tau2, tau3)
    return tau1, tau2, tau3


def _run_baseline_inference(
    df: pd.DataFrame,
    config: dict,
    model: str,
    sensitive_col: str,
) -> tuple[list[float], list[float]]:
    """
    Run baseline and sensitive-masked inference for all rows in parallel.

    Returns (baseline_scores, masked_A_scores).
    """
    n = len(df)
    baseline  = [None] * n
    masked_A  = [None] * n

    def _infer(i, row, mask_attr):
        row_to_use = row.copy()
        if mask_attr:
            row_to_use[sensitive_col] = "UNKNOWN"
        system, prompt = serialize(row_to_use, config)
        return score_row(prompt, model, system)

    tasks = [(i, row, False) for i, row in df.iterrows()] + \
            [(i, row, True)  for i, row in df.iterrows()]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_infer, i, row, mask): (i, mask)
                   for i, row, mask in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks),
                        desc="Baseline inference", leave=False):
            i, is_masked = futures[fut]
            try:
                score = fut.result()
                score = score if score is not None else 0.5
                if is_masked:
                    masked_A[i] = score
                else:
                    baseline[i] = score
            except Exception as exc:
                logger.warning("Inference failed row %d: %s", i, exc)
                if is_masked:
                    masked_A[i] = 0.5
                else:
                    baseline[i] = 0.5

    return (
        [v if v is not None else 0.5 for v in baseline],
        [v if v is not None else 0.5 for v in masked_A],
    )


def _run_mitigation_inference(
    df: pd.DataFrame,
    config: dict,
    model: str,
    strategy: str,
    proxy_risk: dict[str, float],
    top_k_features: list[str],
    prs_threshold: float,
    masked_A_scores: list[float],
) -> list[float]:
    """Apply one mitigation strategy to all rows and collect scores."""
    n = len(df)
    scores = [None] * n

    def _infer(i, row):
        mod_row, sys_suffix, excl_cols = apply_strategy(
            strategy, row, config, proxy_risk, top_k_features, prs_threshold
        )
        system, prompt = serialize(mod_row, config,
                                   exclude_cols=excl_cols,
                                   system_suffix=sys_suffix)
        return score_row(prompt, model, system)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_infer, i, row): i for i, row in df.iterrows()}
        for fut in tqdm(as_completed(futures), total=n,
                        desc=f"Mitigation [{strategy}]", leave=False):
            i = futures[fut]
            try:
                score = fut.result()
                scores[i] = score if score is not None else 0.5
            except Exception as exc:
                logger.warning("Mitigation inference failed row %d (%s): %s",
                               i, strategy, exc)
                scores[i] = 0.5

    return [v if v is not None else 0.5 for v in scores]


def run_experiment(
    dataset_name: str,
    model: str,
    config: dict,
    n_samples: int,
) -> VAYNEResult:
    result = VAYNEResult(dataset=dataset_name, model=model, n_samples=n_samples)

    # ── Step 1: Load data ───────────────────────────────────────────────────
    logger.info("Loading dataset '%s' (%d samples)", dataset_name, n_samples)
    df = load_dataset(config, n_samples=n_samples)
    result.n_samples = len(df)

    sensitive_col  = config["sensitive_attrs"][0]
    feature_cols   = get_feature_cols(df, config)      # no sensitive attr, no target
    all_feat_cols  = get_all_feature_cols(df, config)  # includes sensitive attr

    result.labels           = [get_label(row, config) for _, row in df.iterrows()]
    result.is_protected_arr = [is_protected(row, config) for _, row in df.iterrows()]

    # ── Step 2 & 3: Baseline + sensitive-masked inference ───────────────────
    logger.info("Running baseline inference...")
    baseline_scores, masked_A_scores = _run_baseline_inference(
        df, config, model, sensitive_col
    )
    result.baseline_scores = baseline_scores

    # ── Step 4: Proxy Capacity (statistical, no LLM) ────────────────────────
    logger.info("Computing proxy capacity...")
    # Use feature_cols (excludes sensitive attr) to avoid I(A;A)=H(A) inflating PC
    result.proxy_capacity = compute_proxy_capacity(df, feature_cols, sensitive_col)

    # ── Step 5: Single-feature ablations → Proxy Use ────────────────────────
    logger.info("Running single-feature ablations (%d features × %d samples)...",
                len(feature_cols), len(df))
    single_masked = compute_single_ablations(
        df, config, model, feature_cols, baseline_scores, MAX_WORKERS
    )
    result.proxy_use = compute_proxy_use(baseline_scores, single_masked)

    # ── Step 6: Proxy Risk ──────────────────────────────────────────────────
    result.proxy_risk    = compute_proxy_risk(result.proxy_capacity, result.proxy_use)
    top_k                = top_k_proxy_risk(result.proxy_risk, TOP_K_PROXY)
    result.risk_summary  = proxy_risk_summary(result.proxy_capacity,
                                               result.proxy_use, result.proxy_risk)
    logger.info("Top proxy risks: %s", top_k)

    # ── Step 7: Pairwise ablations → Proxy Interaction ──────────────────────
    if len(top_k) >= 2:
        logger.info("Running pairwise ablations for top-%d features...", len(top_k))
        pairwise_masked = compute_pairwise_ablations(
            df, config, model, top_k, baseline_scores, MAX_WORKERS
        )
        pairwise_pu        = compute_pairwise_proxy_use(baseline_scores, pairwise_masked)
        interaction        = compute_proxy_interaction(result.proxy_use, pairwise_pu)
        result.proxy_interaction   = {str(k): v for k, v in interaction.items()}
        result.interaction_summary = interaction_summary(interaction)
    else:
        interaction = {}

    # ── Step 8: Proxy Path Graph (adaptive 75th-percentile thresholds) ──────
    logger.info("Building proxy path graph...")
    tau1, tau2, tau3 = _adaptive_thresholds(
        result.proxy_capacity, result.proxy_use, interaction
    )
    result.tau_capacity    = round(tau1, 6)
    result.tau_use         = round(tau2, 6)
    result.tau_interaction = round(tau3, 6)

    G = build_proxy_graph(
        sensitive_attr=sensitive_col,
        proxy_capacity=result.proxy_capacity,
        proxy_use=result.proxy_use,
        proxy_interaction=interaction,
        tau_capacity=tau1,
        tau_use=tau2,
        tau_interaction=tau3,
    )
    result.proxy_graph_dict = graph_to_dict(G)
    result.proxy_paths      = get_proxy_paths(G, sensitive_col)
    result.proxy_ascii      = graph_to_ascii(G, sensitive_col)
    result.proxy_graph_obj  = G

    # ── Step 9: Baseline Fairness Metrics ────────────────────────────────────
    logger.info("Computing baseline fairness metrics...")
    result.baseline_report = compute_fairness(
        scores=baseline_scores,
        scores_masked_A=masked_A_scores,
        labels=result.labels,
        is_protected_arr=result.is_protected_arr,
        threshold=DECISION_THRESHOLD,
    )

    # ── Step 10: Mitigation Strategies ──────────────────────────────────────
    for strategy in STRATEGY_NAMES:
        logger.info("Applying mitigation strategy: %s", strategy)
        mit_scores = _run_mitigation_inference(
            df, config, model, strategy,
            result.proxy_risk, top_k, PRS_FILTER_THRESHOLD,
            masked_A_scores,
        )
        # Re-run sensitive-masked inference on mitigated prompts for DU measure
        # (use masked_A_scores from baseline as approximation to save LLM calls)
        mit_report = compute_fairness(
            scores=mit_scores,
            scores_masked_A=masked_A_scores,
            labels=result.labels,
            is_protected_arr=result.is_protected_arr,
            threshold=DECISION_THRESHOLD,
        )
        improvement = mit_report.improvement_over(result.baseline_report)
        result.mitigation_results.append(
            MitigationResult(
                strategy=strategy,
                scores=mit_scores,
                report=mit_report,
                improvement=improvement,
            )
        )

    return result
