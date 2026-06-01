"""
VAYNE Reporting

Saves and prints results from a VAYNEResult:
  - JSON: full proxy analysis + fairness metrics
  - CSV: risk summary per feature
  - CSV: mitigation comparison table
  - Console: proxy path graph + fairness table
"""

import json
import logging
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from src.config import RESULTS_DIR

logger = logging.getLogger(__name__)


def _safe_key(k):
    """Convert tuple keys to strings for JSON serialisation."""
    return str(k) if not isinstance(k, str) else k


def save_results(result, verbose: bool = True) -> None:
    tag = f"{result.dataset}__{result.model.replace('/', '_')}"

    # ── JSON dump ────────────────────────────────────────────────────────────
    payload = {
        "dataset":   result.dataset,
        "model":     result.model,
        "n_samples": result.n_samples,
        "proxy_capacity":   result.proxy_capacity,
        "proxy_use":        result.proxy_use,
        "proxy_risk":       result.proxy_risk,
        "proxy_interaction": result.proxy_interaction,
        "proxy_graph":      result.proxy_graph_dict,
        "proxy_paths":      result.proxy_paths,
        "proxy_ascii":      result.proxy_ascii,
        "risk_summary":     result.risk_summary,
        "interaction_summary": result.interaction_summary,
        "baseline_fairness": result.baseline_report.to_dict() if result.baseline_report else {},
        "mitigation_results": [
            {
                "strategy":    m.strategy,
                "fairness":    m.report.to_dict(),
                "improvement": m.improvement,
            }
            for m in result.mitigation_results
        ],
    }

    json_path = RESULTS_DIR / f"{tag}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Saved JSON → %s", json_path)

    # ── Proxy risk CSV ───────────────────────────────────────────────────────
    if result.risk_summary:
        risk_path = RESULTS_DIR / f"{tag}_proxy_risk.csv"
        pd.DataFrame(result.risk_summary).to_csv(risk_path, index=False)
        logger.info("Saved proxy risk CSV → %s", risk_path)

    # ── Mitigation comparison CSV ────────────────────────────────────────────
    if result.mitigation_results:
        rows = []
        base = result.baseline_report
        rows.append({
            "strategy": "baseline",
            "du":     round(base.du,     4),
            "dp_gap": round(base.dp_gap, 4),
            "eo_gap": round(base.eo_gap, 4),
            "Δdu": 0, "Δdp_gap": 0, "Δeo_gap": 0,
        })
        for m in result.mitigation_results:
            rows.append({
                "strategy": m.strategy,
                "du":     round(m.report.du,     4),
                "dp_gap": round(m.report.dp_gap, 4),
                "eo_gap": round(m.report.eo_gap, 4),
                **m.improvement,
            })
        mit_path = RESULTS_DIR / f"{tag}_mitigation.csv"
        pd.DataFrame(rows).to_csv(mit_path, index=False)
        logger.info("Saved mitigation CSV → %s", mit_path)

    if verbose:
        _print_results(result)


def _print_results(result) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  VAYNE Results  |  {result.dataset}  |  {result.model}")
    print(sep)

    # ── Proxy risk table ─────────────────────────────────────────────────────
    if result.risk_summary:
        print("\n── Proxy Risk Scores (PC × PU) ──")
        print(tabulate(result.risk_summary[:10], headers="keys",
                       tablefmt="rounded_outline", floatfmt=".4f"))

    # ── Interactions ─────────────────────────────────────────────────────────
    if result.interaction_summary:
        top_int = [r for r in result.interaction_summary if r["interaction"] > 0][:5]
        if top_int:
            print("\n── Top Proxy Interactions (INT_ij > 0) ──")
            print(tabulate(top_int, headers="keys",
                           tablefmt="rounded_outline", floatfmt=".4f"))

    # ── Proxy path graph ─────────────────────────────────────────────────────
    print("\n── Proxy Path Graph ──")
    print(result.proxy_ascii or "(no paths above threshold)")

    # ── Fairness comparison ───────────────────────────────────────────────────
    print("\n── Fairness Metrics: Baseline vs Mitigation ──")
    base = result.baseline_report
    table_rows = [["baseline",
                   f"{base.du:.4f}", f"{base.dp_gap:.4f}", f"{base.eo_gap:.4f}",
                   "—", "—", "—"]]
    for m in result.mitigation_results:
        table_rows.append([
            m.strategy,
            f"{m.report.du:.4f}",
            f"{m.report.dp_gap:.4f}",
            f"{m.report.eo_gap:.4f}",
            f"{m.improvement['Δdu']:+.4f}",
            f"{m.improvement['Δdp_gap']:+.4f}",
            f"{m.improvement['Δeo_gap']:+.4f}",
        ])
    print(tabulate(table_rows,
                   headers=["strategy", "DU", "DP gap", "EO gap", "ΔDU", "ΔDP", "ΔEO"],
                   tablefmt="rounded_outline"))
    print()
