"""
VAYNE — Cross-Dataset and Cross-Strategy Synthesis

Combines per-dataset `*_mitigation.csv` and `*_proxy_risk.csv` files
(produced by main.py / src/reporting.py) across multiple datasets into:

  - Direct & Baseline Fairness comparison across datasets   (table + figure)
  - Top proxy-risk features across datasets                 (table)
  - Cross-Dataset x Cross-Strategy mitigation improvements   (table + heatmap)
  - Average-per-strategy ranking across all datasets         (table + figure)

Usage:
  python3 synthesize_results.py --results-dir result_0107
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FONT_TITLE, FONT_LABEL, FONT_TICK, FONT_LEGEND = 18, 15, 13, 13

_DATASET_TITLES = {
    "credit_approval": "Credit Approval",
    "credit_card":     "Credit Card",
    "german_credit":   "German Credit",
}


def _dataset_title(slug: str) -> str:
    return _DATASET_TITLES.get(slug, slug.replace("_", " ").title())


def _split_tag(stem: str, suffix: str) -> tuple[str, str]:
    dataset, model = stem.replace(suffix, "").split("__", 1)
    return dataset, model


def load_mitigation(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*_mitigation.csv")):
        dataset, model = _split_tag(path.stem, "_mitigation")
        df = pd.read_csv(path)
        df.insert(0, "model", model)
        df.insert(0, "dataset", dataset)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No *_mitigation.csv found in {results_dir}")
    return pd.concat(rows, ignore_index=True)


def load_proxy_risk(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*_proxy_risk.csv")):
        dataset, model = _split_tag(path.stem, "_proxy_risk")
        df = pd.read_csv(path)
        df.insert(0, "model", model)
        df.insert(0, "dataset", dataset)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No *_proxy_risk.csv found in {results_dir}")
    return pd.concat(rows, ignore_index=True)


# ── Tables ──────────────────────────────────────────────────────────────────

def direct_and_baseline_table(mitigation_df: pd.DataFrame) -> pd.DataFrame:
    base = mitigation_df[mitigation_df["strategy"] == "baseline"].copy()
    base["dataset"] = base["dataset"].map(_dataset_title)
    return base[["dataset", "du", "sp", "eod", "pp", "ti"]].rename(
        columns={"du": "DU", "sp": "SP", "eod": "EOD", "pp": "PP", "ti": "TI"}
    ).reset_index(drop=True)


def top_proxy_table(proxy_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    out = (
        proxy_df.sort_values(["dataset", "proxy_risk"], ascending=[True, False])
        .groupby("dataset", group_keys=False)
        .head(top_n)
        .copy()
    )
    out["dataset"] = out["dataset"].map(_dataset_title)
    return out[["dataset", "feature", "proxy_capacity", "proxy_use", "proxy_risk"]].reset_index(drop=True)


def cross_strategy_summary(mitigation_df: pd.DataFrame) -> pd.DataFrame:
    strat = mitigation_df[mitigation_df["strategy"] != "baseline"]
    agg = strat.groupby("strategy")[["Δsp", "Δeod", "Δpp", "Δti", "accuracy", "f1"]].mean()
    agg["avg_fairness_gain"] = agg[["Δsp", "Δeod", "Δpp", "Δti"]].mean(axis=1)
    return agg.sort_values("avg_fairness_gain", ascending=False).reset_index()


# ── Figures ─────────────────────────────────────────────────────────────────

def plot_baseline_fairness(table: pd.DataFrame, out_path: Path) -> None:
    metrics = ["DU", "SP", "EOD", "PP", "TI"]
    x = np.arange(len(table))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10.colors
    for i, m in enumerate(metrics):
        ax.bar(x + (i - len(metrics) / 2) * width, table[m], width, label=m, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(table["dataset"], fontsize=FONT_TICK)
    ax.set_ylabel("Score", fontsize=FONT_LABEL)
    ax.set_title("Direct & Baseline Fairness Across Datasets", fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(fontsize=FONT_LEGEND, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_heatmap(mitigation_df: pd.DataFrame, out_path: Path) -> None:
    strat = mitigation_df[mitigation_df["strategy"] != "baseline"].copy()
    strat["dataset"] = strat["dataset"].map(_dataset_title)
    pivot = strat.pivot_table(index="strategy", columns="dataset", values="Δsp")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    vmax = max(abs(np.nanmin(pivot.values)), abs(np.nanmax(pivot.values)), 1e-6)
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=FONT_TICK)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=FONT_TICK)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=FONT_TICK,
                    color="white" if abs(val) / vmax > 0.6 else "black", fontweight="bold")
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=FONT_TICK)
    cbar.set_label("ΔSP  (↑ = fairer)", fontsize=FONT_LABEL)
    ax.set_title("Cross-Dataset × Cross-Strategy ΔSP", fontsize=FONT_TITLE, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_ranking(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(summary))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in summary["avg_fairness_gain"]]
    ax.bar(x, summary["avg_fairness_gain"], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["strategy"], fontsize=FONT_TICK)
    ax.set_ylabel("Avg. Δ(SP, EOD, PP, TI) across datasets", fontsize=FONT_LABEL - 1)
    ax.set_title("Average Mitigation Effectiveness Across Datasets", fontsize=FONT_TITLE, fontweight="bold")
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", lw=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="VAYNE cross-dataset / cross-strategy synthesis")
    ap.add_argument("--results-dir", required=True, help="Directory with per-dataset VAYNE results")
    args = ap.parse_args()
    results_dir = Path(args.results_dir)

    mitigation_df = load_mitigation(results_dir)
    proxy_df      = load_proxy_risk(results_dir)

    baseline_table   = direct_and_baseline_table(mitigation_df)
    proxy_table      = top_proxy_table(proxy_df)
    strategy_summary = cross_strategy_summary(mitigation_df)

    baseline_table.to_csv(results_dir / "synthesis_baseline_fairness.csv", index=False)
    proxy_table.to_csv(results_dir / "synthesis_top_proxy_features.csv", index=False)
    mitigation_df.to_csv(results_dir / "synthesis_cross_dataset_mitigation.csv", index=False)
    strategy_summary.to_csv(results_dir / "synthesis_cross_strategy_summary.csv", index=False)

    plot_baseline_fairness(baseline_table, results_dir / "synthesis_baseline_fairness.png")
    plot_strategy_heatmap(mitigation_df, results_dir / "synthesis_strategy_heatmap.png")
    plot_strategy_ranking(strategy_summary, results_dir / "synthesis_strategy_ranking.png")

    print("\n== Direct & Baseline Fairness ==")
    print(baseline_table.to_string(index=False))
    print("\n== Top Proxy-Risk Features ==")
    print(proxy_table.to_string(index=False))
    print("\n== Cross-Strategy Summary (mean over datasets) ==")
    print(strategy_summary.to_string(index=False))
    print(f"\nSaved synthesis artifacts -> {results_dir}")


if __name__ == "__main__":
    main()
