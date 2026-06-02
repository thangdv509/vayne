"""
VAYNE Reporting

Saves and prints results from a VAYNEResult. Output per experiment:
  - JSON:  full proxy analysis + fairness metrics
  - CSV:   proxy risk scores per feature
  - CSV:   pairwise proxy interactions
  - CSV:   mitigation comparison (6 fairness metrics + utility)
  - PNG:   proxy risk bar chart  (PC, PU, PRS per feature)
  - PNG:   proxy capacity-use scatter  (PC vs PU, bubble = PRS)
  - PNG:   proxy path graph figure
  - PNG:   mitigation improvement heatmap (strategy × metric)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.config import RESULTS_DIR, DECISION_THRESHOLD
from src.proxy_graph import plot_proxy_graph

logger = logging.getLogger(__name__)

_ALL_METRICS   = ["du", "sp", "eod", "pp", "abroca", "ti"]
_GROUP_METRICS = ["sp", "eod", "pp", "abroca", "ti"]
_DELTA_KEYS    = [f"Δ{m}" for m in _ALL_METRICS]

_METRIC_DISPLAY = {
    "du": "DU", "sp": "SP", "eod": "EOD",
    "pp": "PP", "abroca": "ABROCA", "ti": "TI",
}


# ── Utility helpers ───────────────────────────────────────────────────────────

def _compute_utility(scores: list[float], labels: list[int],
                     threshold: float = DECISION_THRESHOLD) -> dict:
    """Return accuracy and macro-F1 for a set of scores vs. ground-truth labels."""
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    n  = len(preds)
    acc  = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": round(acc, 4), "f1": round(f1, 4)}


# ── CSV row builders ─────────────────────────────────────────────────────────

def _baseline_row(base, baseline_scores, labels) -> dict:
    row = {"strategy": "baseline"}
    for k in _ALL_METRICS:
        row[k] = round(getattr(base, k), 4)
    for k in _ALL_METRICS:
        row[f"Δ{k}"] = 0.0
    row.update(_compute_utility(baseline_scores, labels))
    return row


def _mitigation_row(m, labels) -> dict:
    row = {"strategy": m.strategy}
    for k in _ALL_METRICS:
        row[k] = round(getattr(m.report, k), 4)
    for k in _ALL_METRICS:
        row[f"Δ{k}"] = m.improvement.get(f"Δ{k}", 0.0)
    row.update(_compute_utility(m.scores, labels))
    return row


# ── Main save function ────────────────────────────────────────────────────────

def save_results(result, verbose: bool = True) -> None:
    tag = f"{result.dataset}__{result.model.replace('/', '_')}"

    # ── JSON dump ────────────────────────────────────────────────────────────
    payload = {
        "dataset":   result.dataset,
        "model":     result.model,
        "n_samples": result.n_samples,
        "tau_capacity":    result.tau_capacity,
        "tau_use":         result.tau_use,
        "tau_interaction": result.tau_interaction,
        "proxy_capacity":      result.proxy_capacity,
        "proxy_use":           result.proxy_use,
        "proxy_risk":          result.proxy_risk,
        "proxy_interaction":   result.proxy_interaction,
        "proxy_graph":         result.proxy_graph_dict,
        "proxy_paths":         result.proxy_paths,
        "proxy_ascii":         result.proxy_ascii,
        "risk_summary":        result.risk_summary,
        "interaction_summary": result.interaction_summary,
        "baseline_fairness":   result.baseline_report.to_dict() if result.baseline_report else {},
        "mitigation_results":  [
            {"strategy": m.strategy, "fairness": m.report.to_dict(), "improvement": m.improvement}
            for m in result.mitigation_results
        ],
    }
    json_path = RESULTS_DIR / f"{tag}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Saved JSON → %s", json_path)

    # ── Proxy risk CSV ────────────────────────────────────────────────────────
    if result.risk_summary:
        pd.DataFrame(result.risk_summary).to_csv(
            RESULTS_DIR / f"{tag}_proxy_risk.csv", index=False)

    # ── Interaction summary CSV (non-zero only) ───────────────────────────────
    if result.interaction_summary:
        nonzero = [r for r in result.interaction_summary if r["interaction"] > 0]
        if nonzero:
            pd.DataFrame(nonzero).to_csv(
                RESULTS_DIR / f"{tag}_interactions.csv", index=False)
            logger.info("Saved interactions CSV → %d non-zero pairs", len(nonzero))

    # ── Mitigation CSV with utility metrics ──────────────────────────────────
    if result.mitigation_results:
        labels = result.labels
        rows   = [_baseline_row(result.baseline_report, result.baseline_scores, labels)]
        for m in result.mitigation_results:
            rows.append(_mitigation_row(m, labels))
        pd.DataFrame(rows).to_csv(
            RESULTS_DIR / f"{tag}_mitigation.csv", index=False)

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_proxy_risk_bars(result,        RESULTS_DIR / f"{tag}_proxy_risk.png")
    _plot_pc_pu_scatter(result,          RESULTS_DIR / f"{tag}_pc_pu_scatter.png")
    _plot_mitigation_heatmap(result,     RESULTS_DIR / f"{tag}_mitigation_heatmap.png")

    if result.proxy_graph_obj is not None:
        sensitive_col = next(
            (n for n, d in result.proxy_graph_obj.nodes(data=True)
             if d.get("node_type") == "sensitive"),
            "",
        )
        plot_proxy_graph(
            result.proxy_graph_obj,
            sensitive_attr=sensitive_col,
            save_path=RESULTS_DIR / f"{tag}_proxy_graph.png",
            title=f"Proxy Path Graph — {result.dataset} / {result.model.split('/')[-1]}",
        )

    if verbose:
        _print_results(result)


# ── Figure functions ─────────────────────────────────────────────────────────

def _plot_proxy_risk_bars(result, save_path: Path) -> None:
    """Grouped bar chart: PC, PU, PRS for top-10 proxy features."""
    try:
        import matplotlib.pyplot as plt

        rows = result.risk_summary[:10]
        if not rows:
            return

        features = [r["feature"] for r in rows]
        pc_vals  = [r["proxy_capacity"] for r in rows]
        pu_vals  = [r["proxy_use"]      for r in rows]
        prs_vals = [r["proxy_risk"]     for r in rows]

        x     = np.arange(len(features))
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(8, len(features) * 1.1), 4))
        ax.bar(x - width, pc_vals,  width, label="PC (Proxy Capacity)",  color="#aec6e8")
        ax.bar(x,         pu_vals,  width, label="PU (Proxy Use)",        color="#ffd9a0")
        ax.bar(x + width, prs_vals, width, label="PRS (Proxy Risk Score)", color="#f4a8a8")
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Score")
        ax.set_title(
            f"Proxy Risk Scores — {result.dataset} / {result.model.split('/')[-1]}",
            fontsize=10, fontweight="bold",
        )
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Proxy risk bars → %s", save_path)
    except Exception as exc:
        logger.warning("Proxy risk bars failed: %s", exc)


def _plot_pc_pu_scatter(result, save_path: Path) -> None:
    """
    Scatter plot: PC (x-axis, symlog) vs PU (y-axis).
    Color = PRS. All features labeled. Risk zone shaded.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches

        rows = result.risk_summary
        if not rows:
            return

        pc   = np.array([r["proxy_capacity"] for r in rows])
        pu   = np.array([r["proxy_use"]      for r in rows])
        prs  = np.array([r["proxy_risk"]     for r in rows])
        feat = [r["feature"] for r in rows]

        from src.config import GRAPH_THRESHOLD_PERCENTILE as _PCT
        tau1 = getattr(result, "tau_capacity", None) or float(np.percentile(pc, _PCT))
        tau2 = getattr(result, "tau_use",      None) or float(np.percentile(pu, _PCT))

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Shade quadrants
        x_plot_max = pc.max() * 1.5 + 1e-6
        y_plot_max = pu.max() * 1.12 + 1e-6
        y_plot_min = max(0, pu.min() - (pu.max() - pu.min()) * 0.15)

        ax.fill_betweenx([tau2, y_plot_max], tau1, x_plot_max,
                         color="#FDECEA", alpha=0.7, zorder=0)  # risk zone: red tint
        ax.fill_betweenx([y_plot_min, tau2], 0, tau1,
                         color="#EAF4FB", alpha=0.5, zorder=0)  # safe zone: blue tint

        # Scatter — fixed size, color = PRS
        vmin, vmax = prs.min(), max(prs.max(), 1e-9)
        sc = ax.scatter(pc, pu, s=120, c=prs, cmap="YlOrRd", vmin=vmin, vmax=vmax,
                        alpha=0.9, edgecolors="#555555", linewidths=0.7, zorder=3)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Proxy Risk Score (PRS = PC × PU)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        # Threshold lines
        ax.axvline(tau1, color="#2980B9", linestyle="--", lw=1.4,
                   label=f"τ₁ PC = {tau1:.3f} ({_PCT}th pct)")
        ax.axhline(tau2, color="#E67E22", linestyle="--", lw=1.4,
                   label=f"τ₂ PU = {tau2:.3f} ({_PCT}th pct)")

        # Risk zone label
        ax.text(x_plot_max * 0.98, y_plot_max * 0.98,
                "⚠ Proxy Risk Zone", ha="right", va="top",
                fontsize=9, color="#C0392B", fontweight="bold", alpha=0.8)

        # Label ALL features — alternate above/below to reduce overlap
        prs_rank = np.argsort(prs)[::-1]
        for rank, i in enumerate(prs_rank):
            dy_pts = 10 if rank % 2 == 0 else -18
            va = "bottom" if dy_pts > 0 else "top"
            label = feat[i].replace("-", "‑")  # non-breaking hyphen
            ax.annotate(
                label, (pc[i], pu[i]),
                textcoords="offset points", xytext=(6, dy_pts),
                fontsize=7, color="#222222", va=va,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          alpha=0.8, ec="#CCCCCC", lw=0.5),
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.6),
            )

        # Symlog x-axis so clustered-near-zero points spread out
        linthresh = max(tau1 * 0.5, 1e-4)
        ax.set_xscale("symlog", linthresh=linthresh, linscale=0.4)
        ax.xaxis.set_minor_locator(plt.NullLocator())

        ax.set_xlim(0, x_plot_max)
        ax.set_ylim(y_plot_min, y_plot_max)

        ax.set_xlabel("Proxy Capacity  PC = I(X ; A)   [symlog scale]", fontsize=9)
        ax.set_ylabel("Proxy Use  PU = E[ |s(x) − s(x₋ⱼ)| ]", fontsize=9)
        ax.set_title(f"PC vs PU Feature Space — {result.dataset}",
                     fontsize=11, fontweight="bold", pad=10)
        ax.legend(fontsize=8, framealpha=0.95, loc="lower right")
        ax.grid(alpha=0.2, which="both")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("PC-PU scatter → %s", save_path)
    except Exception as exc:
        logger.warning("PC-PU scatter failed: %s", exc)


def _plot_mitigation_heatmap(result, save_path: Path) -> None:
    """
    Heatmap: rows = mitigation strategies, columns = Δ fairness metrics.
    Green = improvement, red = worsening.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        if not result.mitigation_results:
            return

        strategies  = [m.strategy for m in result.mitigation_results]
        disp_keys   = _GROUP_METRICS          # sp, eod, pp, abroca, ti
        delta_keys  = [f"Δ{k}" for k in disp_keys]
        col_labels  = [f"Δ{_METRIC_DISPLAY[k]}" for k in disp_keys]

        data = np.array([
            [m.improvement.get(dk, 0.0) for dk in delta_keys]
            for m in result.mitigation_results
        ])  # shape: (n_strategies, n_metrics)

        vmax = max(np.abs(data).max(), 0.01)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "rg", ["#e74c3c", "#ffffff", "#27ae60"]
        )

        fig, ax = plt.subplots(figsize=(7, max(3, len(strategies) * 0.85 + 1)))
        im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies, fontsize=9)

        for i in range(len(strategies)):
            for j in range(len(col_labels)):
                val       = data[i, j]
                intensity = abs(val) / vmax
                txt_color = "white" if intensity > 0.6 else "black"
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                        fontsize=8, color=txt_color, fontweight="bold")

        plt.colorbar(im, ax=ax, label="Δ metric (↑ = improvement)")
        ax.set_title(
            f"Mitigation Improvement — {result.dataset} / {result.model.split('/')[-1]}",
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Mitigation heatmap → %s", save_path)
    except Exception as exc:
        logger.warning("Mitigation heatmap failed: %s", exc)


# ── Console output ───────────────────────────────────────────────────────────

def _print_results(result) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  VAYNE Results  |  {result.dataset}  |  {result.model}")
    print(sep)

    # Proxy risk table
    if result.risk_summary:
        print("\n── Proxy Risk Scores (PC × PU → PRS) ──")
        print(tabulate(result.risk_summary[:10], headers="keys",
                       tablefmt="rounded_outline", floatfmt=".4f"))

    # Top non-zero interactions
    top_int = [r for r in result.interaction_summary if r["interaction"] > 0][:5]
    if top_int:
        print("\n── Top Proxy Interactions (INT_ij > 0) ──")
        print(tabulate(top_int, headers="keys",
                       tablefmt="rounded_outline", floatfmt=".4f"))

    # Proxy path graph
    print("\n── Proxy Path Graph ──")
    print(result.proxy_ascii or "(no proxy paths above threshold)")

    # Full fairness + utility comparison
    print("\n── Fairness & Utility: Baseline vs Mitigation ──")
    headers = ["Strategy", "DU", "SP", "EOD", "PP", "ABROCA", "TI",
               "ΔSP", "ΔEOD", "ΔPP", "Acc", "F1"]

    def _fmt(label, report, imp, scores, labels):
        u = _compute_utility(scores, labels)
        return [
            label,
            f"{report.du:.4f}",
            f"{report.sp:.4f}",
            f"{report.eod:.4f}",
            f"{report.pp:.4f}",
            f"{report.abroca:.4f}",
            f"{report.ti:.4f}",
            f"{imp.get('Δsp',  0.0):+.4f}" if imp else "—",
            f"{imp.get('Δeod', 0.0):+.4f}" if imp else "—",
            f"{imp.get('Δpp',  0.0):+.4f}" if imp else "—",
            f"{u['accuracy']:.4f}",
            f"{u['f1']:.4f}",
        ]

    table_rows = [_fmt("baseline", result.baseline_report, None,
                       result.baseline_scores, result.labels)]
    for m in result.mitigation_results:
        table_rows.append(_fmt(m.strategy, m.report, m.improvement,
                               m.scores, result.labels))

    print(tabulate(table_rows, headers=headers, tablefmt="rounded_outline"))
    print()
