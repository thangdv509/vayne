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
from src.mitigation import VAYNE_STRATEGY_NAMES

logger = logging.getLogger(__name__)

_ALL_METRICS   = ["du", "sp", "eod", "pp", "abroca", "ti"]
_GROUP_METRICS = ["sp", "eod", "pp", "ti"]
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
    row = {"strategy": "baseline", "type": "—"}
    for k in _ALL_METRICS:
        row[k] = round(getattr(base, k), 4)
    for k in _ALL_METRICS:
        row[f"Δ{k}"] = 0.0
    row.update(_compute_utility(baseline_scores, labels))
    return row


def _mitigation_row(m, labels) -> dict:
    row = {
        "strategy": m.strategy,
        "type": "VAYNE" if m.strategy in VAYNE_STRATEGY_NAMES else "EXT",
    }
    for k in _ALL_METRICS:
        row[k] = round(getattr(m.report, k), 4)
    for k in _ALL_METRICS:
        row[f"Δ{k}"] = m.improvement.get(f"Δ{k}", 0.0)
    row.update(_compute_utility(m.scores, labels))
    return row


# ── Main save function ────────────────────────────────────────────────────────

def save_results(result, verbose: bool = True, results_dir: Path = RESULTS_DIR) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
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
    json_path = results_dir / f"{tag}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Saved JSON → %s", json_path)

    # ── Proxy risk CSV ────────────────────────────────────────────────────────
    if result.risk_summary:
        pd.DataFrame(result.risk_summary).to_csv(
            results_dir / f"{tag}_proxy_risk.csv", index=False)

    # ── Interaction summary CSV (non-zero only) ───────────────────────────────
    if result.interaction_summary:
        nonzero = [r for r in result.interaction_summary if r["interaction"] > 0]
        if nonzero:
            pd.DataFrame(nonzero).to_csv(
                results_dir / f"{tag}_interactions.csv", index=False)
            logger.info("Saved interactions CSV → %d non-zero pairs", len(nonzero))

    # ── Mitigation CSV with utility metrics ──────────────────────────────────
    if result.mitigation_results:
        labels = result.labels
        rows   = [_baseline_row(result.baseline_report, result.baseline_scores, labels)]
        for m in result.mitigation_results:
            rows.append(_mitigation_row(m, labels))
        pd.DataFrame(rows).to_csv(
            results_dir / f"{tag}_mitigation.csv", index=False)

    # ── Figures ───────────────────────────────────────────────────────────────
    # Figures are named without the model tag: fig_{dataset}_{fig_name}.png
    fig_tag = f"fig_{result.dataset}"
    _plot_proxy_risk_bars(result,        results_dir / f"{fig_tag}_proxy_risk.png")
    _plot_pc_pu_scatter(result,          results_dir / f"{fig_tag}_pc_pu_scatter.png")
    _plot_mitigation_heatmap(result,     results_dir / f"{fig_tag}_mitigation_heatmap.png")

    if result.proxy_use:
        plot_proxy_use_bar(
            {
                "dataset":        result.dataset,
                "proxy_use":      result.proxy_use,
                "proxy_capacity": result.proxy_capacity,
                "tau_capacity":   result.tau_capacity,
                "tau_use":        result.tau_use,
            },
            results_dir / f"{fig_tag}_proxy_bar_plot.png",
        )

    if result.proxy_graph_obj is not None:
        sensitive_col = next(
            (n for n, d in result.proxy_graph_obj.nodes(data=True)
             if d.get("node_type") == "sensitive"),
            "",
        )
        plot_proxy_graph(
            result.proxy_graph_obj,
            sensitive_attr=sensitive_col,
            save_path=results_dir / f"{fig_tag}_proxy_graph.png",
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

        def _trunc(name, limit=14):
            return name if len(name) <= limit else name[:limit - 1] + "..."

        features = [_trunc(r["feature"]) for r in rows]
        pc_vals  = [r["proxy_capacity"] for r in rows]
        pu_vals  = [r["proxy_use"]      for r in rows]
        prs_vals = [r["proxy_risk"]     for r in rows]

        x     = np.arange(len(features))
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(10, len(features) * 1.3), 5))
        ax.bar(x - width, pc_vals,  width, label="PC (Proxy Capacity)",  color="#aec6e8")
        ax.bar(x,         pu_vals,  width, label="PU (Proxy Use)",        color="#ffd9a0")
        ax.bar(x + width, prs_vals, width, label="PRS (Proxy Risk Score)", color="#f4a8a8")
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=30, ha="right", fontsize=12)
        ax.set_ylabel("Score", fontsize=14)
        ax.tick_params(axis="y", labelsize=11)
        ax.legend(fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Proxy risk bars → %s", save_path)
    except Exception as exc:
        logger.warning("Proxy risk bars failed: %s", exc)


def _plot_pc_pu_scatter(result, save_path: Path) -> None:
    """
    Scatter plot: PC (x-axis, symlog) vs PU (y-axis).
    Color = PRS. Top-50%-by-PRS features labeled (adjustText-avoided overlap).
    Risk zone shaded.
    """
    try:
        import matplotlib.pyplot as plt

        rows = result.risk_summary
        if not rows:
            return

        # Keep only top 50% by PRS — reduces clutter, focuses on risk-relevant features
        rows_sorted = sorted(rows, key=lambda r: r["proxy_risk"], reverse=True)
        rows = rows_sorted[:max(4, len(rows_sorted) // 2)]

        pc   = np.array([r["proxy_capacity"] for r in rows])
        pu   = np.array([r["proxy_use"]      for r in rows])
        prs  = np.array([r["proxy_risk"]     for r in rows])
        feat = [r["feature"] for r in rows]

        from src.config import GRAPH_THRESHOLD_PERCENTILE as _PCT
        # Compute thresholds from the full (unfiltered) feature set
        pc_all = np.array([r["proxy_capacity"] for r in rows_sorted])
        pu_all = np.array([r["proxy_use"]      for r in rows_sorted])
        tau1 = getattr(result, "tau_capacity", None) or float(np.percentile(pc_all, _PCT))
        tau2 = getattr(result, "tau_use",      None) or float(np.percentile(pu_all, _PCT))

        fig, ax = plt.subplots(figsize=(11, 7))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Axis limits — tight around filtered data, no extra label padding
        y_pad = (pu.max() - pu.min() + 1e-6) * 0.30
        x_plot_max = pc.max() * 1.45 + 1e-6
        y_plot_max = pu.max() + y_pad
        y_plot_min = max(0, pu.min() - y_pad)

        ax.fill_betweenx([tau2, y_plot_max], tau1, x_plot_max,
                         color="#FDECEA", alpha=0.7, zorder=0)  # risk zone: red tint
        ax.fill_betweenx([y_plot_min, tau2], 0, tau1,
                         color="#EAF4FB", alpha=0.5, zorder=0)  # safe zone: blue tint

        # Scatter — color = PRS
        vmin, vmax = prs.min(), max(prs.max(), 1e-9)
        sc = ax.scatter(pc, pu, s=200, c=prs, cmap="YlOrRd", vmin=vmin, vmax=vmax,
                        alpha=0.9, edgecolors="#555555", linewidths=0.9, zorder=3)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Proxy Risk Score (PRS = PC × PU)", fontsize=13)
        cbar.ax.tick_params(labelsize=12)

        # Threshold lines
        ax.axvline(tau1, color="#2980B9", linestyle="--", lw=1.8,
                   label=f"τ₁ PC = {tau1:.3f} ({_PCT}th pct)")
        ax.axhline(tau2, color="#E67E22", linestyle="--", lw=1.8,
                   label=f"τ₂ PU = {tau2:.3f} ({_PCT}th pct)")

        # Risk zone label
        ax.text(0.42, 0.97, "Proxy Risk Zone",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=14, color="#C0392B", fontweight="bold", alpha=0.85)

        # Feature labels — truncate, then de-overlap with adjustText
        def _trunc(name, limit=13):
            return name if len(name) <= limit else name[:limit - 1] + "..."

        texts = []
        for i in range(len(feat)):
            t = ax.text(pc[i], pu[i], _trunc(feat[i]),
                        fontsize=12, color="#222222", clip_on=True,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  alpha=0.9, ec="#CCCCCC", lw=0.6))
            texts.append(t)

        # Symlog x-axis must be set BEFORE adjust_text so coordinates are correct
        linthresh = max(tau1 * 0.5, 1e-4)
        ax.set_xscale("symlog", linthresh=linthresh, linscale=0.4)
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlim(0, x_plot_max)
        ax.set_ylim(y_plot_min, y_plot_max)

        try:
            from adjustText import adjust_text
            adjust_text(
                texts, x=pc, y=pu, ax=ax,
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.8),
                expand=(1.8, 2.0),
                force_text=(0.6, 0.8),
                force_points=(0.4, 0.5),
            )
        except ImportError:
            logger.warning("adjustText not installed — feature labels may overlap.")

        ax.set_xlabel("Proxy Capacity (PC)  [symlog scale]", fontsize=15)
        ax.set_ylabel("Proxy Use (PU)", fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        ax.legend(fontsize=12, framealpha=0.95, loc="lower right")
        ax.grid(alpha=0.2, which="both")

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
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
        disp_keys   = _GROUP_METRICS          # sp, eod, pp, ti
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

        fig, ax = plt.subplots(figsize=(9, max(3.5, len(strategies) * 1.0 + 1.2)))
        im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=13)
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies, fontsize=13)

        for i in range(len(strategies)):
            for j in range(len(col_labels)):
                val       = data[i, j]
                intensity = abs(val) / vmax
                txt_color = "white" if intensity > 0.6 else "black"
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                        fontsize=12, color=txt_color, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, label="Δ metric (↑ = improvement)")
        cbar.ax.tick_params(labelsize=11)
        cbar.set_label("Δ metric (↑ = improvement)", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Mitigation heatmap → %s", save_path)
    except Exception as exc:
        logger.warning("Mitigation heatmap failed: %s", exc)


def _trunc_feat(name, limit=20):
    return name if len(name) <= limit else name[:limit - 1] + "..."


def _draw_proxy_use_bars(ax, res: dict) -> float:
    """Draw one dataset's Proxy Use horizontal bar chart onto `ax`. Returns tau2."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    pu   = res["proxy_use"]
    pc2  = res["proxy_capacity"]
    tau1 = res["tau_capacity"]
    tau2 = res["tau_use"]

    items   = sorted(pu.items(), key=lambda x: -x[1])
    feats   = [_trunc_feat(x[0]) for x in items]
    pu_vals = np.array([x[1] for x in items])
    pc_vals = np.array([pc2.get(x[0], 0) for x in items])

    colors = [
        "#E74C3C" if cv > tau1 and pv > tau2 else
        "#3498DB" if pv > tau2 else
        "#BDC3C7"
        for pv, cv in zip(pu_vals, pc_vals)
    ]

    y = np.arange(len(feats))
    ax.barh(y, pu_vals, color=colors, edgecolor="white",
            linewidth=0.5, height=0.7)
    ax.axvline(tau2, color="#E67E22", linestyle="--", lw=1.8,
               label=f"τ₂ = {tau2:.3f}")
    ax.set_yticks(y)
    ax.set_yticklabels(feats, fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel("Proxy Use (PU)", fontsize=15)
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlim(0, pu_vals.max() * 1.18)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        mpatches.Patch(color="#E74C3C", label="Proxy Risk Zone"),
        mpatches.Patch(color="#3498DB", label="High PU, low PC"),
        mpatches.Patch(color="#BDC3C7", label="Low PU"),
        plt.Line2D([0], [0], color="#E67E22", lw=1.8,
                   linestyle="--", label=f"τ₂ = {tau2:.3f}"),
    ]
    ax.legend(handles=legend_handles, fontsize=12,
              loc="lower right", framealpha=0.95)
    return tau2


def plot_proxy_use_bar(result: dict, save_path: Path) -> None:
    """
    Horizontal bar chart of Proxy Use for a single dataset.
    result: dict with keys 'dataset', 'proxy_use', 'proxy_capacity',
            'tau_capacity', 'tau_use'
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7.5))
        fig.patch.set_facecolor("white")
        _draw_proxy_use_bars(ax, result)
        plt.tight_layout(pad=2.0)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("PU bar plot → %s", save_path)
    except Exception as exc:
        logger.warning("PU bar plot failed: %s", exc)


def plot_proxy_use_barplot(
    results: list,
    save_path: Path,
) -> None:
    """
    Horizontal bar chart of Proxy Use for multiple datasets side by side.
    results: list of dicts with keys 'dataset', 'proxy_use', 'proxy_capacity',
             'tau_capacity', 'tau_use'
    """
    try:
        import matplotlib.pyplot as plt

        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(8 * n, 7.5))
        if n == 1:
            axes = [axes]
        fig.patch.set_facecolor("white")

        for ax, res in zip(axes, results):
            _draw_proxy_use_bars(ax, res)

        plt.tight_layout(pad=2.0)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("PU barplot → %s", save_path)
    except Exception as exc:
        logger.warning("PU barplot failed: %s", exc)


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
    print("  [VAYNE] = proxy-aware mitigation   [EXT] = external baseline")
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
        prefix = "[VAYNE]" if m.strategy in VAYNE_STRATEGY_NAMES else "[EXT]  "
        label  = f"{prefix} {m.strategy}"
        table_rows.append(_fmt(label, m.report, m.improvement,
                               m.scores, result.labels))

    print(tabulate(table_rows, headers=headers, tablefmt="rounded_outline"))
    print()
