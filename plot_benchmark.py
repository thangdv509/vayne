"""
Plot few-shot (k=128) classification utility across temperature for the
three benchmark datasets, comparing the Template and Markdown serialization
formats.

Usage:
  python3 plot_benchmark.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BENCH_DIR = Path(__file__).resolve().parent / "benchmark"
MODEL_SLUG = "claude-3-haiku"

DATASETS = [
    ("credit_approval", "Credit Approval"),
    ("credit_card",     "Credit Card"),
    ("german_credit",   "German Credit"),
]

METHOD_STYLE = {
    "template_fs": {"label": "Template", "color": "#1f77b4", "marker": "o"},
    "markdown_fs": {"label": "Markdown", "color": "#d62728", "marker": "s"},
}

FONT_TITLE  = 20
FONT_LABEL  = 18
FONT_TICK   = 15
FONT_LEGEND = 16


def plot_metric(metric: str, ylabel: str, out_name: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=False)

    for ax, (slug, title) in zip(axes, DATASETS):
        csv_path = BENCH_DIR / f"{MODEL_SLUG}_{slug}_128.csv"
        df = pd.read_csv(csv_path)
        df = df[df["K"] == 128]

        for method, style in METHOD_STYLE.items():
            sub = df[df["Method"] == method].sort_values("Temp")
            ax.plot(
                sub["Temp"], sub[metric],
                marker=style["marker"], markersize=10, linewidth=2.5,
                color=style["color"], label=style["label"],
            )

        ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
        ax.set_xlabel("Temperature", fontsize=FONT_LABEL)
        ax.tick_params(axis="both", labelsize=FONT_TICK)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel(ylabel, fontsize=FONT_LABEL)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2,
        bbox_to_anchor=(0.5, 1.06), fontsize=FONT_LEGEND, frameon=False,
    )
    fig.suptitle(
        f"claude-3-haiku, k=128 few-shot — {ylabel} vs. Temperature",
        fontsize=FONT_TITLE + 2, y=1.14,
    )
    fig.tight_layout()

    out_path = BENCH_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    plot_metric("Acc.", "Accuracy", "claude-3-haiku_k128_acc_vs_temp.png")
    plot_metric("BA", "Balanced Accuracy", "claude-3-haiku_k128_ba_vs_temp.png")
    plot_metric("TI", "Theil Index", "claude-3-haiku_k128_ti_vs_temp.png")
