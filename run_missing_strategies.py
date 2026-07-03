"""
Chạy 2 mitigation strategy còn thiếu (fair_prompt, few_shot_fair) cho các
dataset đã có sẵn kết quả 4 strategy VAYNE (masking/abstraction/debiasing/
filtering), dùng đúng n_samples/model từ JSON đã lưu + temperature/k-shot
được truyền vào.

Không chạy lại proxy_capacity/proxy_use/proxy_risk/proxy_graph (đắt tiền) —
chỉ cần baseline+masked-A (cho DU) và 2 strategy mới. Cập nhật lại JSON, CSV
mitigation, và vẽ lại mitigation_heatmap (6 strategy).

Usage:
  python3 run_missing_strategies.py result_0107/credit_approval.json \\
      --temperature 0.4 --k-shot 128
  python3 run_missing_strategies.py result_0107/*.json --temperature 0.4 --k-shot 128
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.config import DATASET_CONFIGS, PRS_FILTER_THRESHOLD, DECISION_THRESHOLD
from src.data_loader import load_dataset, get_label, is_protected
from src.pipeline import (
    _run_baseline_inference, _run_mitigation_inference,
    _build_fs_examples_per_row, _select_few_shot_examples,
)
from src.fairness_metrics import compute_fairness, FairnessReport
from src.mitigation import BASELINE_STRATEGY_NAMES, VAYNE_STRATEGY_NAMES
from src.reporting import _compute_utility, _plot_mitigation_heatmap

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("run_missing_strategies")


def process_one(json_path: Path, temperature: float, k_shot: int) -> None:
    with open(json_path) as f:
        d = json.load(f)

    dataset = d["dataset"]
    model   = d["model"]
    n       = d["n_samples"]
    config  = DATASET_CONFIGS[dataset]

    existing = {m["strategy"] for m in d["mitigation_results"]}
    missing  = [s for s in BASELINE_STRATEGY_NAMES if s not in existing]
    if not missing:
        logger.info("[%s] nothing missing, skipping.", dataset)
        return
    logger.info("[%s] missing strategies: %s", dataset, missing)

    sensitive_col = config["sensitive_attrs"][0]
    df = load_dataset(config, n_samples=n)  # random_state=42 default, matches original
    labels    = [get_label(row, config) for _, row in df.iterrows()]
    prot_arr  = [is_protected(row, config) for _, row in df.iterrows()]

    fs_examples_per_row = None
    if k_shot > 0:
        logger.info("[%s] sampling %d-shot demonstrations...", dataset, k_shot)
        fs_examples_per_row = _build_fs_examples_per_row(df, config, k_shot)

    logger.info("[%s] running baseline + masked-A inference (%d calls)...", dataset, 2 * n)
    baseline_scores, masked_A_scores = _run_baseline_inference(
        df, config, model, sensitive_col,
        temperature=temperature, fs_examples_per_row=fs_examples_per_row,
    )

    baseline_report = FairnessReport(**d["baseline_fairness"])

    few_shot_examples = _select_few_shot_examples(df, labels, prot_arr, n_per_cell=1)
    mitigation_config = dict(config, _few_shot_examples=few_shot_examples)

    proxy_risk = d.get("proxy_risk", {})
    top_k      = list(proxy_risk.keys())  # unused by fair_prompt / few_shot_fair

    new_results = []
    for strategy in missing:
        logger.info("[%s] running strategy: %s (%d calls)", dataset, strategy, n)
        mit_scores = _run_mitigation_inference(
            df, mitigation_config, model, strategy,
            proxy_risk, top_k, PRS_FILTER_THRESHOLD,
            masked_A_scores,
            temperature=temperature, fs_examples_per_row=fs_examples_per_row,
        )
        mit_report = compute_fairness(
            scores=mit_scores, scores_masked_A=masked_A_scores,
            labels=labels, is_protected_arr=prot_arr,
            threshold=DECISION_THRESHOLD,
        )
        improvement = mit_report.improvement_over(baseline_report)
        new_results.append({
            "strategy": strategy,
            "fairness": mit_report.to_dict(),
            "improvement": improvement,
            "_scores": mit_scores,  # kept only for the CSV utility calc below
        })

    # ── Update JSON ──────────────────────────────────────────────────────────
    for r in new_results:
        d["mitigation_results"].append({
            "strategy": r["strategy"],
            "fairness": r["fairness"],
            "improvement": r["improvement"],
        })
    with open(json_path, "w") as f:
        json.dump(d, f, indent=2, default=str)
    logger.info("[%s] updated JSON → %s", dataset, json_path)

    # ── Update mitigation CSV ────────────────────────────────────────────────
    csv_path = json_path.parent / f"{dataset}_mitigation.csv"
    if csv_path.exists():
        mit_df = pd.read_csv(csv_path)
        for r in new_results:
            u = _compute_utility(r["_scores"], labels)
            row = {"strategy": r["strategy"],
                   "type": "VAYNE" if r["strategy"] in VAYNE_STRATEGY_NAMES else "EXT"}
            row.update(r["fairness"])
            row.update(r["improvement"])
            row.update(u)
            mit_df = pd.concat([mit_df, pd.DataFrame([row])], ignore_index=True)
        mit_df.to_csv(csv_path, index=False)
        logger.info("[%s] updated mitigation CSV → %s", dataset, csv_path)

    # ── Redraw mitigation heatmap with all strategies ───────────────────────
    result = SimpleNamespace(
        dataset=dataset,
        mitigation_results=[
            SimpleNamespace(strategy=m["strategy"], improvement=m["improvement"])
            for m in d["mitigation_results"]
        ],
    )
    _plot_mitigation_heatmap(result, json_path.parent / f"fig_{dataset}_mitigation_heatmap.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", help="Result JSON files to fill in")
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--k-shot", type=int, default=0)
    args = p.parse_args()

    for f in args.files:
        jp = Path(f)
        logger.info("=" * 60)
        logger.info("Processing %s", jp)
        try:
            process_one(jp, args.temperature, args.k_shot)
        except Exception:
            logger.exception("Failed on %s", jp)


if __name__ == "__main__":
    main()
