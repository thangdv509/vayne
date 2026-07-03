"""
VAYNE — Main experiment runner

Usage examples:
  python main.py
  python main.py --datasets german_credit --models deepseek/deepseek-chat --n 30
  python main.py --datasets adult --n 50 --verbose
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — prevents tkinter crash in ThreadPoolExecutor

import argparse
import logging
import sys
from pathlib import Path

from src.config import DATASET_CONFIGS, DEFAULT_MODEL, MODELS, RESULTS_DIR
from src.pipeline import run_experiment
from src.reporting import save_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("vayne.main")


def parse_args():
    p = argparse.ArgumentParser(description="VAYNE: Proxy Fairness in LLM-based Decisions")
    p.add_argument("--datasets", nargs="+",
                   default=list(DATASET_CONFIGS.keys()),
                   choices=list(DATASET_CONFIGS.keys()),
                   help="Datasets to run (default: all)")
    p.add_argument("--models", nargs="+",
                   default=[DEFAULT_MODEL],
                   help="Model identifiers via OpenRouter (default: deepseek/deepseek-chat)")
    p.add_argument("--n", type=int, default=30,
                   help="Number of samples per dataset (default: 30)")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="LLM sampling temperature for all inference calls (default: 0.1)")
    p.add_argument("--k-shot", type=int, default=0,
                   help="Few-shot demonstrations per query, sampled from the full "
                        "dataset (default: 0 = zero-shot)")
    p.add_argument("--results-dir", type=str, default=str(RESULTS_DIR),
                   help="Output directory for JSON/CSV/PNG results (default: ./results)")
    p.add_argument("--verbose", action="store_true", default=True,
                   help="Print results to console (default: True)")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    logger.info("VAYNE experiment started")
    logger.info("Datasets    : %s", args.datasets)
    logger.info("Models      : %s", args.models)
    logger.info("Samples     : %d", args.n)
    logger.info("Temperature : %s", args.temperature)
    logger.info("K-shot      : %d", args.k_shot)
    logger.info("Results dir : %s", results_dir)

    for dataset_name in args.datasets:
        config = DATASET_CONFIGS[dataset_name]
        for model in args.models:
            logger.info("─" * 55)
            logger.info("Running: dataset=%s  model=%s", dataset_name, model)
            try:
                result = run_experiment(
                    dataset_name=dataset_name,
                    model=model,
                    config=config,
                    n_samples=args.n,
                    temperature=args.temperature,
                    k_shot=args.k_shot,
                )
                save_results(result, verbose=args.verbose, results_dir=results_dir)
            except Exception as exc:
                logger.error("Experiment failed (%s / %s): %s", dataset_name, model, exc,
                             exc_info=True)

    logger.info("All experiments complete. Results saved to %s", results_dir)


if __name__ == "__main__":
    main()
