"""
Vẽ lại Proxy Path Graph từ file kết quả JSON có sẵn.
Không cần gọi LLM, chỉ dùng dữ liệu đã lưu.

Usage:
  python3 draw_graph.py                                        # tất cả file JSON trong results/
  python3 draw_graph.py results/german_credit__*.json         # file cụ thể
  python3 draw_graph.py --percentile 75                       # thay đổi percentile threshold
"""

import argparse
import ast
import json
import logging
from pathlib import Path

import numpy as np

from src.config import RESULTS_DIR
from src.proxy_graph import build_proxy_graph, plot_proxy_graph

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("draw_graph")


def draw_from_json(json_path: Path, percentile: int = 75) -> None:
    with open(json_path) as f:
        d = json.load(f)

    dataset   = d["dataset"]
    model     = d["model"]
    # Find sensitive attribute: node tagged "sensitive", or source of all capacity edges
    sensitive = next(
        (n["id"] for n in d["proxy_graph"]["nodes"] if n.get("node_type") == "sensitive"),
        None,
    )
    if sensitive is None:
        cap_sources = [e["source"] for e in d["proxy_graph"]["edges"]
                       if e.get("edge_type") == "capacity" and e["source"] != e["target"]]
        if cap_sources:
            from collections import Counter
            sensitive = Counter(cap_sources).most_common(1)[0][0]
    if sensitive is None:
        logger.warning("Cannot identify sensitive node in %s — skipping.", json_path.name)
        return

    # Proxy capacity (exclude sensitive attr itself)
    pc = {k: v for k, v in d["proxy_capacity"].items() if k != sensitive}
    pu = d["proxy_use"]
    interact = {ast.literal_eval(k): v for k, v in d.get("proxy_interaction", {}).items()}

    # Adaptive thresholds (percentile of actual distributions)
    pc_vals  = list(pc.values())
    pu_vals  = list(pu.values())
    int_nz   = [v for v in interact.values() if v > 0]

    tau1 = float(np.percentile(pc_vals,  percentile)) if len(pc_vals)  >= 4 else 0.03
    tau2 = float(np.percentile(pu_vals,  percentile)) if len(pu_vals)  >= 4 else 0.05
    tau3 = float(np.percentile(int_nz,   percentile)) if len(int_nz)   >= 4 else 0.02
    tau1, tau2, tau3 = max(tau1, 1e-6), max(tau2, 1e-6), max(tau3, 1e-6)

    logger.info("[%s] τ_PC=%.4f  τ_PU=%.4f  τ_INT=%.4f", dataset, tau1, tau2, tau3)

    G = build_proxy_graph(sensitive, pc, pu, interact, tau1, tau2, tau3)

    tag      = json_path.stem                              # e.g. "german_credit__deepseek_..."
    out_path = json_path.parent / f"{tag}_proxy_graph.png"

    plot_proxy_graph(
        G,
        sensitive_attr=sensitive,
        save_path=out_path,
        title=f"Proxy Path Graph — {dataset} / {model.split('/')[-1]}",
    )
    logger.info("Saved → %s", out_path)


def main():
    p = argparse.ArgumentParser(description="Redraw Proxy Path Graph from JSON results.")
    p.add_argument("files", nargs="*",
                   help="JSON result files (default: all in results/)")
    p.add_argument("--percentile", type=int, default=50,
                   help="Percentile for adaptive thresholds (default: 50)")
    args = p.parse_args()

    json_files = (
        [Path(f) for f in args.files]
        if args.files
        else sorted(RESULTS_DIR.glob("*.json"))
    )

    if not json_files:
        logger.error("No JSON files found in %s", RESULTS_DIR)
        return

    for jf in json_files:
        logger.info("Processing %s ...", jf.name)
        try:
            draw_from_json(jf, percentile=args.percentile)
        except Exception as exc:
            logger.error("Failed %s: %s", jf.name, exc)

    logger.info("Done. %d graph(s) saved.", len(json_files))


if __name__ == "__main__":
    main()
