"""
Vẽ lại toàn bộ biểu đồ (proxy_risk, pc_pu_scatter, mitigation_heatmap,
proxy_graph, proxy_use_barplot) từ các file JSON kết quả đã lưu sẵn.
Không cần gọi LLM lại — chỉ dùng dữ liệu đã có.

Usage:
  python3 redraw_plots.py result_0107
  python3 redraw_plots.py results_2626 result_0107   # nhiều thư mục cùng lúc
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import networkx as nx

from src.reporting import (
    _plot_proxy_risk_bars, _plot_pc_pu_scatter, _plot_mitigation_heatmap,
    plot_proxy_use_bar,
)
from src.proxy_graph import plot_proxy_graph

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("redraw_plots")


def _graph_from_dict(gd: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for n in gd.get("nodes", []):
        n = dict(n)
        node_id = n.pop("id")
        G.add_node(node_id, **n)
    for e in gd.get("edges", []):
        e = dict(e)
        u = e.pop("source")
        v = e.pop("target")
        G.add_edge(u, v, **e)
    return G


def redraw_one(json_path: Path) -> dict | None:
    with open(json_path) as f:
        d = json.load(f)

    tag    = f"fig_{d['dataset']}"
    result = SimpleNamespace(
        dataset=d["dataset"],
        model=d["model"],
        risk_summary=d.get("risk_summary", []),
        tau_capacity=d.get("tau_capacity"),
        tau_use=d.get("tau_use"),
        mitigation_results=[
            SimpleNamespace(strategy=m["strategy"], improvement=m["improvement"])
            for m in d.get("mitigation_results", [])
        ],
    )

    if result.risk_summary:
        _plot_proxy_risk_bars(result, json_path.parent / f"{tag}_proxy_risk.png")
        _plot_pc_pu_scatter(result, json_path.parent / f"{tag}_pc_pu_scatter.png")
    if result.mitigation_results:
        _plot_mitigation_heatmap(result, json_path.parent / f"{tag}_mitigation_heatmap.png")

    gd = d.get("proxy_graph")
    if gd and gd.get("nodes"):
        G = _graph_from_dict(gd)
        sensitive = next(
            (n for n, nd in G.nodes(data=True) if nd.get("node_type") == "sensitive"),
            None,
        )
        if sensitive:
            plot_proxy_graph(
                G, sensitive_attr=sensitive,
                save_path=json_path.parent / f"{tag}_proxy_graph.png",
            )
        else:
            logger.warning("No sensitive node found in %s — skipping proxy graph.", json_path.name)

    pu_input = {
        "dataset":        d["dataset"],
        "proxy_use":      d.get("proxy_use", {}),
        "proxy_capacity": d.get("proxy_capacity", {}),
        "tau_capacity":   d.get("tau_capacity") or 0.0,
        "tau_use":        d.get("tau_use") or 0.0,
    }
    if pu_input["proxy_use"]:
        plot_proxy_use_bar(
            pu_input, json_path.parent / f"{tag}_proxy_bar_plot.png"
        )

    return pu_input


def main():
    p = argparse.ArgumentParser(
        description="Redraw all VAYNE plots from saved JSON results (no LLM calls)."
    )
    p.add_argument("dirs", nargs="+", help="Result directories containing *.json files")
    args = p.parse_args()

    for d in args.dirs:
        results_dir = Path(d)
        json_files  = sorted(results_dir.glob("*.json"))
        if not json_files:
            logger.warning("No JSON files found in %s", results_dir)
            continue

        for jf in json_files:
            logger.info("Redrawing %s ...", jf.name)
            try:
                redraw_one(jf)
            except Exception as exc:
                logger.error("Failed %s: %s", jf.name, exc, exc_info=True)

        logger.info("Done %s: %d result(s) redrawn.", results_dir, len(json_files))


if __name__ == "__main__":
    main()
