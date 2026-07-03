"""
Proxy Path Graph

Constructs a directed graph explaining how unfairness propagates:

    Protected Attribute (A)
        |  (dashed, capacity edge if PC_j > τ₁)
    Proxy Features
        |  (solid, use edge if PU_j > τ₂)
    Decision

Additional proxy↔proxy dashed edges if INT_ij > τ₃.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

DECISION_NODE = "DECISION"


# ── Graph construction ────────────────────────────────────────────────────────

def build_proxy_graph(
    sensitive_attr:   str,
    proxy_capacity:   dict[str, float],
    proxy_use:        dict[str, float],
    proxy_interaction: dict[tuple[str, str], float],
    tau_capacity:    float = 0.03,
    tau_use:         float = 0.05,
    tau_interaction: float = 0.02,
) -> nx.DiGraph:
    """
    Build and return the proxy path graph as a NetworkX DiGraph.

    Node types : 'sensitive', 'proxy', 'decision'
    Edge types : 'capacity' | 'use' | 'interaction'
    """
    G = nx.DiGraph()
    G.add_node(sensitive_attr, node_type="sensitive")
    G.add_node(DECISION_NODE,  node_type="decision")

    all_features = (set(proxy_capacity) | set(proxy_use)) - {sensitive_attr}
    for feat in all_features:
        pc = proxy_capacity.get(feat, 0.0)
        pu = proxy_use.get(feat, 0.0)

        if pc > tau_capacity or pu > tau_use:
            G.add_node(feat, node_type="proxy", pc=round(pc, 4), pu=round(pu, 4))
        if pc > tau_capacity:
            G.add_edge(sensitive_attr, feat, weight=round(pc, 4), edge_type="capacity")
        if pu > tau_use:
            G.add_edge(feat, DECISION_NODE, weight=round(pu, 4), edge_type="use")

    for (fi, fj), score in proxy_interaction.items():
        if score > tau_interaction and fi in G.nodes and fj in G.nodes:
            G.add_edge(fi, fj, weight=round(score, 4), edge_type="interaction")
            G.add_edge(fj, fi, weight=round(score, 4), edge_type="interaction")

    logger.info("Proxy graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def get_proxy_paths(G: nx.DiGraph, sensitive_attr: str) -> list[list[str]]:
    try:
        return list(nx.all_simple_paths(G, sensitive_attr, DECISION_NODE))
    except (nx.NetworkXError, nx.NodeNotFound):
        return []


def graph_to_dict(G: nx.DiGraph) -> dict[str, Any]:
    return {
        "nodes": [{"id": n, **G.nodes[n]} for n in G.nodes],
        "edges": [{"source": u, "target": v, **G.edges[u, v]} for u, v in G.edges],
    }


def graph_to_ascii(G: nx.DiGraph, sensitive_attr: str) -> str:
    paths = get_proxy_paths(G, sensitive_attr)
    if not paths:
        return f"{sensitive_attr} → (no proxy paths found)"
    lines = [sensitive_attr]
    for path in paths:
        parts = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            w  = G.edges[u, v].get("weight", "?")
            et = G.edges[u, v].get("edge_type", "")
            parts.append(f"─[{et}={w}]─► {v}")
        lines.append("  " + "".join(parts))
    return "\n".join(lines)


# ── Visualization ─────────────────────────────────────────────────────────────

def _detect_font() -> str:
    """Return Roboto if installed, else best available sans-serif."""
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for name in ("Roboto", "Liberation Sans", "Noto Sans", "DejaVu Sans"):
        if name in available:
            return name
    return "sans-serif"


def plot_proxy_graph(
    G: nx.DiGraph,
    sensitive_attr: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Render the Proxy Path Graph with:
      • Large, high-contrast circular nodes
      • Roboto font (fallback to Liberation Sans / DejaVu Sans)
      • Structured column layout: A | Capacity Only | Capacity & Use | Use Only | Decision
      • Colour-coded column header badges at the bottom
      • Dashed blue  : A → proxy  (capacity edge)
      • Solid dark   : proxy → Decision  (use edge)
      • Dashed gray  : proxy ↔ proxy  (interaction edge)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    if G.number_of_nodes() == 0:
        logger.warning("Empty graph — skipping plot.")
        return

    FONT    = _detect_font()
    COL_GAP = 1.9    # horizontal spacing between columns
    ROW_GAP = 1.2    # vertical spacing between nodes in same column
    NODE_S  = 5500   # uniform node size (points²)
    # node radius in pt = sqrt(NODE_S/π); +12 for border lw + clearance
    SHRINK  = int(np.sqrt(NODE_S / np.pi)) + 12   # ≈ 54 pt
    LW_CAP  = 2.0    # capacity edge line width
    LW_USE  = 2.8    # use edge line width (thicker)

    proxy_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "proxy"]

    # ── Column layout ────────────────────────────────────────────────────────
    cap_set  = {v for u, v, d in G.edges(data=True)
                if d.get("edge_type") == "capacity" and u == sensitive_attr}
    use_set  = {u for u, v, d in G.edges(data=True)
                if d.get("edge_type") == "use" and v == DECISION_NODE}
    both     = sorted(cap_set & use_set)
    cap_only = sorted(cap_set - use_set)
    use_only = sorted(use_set - cap_set)

    def _ys(nodes):
        n = len(nodes)
        return [(i - (n - 1) / 2) * ROW_GAP for i in range(n)]

    cols = {sensitive_attr: -2 * COL_GAP, DECISION_NODE: 2 * COL_GAP}
    for n, y in zip(cap_only, _ys(cap_only)): cols[n] = -COL_GAP
    for n, y in zip(both,     _ys(both)):     cols[n] = 0.0
    for n, y in zip(use_only, _ys(use_only)): cols[n] =  COL_GAP

    pos = {sensitive_attr: np.array([cols[sensitive_attr], 0.0]),
           DECISION_NODE:  np.array([cols[DECISION_NODE],  0.0])}
    for n, y in zip(cap_only, _ys(cap_only)): pos[n] = np.array([cols[n], y])
    for n, y in zip(both,     _ys(both)):     pos[n] = np.array([cols[n], y])
    for n, y in zip(use_only, _ys(use_only)): pos[n] = np.array([cols[n], y])

    # ── PRS colour mapping ───────────────────────────────────────────────────
    prs_map = {n: G.nodes[n].get("pc", 0.0) * G.nodes[n].get("pu", 0.0)
               for n in proxy_nodes}
    max_prs = max(prs_map.values()) if prs_map else 1.0

    def _node_fc(n):
        nt = G.nodes[n].get("node_type", "proxy")
        if nt == "sensitive": return "#E8D5F5"
        if nt == "decision":  return "#DDE3EA"
        r = prs_map.get(n, 0.0) / max(max_prs, 1e-9)
        if r > 0.60: return "#FFDBA4"
        if r > 0.25: return "#C8EDCC"
        return "#C5E3F5"

    def _node_ec(n):
        nt = G.nodes[n].get("node_type", "proxy")
        if nt == "sensitive": return "#9B59B6"
        if nt == "decision":  return "#7F8C8D"
        r = prs_map.get(n, 0.0) / max(max_prs, 1e-9)
        if r > 0.60: return "#E67E22"
        if r > 0.25: return "#27AE60"
        return "#2980B9"

    def _node_lw(n):
        return 3.0 if G.nodes[n].get("node_type") == "sensitive" else 2.0

    # ── Edge classification ──────────────────────────────────────────────────
    cap_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "capacity"]
    use_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "use"]
    int_raw   = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "interaction"]
    seen, int_pairs = set(), []
    for u, v in int_raw:
        k = tuple(sorted([u, v]))
        if k not in seen:
            seen.add(k)
            int_pairs.append((u, v))

    # ── Figure ───────────────────────────────────────────────────────────────
    max_col = max(len(cap_only), len(both), len(use_only), 1)
    fig_h   = max(7.0, max_col * ROW_GAP + 4.0)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plt.rcParams["font.family"] = FONT

    # ── Edges ────────────────────────────────────────────────────────────────
    # Edges that skip a column (sensitive→both, both→DECISION) get a small
    # curvature so they don't overlap straight edges to adjacent columns.
    both_set = set(both)

    def _arrow(u, v, color, ls, lw, rad=None, alpha=1.0):
        if rad is None:
            skip = (u == sensitive_attr and v in both_set) or \
                   (u in both_set and v == DECISION_NODE)
            rad = 0.12 if skip else 0.0
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={rad}",
                color=color, lw=lw,
                linestyle=ls,
                shrinkA=SHRINK, shrinkB=SHRINK,
                mutation_scale=18,
                alpha=alpha,
            ),
            zorder=2,
        )

    for u, v in cap_edges:
        _arrow(u, v, "#4A90D9", "dashed", LW_CAP, alpha=0.9)
    for u, v in use_edges:
        _arrow(u, v, "#1A252F", "solid",  LW_USE, alpha=0.95)

    # Interaction: bidirectional with alternating curve direction to separate arrows
    for idx, (u, v) in enumerate(int_pairs):
        rad_val = 0.28 if idx % 2 == 0 else -0.28
        _arrow(u, v, "#BBBBBB", "dashed", 1.4, rad= rad_val, alpha=0.7)
        _arrow(v, u, "#BBBBBB", "dashed", 1.4, rad=-rad_val, alpha=0.7)
        xu, yu = pos[u]; xv, yv = pos[v]
        w = G.edges[u, v].get("weight", 0.0)
        ax.text((xu + xv) / 2, (yu + yv) / 2 + 0.18,
                f"INT={w:.3f}", ha="center", va="bottom",
                fontsize=10, color="#999999", fontstyle="italic",
                fontfamily=FONT,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          alpha=0.85, ec="none", lw=0))

    # ── Nodes ────────────────────────────────────────────────────────────────
    def _label(name: str, max_ch: int = 9) -> str:
        """Fit name into two lines of max_ch chars, truncate with … if needed."""
        words = name.replace("-", " ").split()
        if len(words) == 1:
            return name[:max_ch] + ("…" if len(name) > max_ch else "")
        # Find split that balances both lines
        best_i, best_diff = 1, float("inf")
        for i in range(1, len(words)):
            diff = abs(len(" ".join(words[:i])) - len(" ".join(words[i:])))
            if diff < best_diff:
                best_diff, best_i = diff, i
        l1 = " ".join(words[:best_i])
        l2 = " ".join(words[best_i:])
        if len(l1) > max_ch: l1 = l1[:max_ch - 1] + "…"
        if len(l2) > max_ch: l2 = l2[:max_ch - 1] + "…"
        return f"{l1}\n{l2}"

    node_list = list(G.nodes)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=[_node_fc(n) for n in node_list],
        node_size=[NODE_S for _ in node_list],
        node_shape="o",
        linewidths=[_node_lw(n) for n in node_list],
        edgecolors=[_node_ec(n) for n in node_list],
        alpha=1.0,
    )
    nx.draw_networkx_labels(
        G, pos, {n: _label(n) for n in G.nodes}, ax=ax,
        font_size=13, font_weight="bold",
        font_color="#1A1A2E", font_family=FONT,
    )

    # ── Column badges ─────────────────────────────────────────────────────────
    all_ys  = [p[1] for p in pos.values()]
    y_badge = min(all_ys) - ROW_GAP * 0.9
    _BADGE = {
        "A":    ("#E8D5F5", "#9B59B6", "Protected Attribute (A)"),
        "cap":  ("#F4F4F4", "#AAAAAA", "Capacity Only"),
        "both": ("#FFE8C2", "#E67E22", "Capacity & Use"),
        "use":  ("#F4F4F4", "#AAAAAA", "Use Only"),
        "dec":  ("#DDE3EA", "#7F8C8D", "Decision"),
    }
    for x_col, key, show in [
        (cols[sensitive_attr], "A",   True),
        (-COL_GAP,            "cap",  bool(cap_only)),
        (0.0,                 "both", bool(both)),
        ( COL_GAP,            "use",  bool(use_only)),
        (cols[DECISION_NODE], "dec",  True),
    ]:
        if not show:
            continue
        fc_b, ec_b, text = _BADGE[key]
        ax.text(x_col, y_badge, text, ha="center", va="center",
                fontsize=15, fontfamily=FONT, color="#333333",
                bbox=dict(boxstyle="round,pad=0.55", facecolor=fc_b,
                          edgecolor=ec_b, linewidth=1.5, alpha=0.95),
                zorder=5)

    # ── Legend — placed outside axes to the right ────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor="#E8D5F5", edgecolor="#9B59B6", lw=1.8,
                       label="Protected Attribute (A)"),
        mpatches.Patch(facecolor="#FFDBA4", edgecolor="#E67E22", lw=1.8,
                       label="High-risk Proxy"),
        mpatches.Patch(facecolor="#C8EDCC", edgecolor="#27AE60", lw=1.8,
                       label="Medium-risk Proxy"),
        mpatches.Patch(facecolor="#C5E3F5", edgecolor="#2980B9", lw=1.8,
                       label="Low-risk Proxy"),
        mpatches.Patch(facecolor="#DDE3EA", edgecolor="#7F8C8D", lw=1.8,
                       label="Decision"),
        Line2D([0], [0], color="#4A90D9", lw=2.0, linestyle="dashed",
               label="Capacity edge (PC)"),
        Line2D([0], [0], color="#1A252F", lw=2.5,
               label="Use edge (PU)"),
        Line2D([0], [0], color="#AAAAAA", lw=1.4, linestyle="dashed",
               label="Interaction (INT)"),
    ]
    ax.legend(handles=legend_handles,
              loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0, fontsize=14, framealpha=1.0,
              edgecolor="#CCCCCC", labelspacing=0.7,
              prop={"family": FONT, "size": 14})

    ax.axis("off")

    # Tight axis limits — no excess whitespace
    all_xs = [p[0] for p in pos.values()]
    ax.set_xlim(min(all_xs) - 0.9, max(all_xs) + 0.9)
    ax.set_ylim(y_badge - 0.35, max(all_ys) + 0.7)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10, top=0.97, right=0.78)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Proxy graph figure saved → %s", save_path)
    else:
        plt.show()
