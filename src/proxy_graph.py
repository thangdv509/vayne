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
    title: str = "Proxy Path Graph",
) -> None:
    """
    Render the Proxy Path Graph with:
      • Rounded-rectangle nodes (FancyBboxPatch)
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
    from matplotlib.patches import FancyArrowPatch

    if G.number_of_nodes() == 0:
        logger.warning("Empty graph — skipping plot.")
        return

    FONT    = _detect_font()
    COL_GAP = 1.55   # horizontal distance between columns
    ROW_GAP = 1.3    # vertical distance between nodes in same column
    SHRINK  = 32     # arrow shrink in points (clears circle radius)

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

    cols = {
        sensitive_attr: -2 * COL_GAP,
        DECISION_NODE:   2 * COL_GAP,
    }
    for n, y in zip(cap_only, _ys(cap_only)):
        cols[n] = -1 * COL_GAP
    for n, y in zip(both, _ys(both)):
        cols[n] = 0.0
    for n, y in zip(use_only, _ys(use_only)):
        cols[n] = 1 * COL_GAP

    pos = {sensitive_attr: np.array([cols[sensitive_attr], 0.0]),
           DECISION_NODE:  np.array([cols[DECISION_NODE],  0.0])}
    for n, y in zip(cap_only, _ys(cap_only)):
        pos[n] = np.array([cols[n], y])
    for n, y in zip(both, _ys(both)):
        pos[n] = np.array([cols[n], y])
    for n, y in zip(use_only, _ys(use_only)):
        pos[n] = np.array([cols[n], y])

    # ── PRS colour mapping ───────────────────────────────────────────────────
    prs_map = {n: G.nodes[n].get("pc", 0.0) * G.nodes[n].get("pu", 0.0)
               for n in proxy_nodes}
    max_prs = max(prs_map.values()) if prs_map else 1.0

    # node face / border colours
    _FC = {"sensitive": "#E8D5F5", "decision": "#DDE3EA"}
    _EC = {"sensitive": "#9B59B6", "decision": "#5D6D7E"}

    def _node_fc(n):
        nt = G.nodes[n].get("node_type", "proxy")
        if nt in _FC:
            return _FC[nt]
        r = prs_map.get(n, 0.0) / max(max_prs, 1e-9)
        if r > 0.60:
            return "#FFDBA4"   # warm orange — high risk
        if r > 0.25:
            return "#C8EDCC"   # soft green — medium
        return "#C5E3F5"       # soft blue — low

    def _node_ec(n):
        nt = G.nodes[n].get("node_type", "proxy")
        if nt in _EC:
            return _EC[nt]
        r = prs_map.get(n, 0.0) / max(max_prs, 1e-9)
        if r > 0.60:
            return "#E67E22"
        if r > 0.25:
            return "#27AE60"
        return "#2980B9"

    def _node_lw(n):
        nt = G.nodes[n].get("node_type", "proxy")
        return 2.5 if nt == "sensitive" else 1.4

    # ── Edge classification ──────────────────────────────────────────────────
    cap_edges = [(u, v) for u, v, d in G.edges(data=True)
                 if d.get("edge_type") == "capacity"]
    use_edges = [(u, v) for u, v, d in G.edges(data=True)
                 if d.get("edge_type") == "use"]
    int_raw   = [(u, v) for u, v, d in G.edges(data=True)
                 if d.get("edge_type") == "interaction"]
    seen, int_pairs = set(), []
    for u, v in int_raw:
        k = tuple(sorted([u, v]))
        if k not in seen:
            seen.add(k)
            int_pairs.append((u, v))

    # ── Figure setup ─────────────────────────────────────────────────────────
    max_col = max(len(cap_only), len(both), len(use_only), 1)
    fig_h   = max(5.0, max_col * ROW_GAP + 3.0)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    plt.rcParams["font.family"] = FONT

    # ── Draw edges FIRST (below nodes) ───────────────────────────────────────
    def _arrow(u, v, color, ls, rad=0.0, lw=1.7, alpha=1.0):
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={rad}",
                color=color, lw=lw,
                linestyle=ls,
                shrinkA=SHRINK, shrinkB=SHRINK,
                mutation_scale=13,
                alpha=alpha,
            ),
            zorder=2,
        )

    def _edge_label(u, v, text, color="#555555", rad=0.0):
        """Place a small label at the midpoint of an edge."""
        xu, yu = pos[u]; xv, yv = pos[v]
        # Offset label perpendicular to edge direction
        dx, dy = xv - xu, yv - yu
        length = max((dx**2 + dy**2) ** 0.5, 1e-6)
        perp_x, perp_y = -dy / length, dx / length
        offset = 0.12 + abs(rad) * 0.5
        mx = (xu + xv) / 2 + perp_x * offset * np.sign(rad if rad != 0 else 1)
        my = (yu + yv) / 2 + perp_y * offset * np.sign(rad if rad != 0 else 1)
        ax.text(mx, my, text, ha="center", va="center",
                fontsize=6, color=color, fontstyle="italic",
                fontfamily=FONT,
                bbox=dict(boxstyle="round,pad=0.12", fc="white",
                          alpha=0.85, ec="none", lw=0))

    # Capacity edges (A → proxy): straight lines, fan out naturally by y-position
    for u, v in cap_edges:
        _arrow(u, v, "#5B9BD5", "dashed", rad=0.0, lw=1.5, alpha=0.85)

    # Use edges (proxy → Decision): straight lines, fan in naturally by y-position
    for u, v in use_edges:
        _arrow(u, v, "#2C3E50", "solid", rad=0.0, lw=1.8, alpha=0.9)

    # Interaction: bidirectional with alternating curve direction to separate arrows
    for idx, (u, v) in enumerate(int_pairs):
        rad_val = 0.28 if idx % 2 == 0 else -0.28
        _arrow(u, v, "#BBBBBB", "dashed", rad= rad_val, lw=1.2, alpha=0.7)
        _arrow(v, u, "#BBBBBB", "dashed", rad=-rad_val, lw=1.2, alpha=0.7)
        xu, yu = pos[u]; xv, yv = pos[v]
        w = G.edges[u, v].get("weight", 0.0)
        ax.text((xu + xv) / 2, (yu + yv) / 2 + 0.15,
                f"INT={w:.3f}", ha="center", va="bottom",
                fontsize=6, color="#999999", fontstyle="italic",
                fontfamily=FONT,
                bbox=dict(boxstyle="round,pad=0.12", fc="white",
                          alpha=0.8, ec="none", lw=0))

    # ── Draw nodes (circles) ─────────────────────────────────────────────────
    def _short(name: str, limit: int = 10) -> str:
        if len(name) <= limit:
            return name
        parts = name.replace("-", " ").split()
        if len(parts) >= 2:
            l1 = parts[0]
            l2 = " ".join(parts[1:])
            if len(l2) > limit:
                l2 = l2[:limit - 1] + "…"
            return f"{l1}\n{l2}"
        return name[:limit] + "…"

    node_list  = list(G.nodes)
    node_fc    = [_node_fc(n)  for n in node_list]
    node_ec    = [_node_ec(n)  for n in node_list]
    node_lw    = [_node_lw(n)  for n in node_list]
    node_sizes = []
    for n in node_list:
        nt = G.nodes[n].get("node_type", "proxy")
        if nt in ("sensitive", "decision"):
            node_sizes.append(2800)
        else:
            r = prs_map.get(n, 0.0) / max(max_prs, 1e-9)
            node_sizes.append(1800 + int(r * 900))

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_fc,
        node_size=node_sizes,
        node_shape="o",
        linewidths=node_lw,
        edgecolors=node_ec,
        alpha=0.95,
    )

    labels = {n: _short(n) for n in G.nodes}
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=7.5, font_weight="bold",
        font_color="#1A1A2E",
        font_family=FONT,
    )

    # ── Column header badges ──────────────────────────────────────────────────
    all_ys    = [p[1] for p in pos.values()]
    y_badge   = min(all_ys) - ROW_GAP * 0.85

    # Badge colours match the node palette:
    #   A     → lavender  (same as sensitive node)
    #   both  → warm orange (these tend to be high-PRS nodes)
    #   cap / use → neutral, as they contain mixed-risk nodes
    #   dec   → steel gray (same as decision node)
    _BADGE_STYLE = {
        "A":     ("#E8D5F5", "#9B59B6", "Protected\nAttribute (A)"),
        "cap":   ("#F0F0F0", "#AAAAAA", "Capacity\nOnly"),
        "both":  ("#FFE8C2", "#E67E22", "Capacity\n& Use"),
        "use":   ("#F0F0F0", "#AAAAAA", "Use\nOnly"),
        "dec":   ("#DDE3EA", "#5D6D7E", "Decision"),
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
        fc_b, ec_b, text = _BADGE_STYLE[key]
        ax.text(
            x_col, y_badge, text,
            ha="center", va="center",
            fontsize=8, fontfamily=FONT, color="#333333",
            linespacing=1.4,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor=fc_b, edgecolor=ec_b,
                linewidth=1.2, alpha=0.95,
            ),
            zorder=5,
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor="#E8D5F5", edgecolor="#9B59B6", lw=1.4,
                       label="Protected Attribute (A)"),
        mpatches.Patch(facecolor="#FFDBA4", edgecolor="#E67E22", lw=1.4,
                       label="High-risk Proxy"),
        mpatches.Patch(facecolor="#C8EDCC", edgecolor="#27AE60", lw=1.4,
                       label="Medium-risk Proxy"),
        mpatches.Patch(facecolor="#C5E3F5", edgecolor="#2980B9", lw=1.4,
                       label="Low-risk Proxy"),
        mpatches.Patch(facecolor="#DDE3EA", edgecolor="#5D6D7E", lw=1.4,
                       label="Decision"),
        Line2D([0],[0], color="#5B9BD5", lw=1.7, linestyle="dashed",
               label="Capacity edge (PC)"),
        Line2D([0],[0], color="#2C3E50", lw=1.7,
               label="Use edge (PU)"),
        Line2D([0],[0], color="#AAAAAA", lw=1.3, linestyle="dashed",
               label="Interaction (INT)"),
    ]
    leg = ax.legend(
        handles=legend_handles, loc="upper right",
        fontsize=7.5, framealpha=0.95,
        edgecolor="#DDDDDD", title="Legend",
        title_fontsize=8.5, labelspacing=0.5,
        prop={"family": FONT, "size": 7.5},
    )

    ax.set_title(title, fontsize=11.5, fontweight="bold",
                 pad=12, color="#1A1A2E", fontfamily=FONT)
    ax.axis("off")

    # Tight axis limits — no excess whitespace
    all_xs = [p[0] for p in pos.values()]
    all_ys_v = [p[1] for p in pos.values()]
    ax.set_xlim(min(all_xs) - 0.9, max(all_xs) + 0.9)
    ax.set_ylim(y_badge - 0.35, max(all_ys_v) + 0.7)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10, top=0.93)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Proxy graph figure saved → %s", save_path)
    else:
        plt.show()
