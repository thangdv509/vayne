"""
Proxy Path Graph

Constructs a directed graph explaining how unfairness propagates:

    Sensitive Attribute
           ↓  (edge if PC_j > TAU_CAPACITY)
      Proxy Features
           ↓  (edge if PU_j > TAU_USE)
         Decision

Additional proxy↔proxy edges if INT_ij > TAU_INTERACTION.

Requires networkx.
"""

import logging
from typing import Any
import networkx as nx

logger = logging.getLogger(__name__)

DECISION_NODE = "DECISION"


def build_proxy_graph(
    sensitive_attr: str,
    proxy_capacity: dict[str, float],
    proxy_use: dict[str, float],
    proxy_interaction: dict[tuple[str, str], float],
    tau_capacity:    float = 0.03,
    tau_use:         float = 0.05,
    tau_interaction: float = 0.02,
) -> nx.DiGraph:
    """
    Build and return the proxy path graph as a NetworkX DiGraph.

    Node types: 'sensitive', 'proxy', 'decision'
    Edge types: 'capacity' | 'use' | 'interaction'
    """
    G = nx.DiGraph()
    G.add_node(sensitive_attr, node_type="sensitive")
    G.add_node(DECISION_NODE,  node_type="decision")

    # Add proxy nodes + edges
    all_features = set(proxy_capacity) | set(proxy_use)
    for feat in all_features:
        pc = proxy_capacity.get(feat, 0.0)
        pu = proxy_use.get(feat, 0.0)

        if pc > tau_capacity or pu > tau_use:
            G.add_node(feat, node_type="proxy", pc=round(pc, 4), pu=round(pu, 4))

        if pc > tau_capacity:
            G.add_edge(sensitive_attr, feat,
                       weight=round(pc, 4), edge_type="capacity")

        if pu > tau_use:
            G.add_edge(feat, DECISION_NODE,
                       weight=round(pu, 4), edge_type="use")

    # Add proxy↔proxy interaction edges
    for (fi, fj), score in proxy_interaction.items():
        if score > tau_interaction and fi in G.nodes and fj in G.nodes:
            G.add_edge(fi, fj, weight=round(score, 4), edge_type="interaction")
            G.add_edge(fj, fi, weight=round(score, 4), edge_type="interaction")

    logger.info("Proxy graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def get_proxy_paths(G: nx.DiGraph, sensitive_attr: str) -> list[list[str]]:
    """Return all simple paths from sensitive_attr to DECISION."""
    try:
        return list(nx.all_simple_paths(G, sensitive_attr, DECISION_NODE))
    except (nx.NetworkXError, nx.NodeNotFound):
        return []


def graph_to_dict(G: nx.DiGraph) -> dict[str, Any]:
    """Serializable representation of the graph."""
    return {
        "nodes": [{"id": n, **G.nodes[n]} for n in G.nodes],
        "edges": [{"source": u, "target": v, **G.edges[u, v]} for u, v in G.edges],
    }


def graph_to_ascii(G: nx.DiGraph, sensitive_attr: str) -> str:
    """
    Return a simple text representation of proxy paths for console output.

    Example:
        Sex
         └─[capacity=0.12]─► job ─[use=0.08]─► DECISION
         └─[capacity=0.09]─► marital-status ─[use=0.06]─► DECISION
    """
    paths = get_proxy_paths(G, sensitive_attr)
    if not paths:
        return f"{sensitive_attr} → (no proxy paths found)"

    lines = [sensitive_attr]
    for path in paths:
        parts = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            w = G.edges[u, v].get("weight", "?")
            et = G.edges[u, v].get("edge_type", "")
            parts.append(f"─[{et}={w}]─► {v}")
        lines.append("  " + "".join(parts))
    return "\n".join(lines)
