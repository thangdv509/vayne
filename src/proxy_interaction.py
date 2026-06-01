"""
Proxy Interaction: INT_ij = PU_ij - (PU_i + PU_j)

Positive INT_ij means that features i and j *jointly* affect the LLM's decision
beyond their individual contributions — they form a combined proxy pathway.

Example: zipcode alone (PU_i = 0.04) and occupation alone (PU_j = 0.03)
each have moderate proxy use, but together (PU_ij = 0.12) the interaction is
0.12 - 0.07 = 0.05, indicating a synergistic proxy pathway.
"""


def compute_proxy_interaction(
    proxy_use: dict[str, float],
    pairwise_proxy_use: dict[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    """
    INT_ij = PU_ij - (PU_i + PU_j)

    Returns dict: (feat_i, feat_j) → interaction score.
    Negative values are clipped to 0 (no meaningful joint contribution).
    """
    result = {}
    for (fi, fj), pu_ij in pairwise_proxy_use.items():
        pu_i = proxy_use.get(fi, 0.0)
        pu_j = proxy_use.get(fj, 0.0)
        result[(fi, fj)] = max(0.0, pu_ij - (pu_i + pu_j))
    return result


def interaction_summary(interactions: dict[tuple[str, str], float]) -> list[dict]:
    """Return sorted list of dicts for display / export."""
    rows = []
    for (fi, fj), score in sorted(interactions.items(), key=lambda x: -x[1]):
        rows.append({
            "feature_i":   fi,
            "feature_j":   fj,
            "interaction": round(score, 4),
        })
    return rows
