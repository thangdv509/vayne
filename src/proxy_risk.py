"""
Proxy Risk Score: PRS_j = PC_j × PU_j

A feature becomes a dangerous proxy when it BOTH:
  1. encodes sensitive information (high PC_j), AND
  2. strongly affects the LLM's decision (high PU_j).

PRS is the joint signal used to rank and select high-risk proxies.
"""


def compute_proxy_risk(
    proxy_capacity: dict[str, float],
    proxy_use: dict[str, float],
) -> dict[str, float]:
    """PRS_j = I(X_j; A) * PU_j for every feature present in both dicts."""
    return {
        feat: proxy_capacity[feat] * proxy_use[feat]
        for feat in proxy_capacity
        if feat in proxy_use
    }


def top_k_proxy_risk(proxy_risk: dict[str, float], k: int) -> list[str]:
    """Return the names of the k features with the highest proxy risk score."""
    ranked = sorted(proxy_risk.items(), key=lambda x: -x[1])
    return [feat for feat, _ in ranked[:k]]


def proxy_risk_summary(
    proxy_capacity: dict[str, float],
    proxy_use: dict[str, float],
    proxy_risk: dict[str, float],
) -> list[dict]:
    """Return a list of dicts suitable for tabular display / CSV export."""
    rows = []
    for feat in sorted(proxy_risk, key=lambda f: -proxy_risk[f]):
        rows.append({
            "feature":        feat,
            "proxy_capacity": round(proxy_capacity.get(feat, 0.0), 4),
            "proxy_use":      round(proxy_use.get(feat, 0.0),      4),
            "proxy_risk":     round(proxy_risk[feat],               4),
        })
    return rows
