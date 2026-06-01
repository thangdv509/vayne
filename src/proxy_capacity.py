"""
Proxy Capacity: PC_j = I(X_j ; A)

Measures how much each feature X_j encodes the protected attribute A,
using mutual information from scikit-learn (handles both categorical and
continuous features via discretization internally).

High PC_j → feature is a strong statistical proxy for the sensitive attribute.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def compute_proxy_capacity(
    df: pd.DataFrame,
    feature_cols: list[str],
    sensitive_col: str,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Compute I(X_j; A) for every feature in feature_cols.

    Returns dict: feature_name → mutual information (non-negative float).
    """
    le = LabelEncoder()
    A_encoded = le.fit_transform(df[sensitive_col].astype(str))

    X = pd.DataFrame(index=df.index)
    discrete_mask = []

    for col in feature_cols:
        series = df[col]
        if series.dtype == object or str(series.dtype) == "category":
            le2 = LabelEncoder()
            X[col] = le2.fit_transform(series.astype(str))
            discrete_mask.append(True)
        else:
            X[col] = pd.to_numeric(series, errors="coerce").fillna(0)
            discrete_mask.append(False)

    mi_values = mutual_info_classif(
        X, A_encoded,
        discrete_features=discrete_mask,
        random_state=random_state,
    )

    result = {col: float(mi) for col, mi in zip(feature_cols, mi_values)}

    logger.info("Proxy capacity computed for %d features. Top-3: %s",
                len(result),
                sorted(result.items(), key=lambda x: -x[1])[:3])
    return result


def rank_by_capacity(proxy_capacity: dict[str, float]) -> list[tuple[str, float]]:
    """Return features sorted by proxy capacity descending."""
    return sorted(proxy_capacity.items(), key=lambda x: -x[1])
