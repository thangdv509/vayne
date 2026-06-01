import logging
import pandas as pd
from typing import Optional
from src.config import DATA_DIR

logger = logging.getLogger(__name__)


def load_dataset(config: dict, n_samples: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    path = DATA_DIR / config["file"]
    df = pd.read_csv(path, skipinitialspace=True)

    drop = [c for c in config.get("exclude_cols", []) if c in df.columns]
    if drop:
        df = df.drop(columns=drop)

    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    if n_samples is not None and n_samples < len(df):
        df = _balanced_sample(df, config, n_samples, random_state)

    return df.reset_index(drop=True)


def _balanced_sample(df: pd.DataFrame, config: dict, n: int, random_state: int) -> pd.DataFrame:
    primary_sa    = config["sensitive_attrs"][0]
    protected_val = config["protected_values"][primary_sa]

    prot     = df[df[primary_sa] == protected_val]
    non_prot = df[df[primary_sa] != protected_val]

    n_prot     = min(n // 2, len(prot))
    n_non_prot = min(n - n_prot, len(non_prot))
    n_prot     = min(n - n_non_prot, len(prot))

    sampled = pd.concat([
        prot.sample(n=n_prot,     random_state=random_state),
        non_prot.sample(n=n_non_prot, random_state=random_state),
    ]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    logger.info("Balanced sample: %d protected + %d non-protected = %d total",
                n_prot, n_non_prot, len(sampled))
    return sampled


def get_feature_cols(df: pd.DataFrame, config: dict) -> list[str]:
    """Return all feature columns excluding target and sensitive attributes."""
    exclude = set(config.get("sensitive_attrs", [])) | {config["target_col"]}
    return [c for c in df.columns if c not in exclude]


def get_all_feature_cols(df: pd.DataFrame, config: dict) -> list[str]:
    """Return all feature columns excluding only target (sensitive attrs included)."""
    return [c for c in df.columns if c != config["target_col"]]


def is_protected(row: pd.Series, config: dict) -> bool:
    sa  = config["sensitive_attrs"][0]
    val = config["protected_values"][sa]
    return str(row.get(sa, "")).strip() == str(val).strip()


def get_label(row: pd.Series, config: dict) -> int:
    val = row.get(config["target_col"])
    pos = config["positive_label"]
    # compare as-is first, then fall back to string comparison
    try:
        return 1 if val == pos or str(val).strip() == str(pos).strip() else 0
    except Exception:
        return 0
