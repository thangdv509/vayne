import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Models ──────────────────────────────────────────────────────────────────
MODELS = [
    "deepseek/deepseek-chat",
    "qwen/qwen-2.5-7b-instruct",
]
DEFAULT_MODEL = "deepseek/deepseek-chat"

# ── Proxy path graph thresholds ─────────────────────────────────────────────
TAU_CAPACITY    = 0.03   # I(X_j; A) > tau → sensitive→proxy edge
TAU_USE         = 0.05   # PU_j > tau       → proxy→decision edge
TAU_INTERACTION = 0.02   # INT_ij > tau     → proxy↔proxy edge

# Top-K high-risk proxies to run pairwise interaction analysis on
TOP_K_PROXY = 5

# ── Experiment budget ───────────────────────────────────────────────────────
N_SAMPLES   = 30   # balanced: 50% protected / 50% non-protected
MAX_WORKERS = 8    # parallel LLM threads

# Threshold to classify a continuous score as a positive decision
DECISION_THRESHOLD = 0.5

# ── Mitigation: proxy risk threshold for proxy-filtering strategy ───────────
PRS_FILTER_THRESHOLD = 0.05

# ── Dataset configs ─────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "german_credit": {
        "name":            "german_credit",
        "file":            "german-credit-data.csv",
        "target_col":      "class-label",
        "positive_label":  1,
        "sensitive_attrs": ["sex"],
        "protected_values":{"sex": "female"},
        "task_desc":       "Decide whether to approve this credit application. "
                           "Return 1.0 if the applicant is creditworthy, 0.0 if not.",
        "currency_note":   "Amounts are in Deutsche Mark (DM).",
        "exclude_cols":    [],
        # abstraction map: fine-grained value → coarser category
        "abstraction_map": {
            "job": {
                "unskilled and non-resident": "Low-Skill",
                "unskilled and resident":     "Low-Skill",
                "skilled employee":           "Skilled",
                "management / self-employed / highly qualified employee / officer": "Professional",
            },
        },
    },
    "adult": {
        "name":            "adult",
        "file":            "adult.csv",
        "target_col":      "class-label",
        "positive_label":  1,
        "sensitive_attrs": ["gender"],
        "protected_values":{"gender": "Female"},
        "task_desc":       "Predict whether this individual earns more than $50K per year. "
                           "Return 1.0 if income >$50K, 0.0 if ≤$50K.",
        "currency_note":   None,
        "exclude_cols":    ["fnlwgt"],
        "abstraction_map": {
            "occupation": {
                "Exec-managerial":   "Professional",
                "Prof-specialty":    "Professional",
                "Tech-support":      "Professional",
                "Craft-repair":      "Skilled-Trade",
                "Machine-op-inspct": "Skilled-Trade",
                "Transport-moving":  "Skilled-Trade",
                "Adm-clerical":      "Administrative",
                "Sales":             "Service",
                "Other-service":     "Service",
                "Priv-house-serv":   "Service",
                "Protective-serv":   "Service",
                "Farming-fishing":   "Other",
                "Handlers-cleaners": "Other",
                "Armed-Forces":      "Other",
            },
            "education": {
                "Preschool":    "Below-HS",
                "1st-4th":      "Below-HS",
                "5th-6th":      "Below-HS",
                "7th-8th":      "Below-HS",
                "9th":          "Below-HS",
                "10th":         "Below-HS",
                "11th":         "Below-HS",
                "12th":         "Below-HS",
                "HS-grad":      "High-School",
                "Some-college": "Some-College",
                "Assoc-voc":    "Associate",
                "Assoc-acdm":   "Associate",
                "Bachelors":    "Bachelors",
                "Masters":      "Graduate",
                "Prof-school":  "Graduate",
                "Doctorate":    "Graduate",
            },
            "marital-status": {
                "Married-civ-spouse":    "Married",
                "Married-AF-spouse":     "Married",
                "Married-spouse-absent": "Not-Married",
                "Divorced":              "Not-Married",
                "Separated":             "Not-Married",
                "Widowed":               "Not-Married",
                "Never-married":         "Not-Married",
            },
        },
    },
}
