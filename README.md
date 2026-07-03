# VAYNE

**Explaining and Mitigating Proxy Fairness in LLM-based Tabular Decisions**

VAYNE is a black-box framework that audits Large Language Models (LLMs) used as tabular classifiers (credit approval, loan scoring, etc.). It does two things existing fairness audits don't:

1. **Utility-qualified fairness** — it first checks whether an LLM is actually a competent few-shot tabular classifier before trusting any fairness conclusion drawn from it. A model that can't do the task doesn't produce a meaningful fairness signal.
2. **Proxy-aware explanation** — instead of only reporting *that* a fairness gap exists (via Statistical Parity, Equalized Odds, etc.), it explains *why*: whether the LLM is directly sensitive to the protected attribute, or whether unfairness is transmitted indirectly through non-protected features that act as demographic proxies (e.g. zipcode standing in for race, occupation standing in for gender).

This repository contains the experiment code behind the VAYNE paper.

## How it works

For a black-box scorer `s(x) = p_yes(x) ∈ [0,1]` over a tabular record `x = (a, x_1, ..., x_d)` with protected attribute `A` and non-protected features `X_j`:

| Quantity | Definition | Meaning |
|---|---|---|
| **DU** (Direct Unfairness) | `E_x[ abs(s(x) - s(x_{-A})) ]` | How much masking the protected attribute alone changes the decision. |
| **PC_j** (Proxy Capacity) | `I(X_j; A)` (mutual information) | How much feature `j` statistically encodes the protected attribute. |
| **PU_j** (Proxy Use) | `E_x[ abs(s(x) - s(x_{-j})) ]` | How much the LLM's decision actually changes when feature `j` is masked. |
| **PRS_j** (Proxy Risk Score) | `PC_j × PU_j` | High only when a feature both *encodes* protected info **and** the model *uses* it — the dangerous case. |

`x_{-A}` / `x_{-j}` denote the record with the protected attribute / feature `j` replaced by the sentinel `UNKNOWN`, keeping the rest of the prompt structure intact.

Features ranked by `PRS_j` are assembled into a **Proxy Path Graph** — a directed graph `A → X_j → Decision` where a capacity edge is added when `PC_j > τ₁` and a use edge when `PU_j > τ₂` (adaptive percentile thresholds, see `GRAPH_THRESHOLD_PERCENTILE` in `src/config.py`). A complete `A → X_j → Decision` path is the visual signature of proxy-mediated unfairness.

Group/individual fairness is measured with Statistical Parity (SP), Equalized Odds Difference (EOD), Predictive Parity (PP), ABROCA, and the Theil Index (TI) — see `src/fairness_metrics.py`.

### Mitigation strategies

| Strategy | Type | What it does |
|---|---|---|
| `masking` | VAYNE-native | Replaces the highest-`PRS_j` features with `UNKNOWN` in the prompt. |
| `filtering` | VAYNE-native | Drops every feature whose `PRS_j` exceeds `PRS_FILTER_THRESHOLD` from the prompt entirely. |
| `fair_prompt` | External baseline (Cherepanova et al., AIES 2025) | Appends an explicit demographic-parity instruction to the system prompt; no feature changes. |
| `few_shot_fair` | External baseline (Chhikara et al., arXiv 2402) | Prepends a small set of balanced few-shot examples (one per protected×label cell) plus a fairness rule. |

All four run for every experiment (`src/mitigation.py::STRATEGY_NAMES`), so results are directly comparable in the same table/heatmap.

## Repository structure

```
main.py                    Full VAYNE pipeline: baseline, ablations, proxy graph, mitigation
benchmark.py                Few-shot classifier-utility sweep (model / temperature / k-shot selection)
redraw_plots.py             Regenerate all figures from saved JSON results — no LLM calls
run_missing_strategies.py   Backfill missing mitigation strategies onto existing results (cheap re-run)
synthesize_results.py       Cross-dataset / cross-strategy tables & figures from a results directory
draw_graph.py                Redraw just the Proxy Path Graph from saved JSON, with tunable percentile
plot_benchmark.py           Plot utility vs. temperature for the benchmark.py sweep (k=128, 3 datasets)

src/
  config.py                 Dataset configs, thresholds (τ_PC, τ_PU), model list
  data_loader.py             CSV loading + balanced protected/non-protected sampling
  serializer.py              Row → natural-language prompt (template / markdown formats)
  llm_client.py               OpenRouter client, retry logic, p_yes extraction
  pipeline.py                 run_experiment(): orchestrates the full 12-step VAYNE flow
  proxy_capacity.py           PC_j via mutual information (categorical + k-NN for numeric)
  proxy_use.py                PU_j via single/pairwise feature ablation
  proxy_interaction.py        Pairwise proxy interaction (INT_ij) among top-K risky features
  proxy_risk.py                PRS_j ranking
  proxy_graph.py               Proxy Path Graph construction + rendering
  fairness_metrics.py         DU, SP, EOD, PP, ABROCA, TI
  mitigation.py                The 4 mitigation strategies + dispatcher
  reporting.py                 Saves JSON/CSV/PNG, console tables

data/                        Source CSVs (see Datasets below)
result_0107/, results/       Example experiment output directories
benchmark/, benchmark_ori/   Saved benchmark.py sweep outputs
```

## Setup

```bash
./setup.sh                      # creates .venv, installs requirements.txt
source .venv/bin/activate
cp .env.example .env            # then set OPENROUTER_API_KEY=... inside
```

All LLM calls go through [OpenRouter](https://openrouter.ai) (`src/llm_client.py`), so any OpenRouter-hosted model ID works, e.g. `deepseek/deepseek-v4-flash`, `anthropic/claude-3-haiku`, `openai/gpt-4o-mini`.

## Usage

### 1. Classifier-utility benchmark (pick a model / temperature / k-shot)

Establishes whether few-shot prompting makes the LLM a competent tabular classifier *before* running any fairness analysis on it.

```bash
python benchmark.py --datasets german_credit credit_card credit_approval \
    --models deepseek/deepseek-v4-flash anthropic/claude-3-haiku \
    --n 100 --multi-shot            # sweep k = 2,8,32,128,256,512
python benchmark.py --datasets german_credit --models deepseek/deepseek-v4-flash \
    --n 100 --multi-temp --few-shot --k-shot 128   # sweep temperature at fixed k
```

Results land in `benchmark/{model}_{dataset}.csv`. Visualize with `python plot_benchmark.py` (hardcoded to the claude-3-haiku, k=128 sweep — edit `MODEL_SLUG` for other models).

### 2. Full VAYNE pipeline

Runs baseline inference, direct-unfairness masking, per-feature and pairwise ablations, builds the Proxy Path Graph, then applies all 4 mitigation strategies.

```bash
python main.py \
    --datasets german_credit credit_card credit_approval \
    --models deepseek/deepseek-v4-flash \
    --n 100 --temperature 0.4 --k-shot 128 \
    --results-dir result_0107
```

| Flag | Default | Notes |
|---|---|---|
| `--datasets` | all configured | space-separated dataset keys |
| `--models` | `deepseek/deepseek-chat` | OpenRouter model ID(s) |
| `--n` | `30` | balanced samples per dataset (50/50 protected split where possible) |
| `--temperature` | `0.1` | applied to every inference call |
| `--k-shot` | `0` | few-shot demonstrations sampled from the full (unsampled) dataset per row |
| `--results-dir` | `./results` | output directory |

⚠️ **Cost note:** the pipeline makes roughly `n × (2 + n_features + 10 + 4)` LLM calls per dataset (baseline+masked, per-feature ablation, top-5 pairwise ablation, 4 mitigation strategies) — this scales quickly with `n` and dataset width. Sanity-check the call count before running large `n` on wide datasets (e.g. `pakdd`, 42 features).

### 3. Post-hoc utilities (no LLM calls)

```bash
python redraw_plots.py result_0107               # regenerate all figures from saved JSON
python draw_graph.py result_0107/*.json --percentile 75   # just the proxy graph, different threshold
python synthesize_results.py --results-dir result_0107     # cross-dataset comparison tables/figures
python run_missing_strategies.py result_0107/*.json \
    --temperature 0.4 --k-shot 128              # backfill a strategy added after the original run
```

## Datasets

Configured in `src/config.py::DATASET_CONFIGS`. Only the first three ship their source CSV in `data/`:

| Key | Source file | Protected attribute |
|---|---|---|
| `german_credit` | `german-data-credit.csv` | `sex` |
| `credit_card` | `credit-card-clients.csv` | `SEX` |
| `credit_approval` | `credit-approval.data` | `Male` |

## Output artifacts

For each `(dataset, model)` experiment, `src/reporting.py::save_results()` writes to `--results-dir`:

- `{dataset}__{model}.json` — full result: proxy capacity/use/risk per feature, the proxy graph, baseline fairness, and all mitigation results.
- `{dataset}__{model}_proxy_risk.csv`, `_interactions.csv`, `_mitigation.csv`
- `fig_{dataset}_proxy_risk.png`, `_pc_pu_scatter.png`, `_mitigation_heatmap.png`, `_proxy_bar_plot.png`, `_proxy_graph.png` — figures are named by dataset only (no model tag, no title baked in — for figure captions in a paper).

## Notes

- `.gitignore` currently excludes `*.md` and `*.sh` repo-wide — if you want this README (or `setup.sh`) tracked in git, add an explicit negation (`!README.md`) to `.gitignore`.
- There is no `LICENSE` file yet.

<!-- ## Citation

A formal citation will be added once the paper is published. In the meantime, cite the repository directly:

```bibtex
@misc{vayne2026,
  title  = {VAYNE: Explaining and Mitigating Proxy Fairness in LLM-based Tabular Decisions},
  author = {Giang Thi Thu, Huyen and Doan Viet, Thang and Nguyen Thanh, Trung and
            Ta Quoc, Tuan and Le Quy, Tai and Nguyen Long, Giang and Ban, Ha-Bang},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```
-->

