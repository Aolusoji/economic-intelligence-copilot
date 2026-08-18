# Economic Intelligence Copilot (EIC)

**An Open-Source Predictive Economic Decision-Support Framework for U.S. Trade Policy, Industrial Strategy, and Domestic Manufacturing Resilience**

Author: Adenekan Olusoji

---

## What this is

EIC integrates real U.S. federal economic data — from the Bureau of Economic Analysis, the Bureau of Labor Statistics, and the U.S. International Trade Commission — into a single manufacturing-subsector panel, and applies panel econometrics, out-of-sample forecast evaluation, and model-implied scenario analysis to support trade-policy and industrial-strategy decisions.

It is the empirical companion to the working paper *"Development of Predictive Economic Decision-Support Frameworks for Modernizing U.S. Trade and Industrial Policy and Strengthening Domestic Manufacturing Resilience and Competitiveness"* (Olusoji, 2026), and implements the first three phases of that paper's proposed five-phase methodology.

Every number in this repository — every regression coefficient, every forecast-accuracy figure, every scenario estimate — is computed from real, cited government data. Nothing is simulated or illustrative unless explicitly labeled as such.

## What's in the repository

```
eic/
├── README.md                  This file
├── LICENSE                    MIT License
├── requirements.txt           Python dependencies
├── data/
│   └── eic_panel_dataset.csv  The full real panel: 19 manufacturing subsectors x 11 years (2015-2025), N=209
├── src/                       The data pipeline and analysis, in run order
│   ├── 01_build_employment.py         Assembles BLS Current Employment Statistics by subsector
│   ├── 02_build_hhi_concentration.py  Computes the Supplier Concentration Indicator (HHI) from USITC country-of-origin data
│   ├── 03_build_productivity.py       Assembles BLS Industry Productivity data by subsector
│   ├── 04_merge_panel.py              Merges BEA, BLS, and USITC sources into the final panel
│   ├── 05_panel_regression.py         Two-way fixed-effects panel regression (the main model)
│   ├── 06_robustness_checks.py        Alternative specifications + corrected Hausman test
│   ├── 07_predictive_model.py         Genuine ex-ante forecast evaluation (rolling-origin, lagged predictors)
│   └── 08_additional_diagnostics.py   Corrected Hausman/Mundlak tests, wild-cluster bootstrap, interaction term, VIF, diagnostics
├── docs/
│   ├── data_sources.md         Full citation and construction methodology for every variable
│   └── methodology.md          The econometric and forecasting methodology
└── dashboard/
    └── index.html               Self-contained interactive dashboard (open directly in any browser)
```

## Using this framework with your own data

The analysis scripts (`05`–`07`) are not hardcoded to these 19 manufacturing subsectors or to 2015–2025. They operate purely on column structure, so this framework can be pointed at a different industry set, a different country, or an extended date range without modifying the model, feature set, or evaluation logic.

**To use it with your own data**, build a CSV with one row per entity-period observation and these columns:

| Column | Type | Description |
|---|---|---|
| `entity` | string | The unit of analysis (an industry, a region, a company — whatever your cross-sectional dimension is) |
| `year` | integer | The time period |
| `gross_output_billions` | float | Your dependent variable (output, revenue, or another outcome measure) |
| `employment_thousands` | float | A labor input measure |
| `labor_productivity_index` | float | A productivity measure |
| `imports_usd` | float | A trade/input variable |
| `exports_usd` | float | A trade/output variable |
| `hhi` | float | A concentration or risk indicator (0–10,000 scale, or any comparable index) |

Then:

```bash
cp your_data.csv data/eic_panel_dataset.csv
cd src
python 05_panel_regression.py    # your panel regression, same methodology
python 06_robustness_checks.py   # your robustness checks and Hausman test
python 07_predictive_model.py    # your genuine ex-ante rolling-origin forecast comparison
python 08_additional_diagnostics.py  # corrected Hausman/Mundlak, wild-cluster bootstrap, interaction term
```

For `07_predictive_model.py`, edit the two configuration variables at the top of the file (`DATA_PATH` and `ORIGIN_YEARS`, a list of rolling forecast-origin years) to match your date range — everything else adapts automatically to however many entities and years your data actually contains.

**What is now genuinely dynamic**: the dashboard (`dashboard/index.html`) includes a working CSV upload feature (sidebar → "Load Your Own Data"). Upload a CSV with the required column schema and the Overview KPIs, both charts, and the Scenario Simulator's entity list all recompute live and correctly from your data — this was verified with a full DOM-level test, not just visual inspection. A "Reset to original data" button reverts everything back.

**What deliberately stays fixed**: the Panel Regression, Robustness Checks, and Forecast Evaluation tables always show the original study's results, with a clearly visible note when custom data is loaded. This is intentional: a two-way fixed-effects panel regression with clustered standard errors and a Hausman test involves real statistical computation (matrix algebra, degrees-of-freedom corrections, hypothesis testing) that would be irresponsible to reimplement in browser JavaScript without rigorous validation — a subtly wrong coefficient or p-value displayed as fact is worse than no live update at all. To get real regression results for your own data, run the Python scripts in `src/` against your CSV.

## Quick start

Scripts `05` through `07` run directly against the included dataset (`data/eic_panel_dataset.csv`):

```bash
pip install -r requirements.txt
cd src
python 05_panel_regression.py      # reproduces Table 4.2 of the paper
python 06_robustness_checks.py     # reproduces Table 4.3 and the Hausman test
python 07_predictive_model.py      # reproduces the genuine ex-ante rolling-origin forecast comparison
python 08_additional_diagnostics.py  # reproduces the corrected Hausman/Mundlak tests, wild-cluster bootstrap, and interaction term
```

Scripts `01` through `04` document the exact original data-construction pipeline (BLS employment assembly, HHI computation, BLS productivity assembly, and the final merge) but require the raw agency source pulls described in `docs/data_sources.md` (BEA, BLS, USITC downloads), which are not redistributed in this repository due to size and licensing considerations. They are included for methodological transparency and reproducibility of the construction process, not as a one-command pipeline.

Or just open `dashboard/index.html` in a browser — no installation required.

## Key findings

**This table was corrected following independent methodological review** — see `08_additional_diagnostics.py` and the note below for what changed and why.

| Result | Finding |
|---|---|
| Panel regression (Table 4.2) | Trade activity (imports, exports) is positively associated with manufacturing output under asymptotic entity-clustered SEs, but **not statistically significant under a wild-cluster bootstrap** (the more appropriate test given only 19 entity clusters). Import-Origin Concentration (HHI) is negatively associated but not statistically significant under any specification. |
| Robustness (Table 4.3-4.4) | A **corrected, like-for-like Hausman test** (χ²(5)=5.36, p=0.374) does not decisively favor Fixed Effects; a **Mundlak test** (χ²(5)=48.97, p<0.0001) does. Both are reported. An earlier version of this table incorrectly treated significance in *weaker* specifications as "rescuing" the main model's null result — that reasoning has been corrected. |
| Trade × concentration interaction | Tested directly (as recommended by review): not statistically significant (p=0.653). No evidence in this panel that concentrated subsectors are more vulnerable to a shock than diversified ones. |
| Forecasting (genuine ex-ante, rolling-origin) | **Rebuilt entirely** — an earlier version used realized future predictor values (conditional prediction, not forecasting). The corrected, genuine ex-ante version (lagged predictors, retrained at 4 rolling origins) finds a naive persistence benchmark even harder to beat (4.94% avg MAPE vs. 8%+ for every model-based approach). |
| Scenario analysis | Relabeled an **"Exploratory Model-Implied Scenario Tool"** — illustrative arithmetic on regression coefficients, explicitly not a causal policy simulator. Example: a 20% import disruption is associated with a ~4.4% illustrative output change. |

### What was corrected, and why

An independent peer-review-style assessment of this project identified several real issues, all confirmed against the actual code before being fixed:
1. **The "out-of-sample forecast" used realized future predictor values** — this was conditional prediction, not genuine forecasting. Rebuilt with lagged predictors and a rolling-origin design.
2. **The Hausman test compared non-like-for-like models** (Random Effects without year effects vs. Fixed Effects with year effects). Corrected, and a Mundlak test added as a complementary check.
3. **Standard errors were described as "entity- and time-clustered" but the code only clusters by entity.** Description corrected everywhere; a wild-cluster bootstrap added given the small number of clusters (19).
4. **"Supplier Concentration Indicator" was renamed to "Import-Origin Concentration Index"** to precisely describe what a country-of-origin HHI actually measures (not firm-level supplier data).
5. **A trade × concentration interaction term was added and tested** rather than only noted as a limitation — found not significant, reported as a real null result.

## Data sources

- U.S. Bureau of Economic Analysis — Gross Output by Industry
- U.S. Bureau of Labor Statistics — Current Employment Statistics
- U.S. Bureau of Labor Statistics — Industry Productivity Program
- U.S. International Trade Commission — DataWeb (imports, exports, and country-of-origin trade data)

Full citations and construction methodology are in `docs/data_sources.md`.

## What this framework does NOT yet do

In the interest of the same transparency applied throughout the underlying research, this framework's current scope is limited. It does not yet include: a formal dependency network mapping specific foreign suppliers to specific domestic manufacturers; economic-impact models estimating disruption effects on prices and employment; scenario simulation with trade×concentration interaction effects; retrospective validation against specific historical disruption episodes; or a dynamic panel (GMM) specification. These are documented as directions for future work in the accompanying paper.

## Citation

```
Olusoji, A. (2026). Economic Intelligence Copilot (EIC): An Open-Source Predictive Economic
Decision-Support Framework for U.S. Trade Policy, Industrial Strategy, and Domestic
Manufacturing Resilience. [Software]. Companion to: Development of Predictive Economic
Decision-Support Frameworks for Modernizing U.S. Trade and Industrial Policy and
Strengthening Domestic Manufacturing Resilience and Competitiveness. Unpublished working paper.
```

## License

MIT License — see `LICENSE`. This framework is released as open-source specifically so that its methodology, data-integration approach, and findings can be reviewed, replicated, and extended by other researchers, policymakers, and manufacturers, consistent with the national-scaling and dissemination goals described in the companion paper.
