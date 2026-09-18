# Economic Intelligence Copilot (EIC)

**An Open-Source Predictive Economic Decision-Support Framework for U.S. Trade Policy, Industrial Strategy, and Domestic Manufacturing Resilience**

Author: Olusoji Adenekan

## About

EIC integrates U.S. federal economic data from the Bureau of Economic Analysis (BEA), Bureau of Labor Statistics (BLS), and U.S. International Trade Commission (USITC) into a manufacturing-subsector panel. It combines panel econometrics, forecast evaluation, concentration analysis, and model-implied scenario analysis in an interactive decision-support interface.

The empirical panel covers 19 U.S. manufacturing subsectors from 2015–2025 (N = 209 industry-year observations).

## Live dashboard

The repository homepage is the interactive EIC dashboard served by `index.html`.

## Repository files

All project files are stored directly in the repository root. The repository includes the interactive dashboard, research paper, panel dataset, example data template, methodology and data-source documentation, Python analysis scripts, dependency list, and MIT license.

## Run the analysis

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then run the analysis scripts from the repository root, for example:

```bash
python 05_panel_regression.py
python 06_robustness_checks.py
python 07_predictive_model.py
python 08_additional_diagnostics.py
```

The data-construction scripts (`01`–`04`) document the original construction workflow and may require the raw source files used in the study.

## Data and methodology

See `data_sources.md` for data construction and source documentation and `methodology.md` for the econometric, forecasting, and scenario-analysis methodology.

## License

MIT License. See `LICENSE`.
