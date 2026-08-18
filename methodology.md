# Methodology

## 1. Panel data model specification

The core econometric model is a two-way (entity- and time-) fixed effects panel regression:

```
Y_it = α_i + λ_t + β1·Employment_it + β2·LaborProductivity_it
       + β3·log(Imports)_it + β4·log(Exports)_it + β5·HHI_std_it + ε_it
```

where `i` indexes manufacturing subsector, `t` indexes year, `α_i` is an entity fixed effect, `λ_t` is a time fixed effect, and `Y_it` is log(Gross Output).

**Model selection**: a Hausman specification test (Hausman, 1978) is used to choose between fixed and random effects, run on a **like-for-like basis** — both the Random Effects and Fixed Effects models include the same year-effect structure, since comparing models with different time structures produces an invalid test statistic. This study's Hausman test is supplemented with a **Mundlak (correlated random effects) test**, which tests the same underlying question (whether individual effects correlate with regressors) via a complementary approach: entity-level means of each regressor are added to the Random Effects model and tested for joint significance. Because key regressors (particularly import-origin concentration) are plausibly correlated with persistent, unobserved industry characteristics, fixed effects is adopted as the preferred specification, based on the balance of evidence from both tests rather than either test in isolation. Standard errors are clustered at the entity level (Petersen, 2009); because this panel has a modest number of entity clusters (19), a **wild-cluster bootstrap** is also reported and treated as the more credible inferential statement than the asymptotic cluster-robust standard errors alone.

**Import-Origin Concentration Index**: the concentration variable in this framework is named precisely to describe what it measures — the concentration of a subsector's import value among its countries of origin (a macro-level, country-of-origin proxy), not firm-level supplier concentration, which would require transaction-level data this framework does not have access to.

## 2. Supplier Concentration Indicator

Computed using the Herfindahl-Hirschman methodology (as formalized in U.S. antitrust merger-review practice; U.S. DOJ & FTC, 2023): for each subsector-year, the percentage import share of every country of origin is squared and summed. Higher values indicate import dependence concentrated among fewer countries.

## 3. Genuine ex-ante predictive modeling

**Correction note**: an earlier version of this framework's forecasting exercise used the *realized* values of employment, labor productivity, imports, exports, and concentration for the forecast-period years as model predictors. Because those values would not genuinely have been known in advance of a real forecast, that design was conditional prediction (nowcasting), not ex-ante forecasting. It has been corrected as described below.

The framework now uses a **rolling-origin design with one-year-lagged predictors**: every predictor is lagged by one year (X at time t-1 predicts Y at time t), and the model is retrained at each of several origin years using only data available up to that point, then used to forecast exactly one year ahead. This is repeated across multiple origins to produce the reported forecast-accuracy figures, ensuring every single forecast uses only information that would genuinely have been available at the time it was made. Four approaches are compared:

- **Naive persistence**: carries the prior-year value forward unchanged.
- **Linear (entity FE + year trend)**: a linear model on lagged predictors, entity fixed effects, and a year trend.
- **Gradient Boosting** (Friedman, 2001): a nonlinear machine-learning ensemble, also on lagged predictors.
- **Hybrid**: the average of the linear and gradient-boosting forecasts.

Forecast accuracy is evaluated with MAPE, RMSE, and MAE, averaged across rolling origins and explicitly benchmarked against the naive forecast. Reporting cases where the naive benchmark is competitive or superior is treated as a required part of the methodology, not an optional robustness check.

## 4. Model-implied scenario analysis

Using the real fitted coefficients from the panel regression, the predicted percentage change in output associated with a hypothetical change in imports, exports, or supplier concentration is computed as:

```
predicted_change = (1 + shock)^beta - 1
```

for the trade-volume variables (log-linear elasticity), and

```
predicted_change = e^(beta_hhi * sd_shock) - 1
```

for concentration (entered as a standardized level). These are model-implied associations under the fitted specification, not causal forecasts of any specific real-world policy or disruption.

**Known limitation**: because the model includes no interaction terms between trade exposure and supplier concentration, it implies the same percentage output response to a given shock magnitude across every subsector. An extended specification with a trade × concentration interaction term is identified as a priority next step, since it would let the model estimate differential vulnerability by concentration level — directly testing whether highly concentrated subsectors (e.g., Wood products, HHI ~1,962 in 2025) are more vulnerable to a trade shock than diversified ones (e.g., Food and beverage and tobacco products, HHI ~837).

## 5. Robustness checks

Five specifications are compared on the identical panel: Pooled OLS, Random Effects, Two-Way Fixed Effects without the concentration variable, the main Two-Way Fixed Effects model, and Entity-Fixed-Effects-only. This tests whether findings are an artifact of a single modeling choice rather than a robust pattern in the data.

## References

- Anderson, J. E. (1979). A theoretical foundation for the gravity equation. *American Economic Review*, 69(1), 106-116.
- Arellano, M., & Bond, S. (1991). Some tests of specification for panel data. *Review of Economic Studies*, 58(2), 277-297.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.
- Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica*, 46(6), 1251-1271.
- Petersen, M. A. (2009). Estimating standard errors in finance panel data sets. *Review of Financial Studies*, 22(1), 435-480.
- Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 48(1), 1-48.
- U.S. Department of Justice & Federal Trade Commission. (2023). Merger Guidelines, §2.1.
