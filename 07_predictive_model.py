import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# ============================================================================
# GENUINE EX-ANTE FORECASTING (rolling-origin, lagged predictors)
#
# CORRECTION NOTE: an earlier version of this script used the *realized*
# values of employment, labor productivity, imports, exports, and HHI for the
# test-period years to predict test-period output. That is conditional
# prediction / nowcasting, not genuine forecasting, because those predictor
# values would not actually have been known in advance of the forecast date.
#
# This version fixes that: every predictor is lagged by one year (X at time
# t-1 predicts Y at time t), and evaluation uses a rolling-origin design --
# the model is retrained at each origin year using only data available up to
# that point, and used to forecast exactly one year ahead. This means every
# single forecast in the evaluation uses only information that would
# genuinely have been available at the time the forecast was made.
#
# HHI standardization is also now computed using only the training window at
# each origin (not the full 2015-2025 sample), to avoid a second, smaller
# form of the same look-ahead problem.
# ============================================================================

DATA_PATH = '../data/eic_panel_dataset.csv'
ORIGIN_YEARS = [2021, 2022, 2023, 2024]  # each origin year forecasts origin+1

df = pd.read_csv(DATA_PATH).sort_values(['entity', 'year']).reset_index(drop=True)
df['log_output'] = np.log(df['gross_output_billions'])
df['log_imports'] = np.log(df['imports_usd'])
df['log_exports'] = np.log(df['exports_usd'])

df = df.sort_values(['entity', 'year'])
lag_cols = ['employment_thousands', 'labor_productivity_index', 'log_imports', 'log_exports', 'hhi']
for col in lag_cols:
    df[f'{col}_lag1'] = df.groupby('entity')[col].shift(1)

feat_cols = [f'{c}_lag1' for c in lag_cols] + ['year']

def mape(actual, pred):
    return np.mean(np.abs((actual - pred) / actual)) * 100

def rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred) ** 2))

def mae(actual, pred):
    return np.mean(np.abs(actual - pred))

results_by_model = {'Naive (persistence)': [], 'Linear (entity FE + trend)': [], 'Gradient Boosting': [], 'Hybrid': []}
per_origin_records = []

for origin in ORIGIN_YEARS:
    forecast_year = origin + 1

    train = df[(df.year <= origin) & (df[feat_cols].notna().all(axis=1))].copy()
    test = df[(df.year == forecast_year) & (df[feat_cols].notna().all(axis=1))].copy()

    if len(train) == 0 or len(test) == 0:
        continue

    hhi_mean, hhi_std = train['hhi_lag1'].mean(), train['hhi_lag1'].std()
    train['hhi_std_lag1'] = (train['hhi_lag1'] - hhi_mean) / hhi_std
    test['hhi_std_lag1'] = (test['hhi_lag1'] - hhi_mean) / hhi_std

    model_feats = ['employment_thousands_lag1', 'labor_productivity_index_lag1',
                   'log_imports_lag1', 'log_exports_lag1', 'hhi_std_lag1', 'year']

    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    enc.fit(train[['entity']])
    ent_train = enc.transform(train[['entity']])
    ent_test = enc.transform(test[['entity']])

    X_train = np.hstack([train[model_feats].values, ent_train])
    X_test = np.hstack([test[model_feats].values, ent_test])
    y_train = train['log_output'].values
    y_test = test['log_output'].values

    last_val = df[df.year == origin].set_index('entity')['log_output']
    naive_pred = test['entity'].map(last_val).values

    lin = LinearRegression().fit(X_train, y_train)
    lin_pred = lin.predict(X_test)

    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    gbr.fit(X_train, y_train)
    gbr_pred = gbr.predict(X_test)

    hybrid_pred = (lin_pred + gbr_pred) / 2

    preds = {'Naive (persistence)': naive_pred, 'Linear (entity FE + trend)': lin_pred,
             'Gradient Boosting': gbr_pred, 'Hybrid': hybrid_pred}

    for name, pred in preds.items():
        actual = np.exp(y_test)
        predicted = np.exp(pred)
        results_by_model[name].append({
            'origin': origin, 'forecast_year': forecast_year, 'n': len(test),
            'mape': mape(actual, predicted), 'rmse': rmse(actual, predicted), 'mae': mae(actual, predicted)
        })

    per_origin_records.append(forecast_year)

print("="*78)
print("GENUINE EX-ANTE ROLLING-ORIGIN FORECAST EVALUATION")
print("Each origin year's model uses ONLY lagged (t-1) predictors and data")
print("available up to and including the origin year. No realized future")
print("predictor values are used at any point.")
print("="*78)
print(f"\nForecast years evaluated: {per_origin_records}\n")

summary_rows = []
for name, records in results_by_model.items():
    avg_mape = np.mean([r['mape'] for r in records])
    avg_rmse = np.mean([r['rmse'] for r in records])
    avg_mae = np.mean([r['mae'] for r in records])
    summary_rows.append((name, avg_mape, avg_rmse, avg_mae))
    print(f"{name:32s}  avg MAPE={avg_mape:6.2f}%   avg RMSE=${avg_rmse:7.2f}B   avg MAE=${avg_mae:7.2f}B")
    for r in records:
        print(f"    origin {r['origin']} -> forecast {r['forecast_year']}: MAPE={r['mape']:.2f}%  RMSE=${r['rmse']:.2f}B  n={r['n']}")

print("\n" + "="*78)
print("Per-origin-year comparison table (MAPE %)")
print("="*78)
header = "Model".ljust(32) + "".join([f"{y:>10}" for y in [o+1 for o in ORIGIN_YEARS]]) + "     Average"
print(header)
for name, records in results_by_model.items():
    row = name.ljust(32)
    rec_by_year = {r['forecast_year']: r['mape'] for r in records}
    for y in [o+1 for o in ORIGIN_YEARS]:
        row += f"{rec_by_year.get(y, float('nan')):>9.2f}%"
    row += f"    {np.mean([r['mape'] for r in records]):>7.2f}%"
    print(row)

pd.DataFrame(summary_rows, columns=['model', 'avg_mape', 'avg_rmse', 'avg_mae']).to_csv('rolling_forecast_summary.csv', index=False)
