import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS, RandomEffects
import statsmodels.api as sm

df = pd.read_csv("../data/eic_panel_dataset.csv")
df['period'] = pd.PeriodIndex(df['year'].astype(str), freq="Y").to_timestamp()
df = df.set_index(["entity", "period"])

df['log_output'] = np.log(df['gross_output_billions'])
df['log_imports'] = np.log(df['imports_usd'])
df['log_exports'] = np.log(df['exports_usd'])
df['employment'] = df['employment_thousands']
df['labor_productivity'] = df['labor_productivity_index']
df['hhi_std'] = (df['hhi'] - df['hhi'].mean()) / df['hhi'].std()

y = df['log_output']
X = df[['employment', 'labor_productivity', 'log_imports', 'log_exports', 'hhi_std']]

re = RandomEffects(y, sm.add_constant(X)).fit(cov_type="clustered", cluster_entity=True)
fe2 = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)

print("="*90)
print("FULL MODEL WITH SUPPLIER CONCENTRATION (HHI) — REAL DATA, N=209")
print("="*90)
print(fe2)

common = [c for c in fe2.params.index if c in re.params.index]
b_fe, b_re = fe2.params[common].values, re.params[common].values
v_fe, v_re = fe2.cov.loc[common, common].values, re.cov.loc[common, common].values
diff = b_fe - b_re
cov_diff = v_fe - v_re
try:
    stat = diff.T @ np.linalg.inv(cov_diff) @ diff
    from scipy import stats as sstats
    pval = 1 - sstats.chi2.cdf(stat, df=len(common))
    print(f"\nHausman-type test, chi2({len(common)}) = {stat:.2f}, p-value = {pval:.4f}")
except np.linalg.LinAlgError:
    print("\nHausman covariance not invertible with clustered SEs.")

print("\nR-squared (within):", round(fe2.rsquared_within, 4))
print("F-statistic:", round(fe2.f_statistic.stat, 2), " p-value:", fe2.f_statistic.pval)
