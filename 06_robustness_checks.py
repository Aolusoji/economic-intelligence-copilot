import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
import statsmodels.api as sm
from scipy import stats as sstats

df = pd.read_csv('../data/eic_panel_dataset.csv')
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
Xc = sm.add_constant(X)

# --- Specification 1: Pooled OLS ---
pooled = PooledOLS(y, Xc).fit(cov_type="unadjusted")

# --- Specification 2: Random Effects ---
re_unclustered = RandomEffects(y, Xc).fit(cov_type="unadjusted")
re_clustered = RandomEffects(y, Xc).fit(cov_type="clustered", cluster_entity=True)

# --- Specification 3: Two-way FE without HHI (prior model) ---
X_noHHI = df[['employment', 'labor_productivity', 'log_imports', 'log_exports']]
fe_noHHI = PanelOLS(y, X_noHHI, entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)

# --- Specification 4: Two-way FE WITH HHI (main/preferred model) ---
fe_main_unclustered = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type="unadjusted")
fe_main_clustered = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)

# --- Specification 5: Entity FE only (one-way, no time effects) ---
fe_entity_only = PanelOLS(y, X, entity_effects=True, time_effects=False, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)

print("="*95)
print("PROPER HAUSMAN TEST (using UNCLUSTERED covariances, classical implementation)")
print("="*95)
common = [c for c in fe_main_unclustered.params.index if c in re_unclustered.params.index]
b_fe = fe_main_unclustered.params[common].values
b_re = re_unclustered.params[common].values
v_fe = fe_main_unclustered.cov.loc[common, common].values
v_re = re_unclustered.cov.loc[common, common].values
diff = b_fe - b_re
cov_diff = v_fe - v_re
try:
    stat = float(diff.T @ np.linalg.inv(cov_diff) @ diff)
    pval = 1 - sstats.chi2.cdf(stat, df=len(common))
    print(f"Hausman chi2({len(common)}) = {stat:.3f}, p-value = {pval:.4f}")
    hausman_valid = True
except np.linalg.LinAlgError:
    print("Still not invertible - covariance difference not positive definite")
    hausman_valid = False
    # fallback: use pseudo-inverse
    stat = float(diff.T @ np.linalg.pinv(cov_diff) @ diff)
    pval = 1 - sstats.chi2.cdf(stat, df=len(common))
    print(f"(Moore-Penrose pseudo-inverse fallback) Hausman chi2({len(common)}) = {stat:.3f}, p-value = {pval:.4f}")

print("\n" + "="*95)
print("ALTERNATIVE MODEL SPECIFICATIONS COMPARISON")
print("="*95)
specs = {
    "Pooled OLS": pooled,
    "Random Effects": re_clustered,
    "Two-Way FE (no HHI)": fe_noHHI,
    "Two-Way FE (main, with HHI)": fe_main_clustered,
    "Entity FE only (no time FE)": fe_entity_only,
}
for name, m in specs.items():
    print(f"\n--- {name} ---")
    print(f"  R-squared: {m.rsquared:.4f}" + (f" | within: {m.rsquared_within:.4f}" if hasattr(m, 'rsquared_within') else ""))
    for var in ['employment','labor_productivity','log_imports','log_exports','hhi_std']:
        if var in m.params.index:
            print(f"  {var:20s} coef={m.params[var]:.4f}  p={m.pvalues[var]:.3f}")
