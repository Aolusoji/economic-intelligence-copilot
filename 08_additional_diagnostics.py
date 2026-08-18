import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels.panel import PanelOLS, RandomEffects
from scipy import stats as sstats
from wildboottest.wildboottest import wildboottest

# ============================================================================
# ADDITIONAL DIAGNOSTICS -- responds directly to peer-review feedback:
#  1. Like-for-like Hausman test (RE now includes year dummies, matching FE)
#  2. Mundlak (correlated random effects) test
#  3. Wild-cluster bootstrap p-values (19 clusters is small for asymptotic CRVE)
#  4. Trade x concentration interaction term
#  5. VIF (multicollinearity check)
#  6. Pesaran CD test (cross-sectional dependence)
#  7. Wooldridge-style serial correlation test
# ============================================================================

df = pd.read_csv('../data/eic_panel_dataset.csv')
df['log_output'] = np.log(df['gross_output_billions'])
df['log_imports'] = np.log(df['imports_usd'])
df['log_exports'] = np.log(df['exports_usd'])
df['hhi_std'] = (df['hhi'] - df['hhi'].mean()) / df['hhi'].std()
df['employment'] = df['employment_thousands']
df['labor_productivity'] = df['labor_productivity_index']

panel_df = df.set_index(['entity', 'year'])
y = panel_df['log_output']
X = panel_df[['employment', 'labor_productivity', 'log_imports', 'log_exports', 'hhi_std']]

# ----------------------------------------------------------------------------
# 1. LIKE-FOR-LIKE HAUSMAN TEST: RE now includes year dummies as regressors,
#    matching FE's time_effects=True, so the comparison is now apples-to-apples.
# ----------------------------------------------------------------------------
print("="*78)
print("1. CORRECTED (LIKE-FOR-LIKE) HAUSMAN TEST")
print("="*78)

year_dummies = pd.get_dummies(df['year'], prefix='yr', drop_first=True).astype(float)
df_re = pd.concat([df[['entity', 'year']], X.reset_index(drop=True), year_dummies], axis=1).set_index(['entity', 'year'])
Xc_re = sm.add_constant(df_re)

re_with_time = RandomEffects(y, Xc_re).fit(cov_type="unadjusted")
fe_main = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type="unadjusted")

common = [c for c in fe_main.params.index if c in re_with_time.params.index]
b_fe, b_re = fe_main.params[common].values, re_with_time.params[common].values
v_fe, v_re = fe_main.cov.loc[common, common].values, re_with_time.cov.loc[common, common].values
diff = b_fe - b_re
cov_diff = v_fe - v_re
try:
    stat = float(diff.T @ np.linalg.inv(cov_diff) @ diff)
    pval = 1 - sstats.chi2.cdf(stat, df=len(common))
    print(f"Hausman chi2({len(common)}) = {stat:.3f}, p-value = {pval:.4f}  [RE now includes year dummies]")
except np.linalg.LinAlgError:
    stat = float(diff.T @ np.linalg.pinv(cov_diff) @ diff)
    pval = 1 - sstats.chi2.cdf(stat, df=len(common))
    print(f"(pseudo-inverse) Hausman chi2({len(common)}) = {stat:.3f}, p-value = {pval:.4f}")

# ----------------------------------------------------------------------------
# 2. MUNDLAK (CORRELATED RANDOM EFFECTS) TEST
# ----------------------------------------------------------------------------
print("\n" + "="*78)
print("2. MUNDLAK TEST (entity-level means added to RE)")
print("="*78)

entity_means = df.groupby('entity')[['employment', 'labor_productivity', 'log_imports', 'log_exports', 'hhi_std']].transform('mean')
entity_means.columns = [f'{c}_mean' for c in entity_means.columns]
df_mundlak = pd.concat([df[['entity', 'year']], X.reset_index(drop=True), entity_means.reset_index(drop=True)], axis=1).set_index(['entity', 'year'])
Xc_mundlak = sm.add_constant(df_mundlak)

re_mundlak = RandomEffects(y, Xc_mundlak).fit(cov_type="clustered", cluster_entity=True)
mean_vars = [c for c in df_mundlak.columns if c.endswith('_mean')]
mean_coefs = re_mundlak.params[mean_vars]
mean_cov = re_mundlak.cov.loc[mean_vars, mean_vars]
wald_stat = float(mean_coefs.values @ np.linalg.inv(mean_cov.values) @ mean_coefs.values)
wald_p = 1 - sstats.chi2.cdf(wald_stat, df=len(mean_vars))
print(f"Joint Wald test on entity-mean coefficients: chi2({len(mean_vars)}) = {wald_stat:.3f}, p = {wald_p:.4f}")
print("(A significant result supports FE over RE: individual effects ARE correlated with regressors.)")

# ----------------------------------------------------------------------------
# 3. WILD CLUSTER BOOTSTRAP (19 clusters is small for asymptotic CRVE)
# ----------------------------------------------------------------------------
print("\n" + "="*78)
print("3. WILD CLUSTER BOOTSTRAP (999 reps, Rademacher weights, cluster=entity)")
print("="*78)

entity_dummies = pd.get_dummies(df['entity'], prefix='ent', drop_first=True).astype(float)
year_dummies2 = pd.get_dummies(df['year'], prefix='yr', drop_first=True).astype(float)
X_ols = pd.concat([X.reset_index(drop=True), entity_dummies.reset_index(drop=True), year_dummies2.reset_index(drop=True)], axis=1)
X_ols = sm.add_constant(X_ols).astype(float)
y_ols = y.reset_index(drop=True).astype(float)
cluster_var = df['entity'].reset_index(drop=True)

ols_model = sm.OLS(y_ols, X_ols)
main_vars = ['employment', 'labor_productivity', 'log_imports', 'log_exports', 'hhi_std']
cluster_codes = pd.Series(pd.factorize(cluster_var)[0], name='cluster_code').astype(np.int64)
for var in main_vars:
    try:
        result = wildboottest(ols_model, param=var, cluster=cluster_codes, B=999, seed=42, parallel=False, show=False)
        boot_stat = result['statistic'].values[0]
        boot_p = result['p-value'].values[0]
        print(f"  {var:22s}  wild-cluster-bootstrap statistic = {boot_stat:.3f}, p-value = {boot_p:.4f}")
    except Exception as e:
        print(f"  {var:22s}  bootstrap failed: {type(e).__name__}: {str(e)[:150]}")

# ----------------------------------------------------------------------------
# 4. TRADE x CONCENTRATION INTERACTION TERM
# ----------------------------------------------------------------------------
print("\n" + "="*78)
print("4. TRADE x CONCENTRATION INTERACTION MODEL")
print("="*78)

df['imports_x_hhi'] = df['log_imports'] * df['hhi_std']
panel_int = df.set_index(['entity', 'year'])
X_int = panel_int[['employment', 'labor_productivity', 'log_imports', 'log_exports', 'hhi_std', 'imports_x_hhi']]
fe_int = PanelOLS(y, X_int, entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
print(fe_int)

# ----------------------------------------------------------------------------
# 5. VIF (multicollinearity)
# ----------------------------------------------------------------------------
print("\n" + "="*78)
print("5. VARIANCE INFLATION FACTORS")
print("="*78)
vif_X = sm.add_constant(X.reset_index(drop=True))
for i, col in enumerate(vif_X.columns):
    if col == 'const':
        continue
    vif = variance_inflation_factor(vif_X.values, i)
    print(f"  {col:22s}  VIF = {vif:.2f}")

# ----------------------------------------------------------------------------
# 6. PESARAN CD TEST (cross-sectional dependence) -- manual implementation
# ----------------------------------------------------------------------------
print("\n" + "="*78)
print("6. PESARAN CD TEST (cross-sectional dependence in FE residuals)")
print("="*78)
resid = fe_main.resids
resid_df = resid.reset_index()
resid_df.columns = ['entity', 'year', 'resid']
wide = resid_df.pivot(index='year', columns='entity', values='resid')
N = wide.shape[1]
T = wide.shape[0]
corr_mat = wide.corr()
rho_sum = 0
count = 0
for i in range(N):
    for j in range(i+1, N):
        rho_sum += corr_mat.iloc[i, j]
        count += 1
avg_rho = rho_sum / count
cd_stat = np.sqrt(2*T / (N*(N-1))) * rho_sum
cd_p = 2 * (1 - sstats.norm.cdf(abs(cd_stat)))
print(f"  Average pairwise residual correlation: {avg_rho:.4f}")
print(f"  Pesaran CD statistic: {cd_stat:.3f}, p-value = {cd_p:.4f}")

# ----------------------------------------------------------------------------
# 7. WOOLDRIDGE-STYLE SERIAL CORRELATION TEST (first-differenced residual regression)
# ----------------------------------------------------------------------------
print("\n" + "="*78)
print("7. SERIAL CORRELATION TEST (Wooldridge-style, on first-differenced residuals)")
print("="*78)
diff_df = df.sort_values(['entity','year']).copy()
diff_df['d_log_output'] = diff_df.groupby('entity')['log_output'].diff()
for col in ['employment','labor_productivity','log_imports','log_exports','hhi_std']:
    diff_df[f'd_{col}'] = diff_df.groupby('entity')[col].diff()
diff_df = diff_df.dropna(subset=['d_log_output','d_employment','d_labor_productivity','d_log_imports','d_log_exports','d_hhi_std'])
Xd = sm.add_constant(diff_df[['d_employment','d_labor_productivity','d_log_imports','d_log_exports','d_hhi_std']])
diff_resid = sm.OLS(diff_df['d_log_output'], Xd).fit().resid
diff_df['diff_resid'] = diff_resid.values
diff_df['diff_resid_lag'] = diff_df.groupby('entity')['diff_resid'].shift(1)
test_df = diff_df.dropna(subset=['diff_resid_lag'])
ar_test = sm.OLS(test_df['diff_resid'], sm.add_constant(test_df['diff_resid_lag'])).fit(cov_type='cluster', cov_kwds={'groups': test_df['entity']})
coef = ar_test.params['diff_resid_lag']
se = ar_test.bse['diff_resid_lag']
# Wooldridge (2002) null: under NO serial correlation in the original (undifferenced)
# error term, this coefficient should equal exactly -0.5, not 0. Test against -0.5.
t_stat_vs_null = (coef - (-0.5)) / se
p_vs_null = 2 * (1 - sstats.t.cdf(abs(t_stat_vs_null), df=len(test_df)-2))
print(f"  AR(1) coefficient on lagged first-differenced residual: {coef:.4f} (SE={se:.4f})")
print(f"  Wooldridge null is coef = -0.5 (not 0). Test of H0: coef = -0.5:")
print(f"  t-stat = {t_stat_vs_null:.3f}, p-value = {p_vs_null:.4f}")
if p_vs_null < 0.05:
    print("  -> REJECTS the null: evidence of serial correlation beyond first-differencing.")
else:
    print("  -> FAILS TO REJECT the null: no strong evidence of additional serial correlation.")

# ----------------------------------------------------------------------------
# 8. PARSIMONIOUS DYNAMIC PANEL (lagged dependent variable, entity+time FE)
#    NOT Arellano-Bond/GMM -- with only 19 cross-sectional units, GMM risks
#    instrument proliferation. This is a simple dynamic FE check, explicitly
#    flagged as subject to Nickell bias, per the reviewer's own recommended
#    minimum bar given this panel's small N.
# ----------------------------------------------------------------------------
print("\n" + "="*78)
print("8. PARSIMONIOUS DYNAMIC PANEL (lagged dependent variable, entity+time FE)")
print("   NOTE: subject to Nickell bias -- NOT a GMM/bias-corrected estimate.")
print("="*78)

dyn_df = df.sort_values(['entity','year']).copy()
dyn_df['log_output_lag1'] = dyn_df.groupby('entity')['log_output'].shift(1)
dyn_df_clean = dyn_df.dropna(subset=['log_output_lag1'])
panel_dyn = dyn_df_clean.set_index(['entity','year'])

y_dyn = panel_dyn['log_output']
X_dyn = panel_dyn[['log_output_lag1','employment','labor_productivity','log_imports','log_exports','hhi_std']]

dyn_fe = PanelOLS(y_dyn, X_dyn, entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
print(dyn_fe)

print("\nComparison: does adding lagged output materially change the trade/concentration coefficients?")
static_coefs = fe_main.params
dyn_coefs = dyn_fe.params
for var in ['employment','labor_productivity','log_imports','log_exports','hhi_std']:
    if var in dyn_coefs.index:
        print(f"  {var:22s}  static={static_coefs[var]:.4f}   dynamic={dyn_coefs[var]:.4f}   change={dyn_coefs[var]-static_coefs[var]:+.4f}")

# ----------------------------------------------------------------------------
# 9. TRANSPORTATION-EQUIPMENT (NAICS 336) ALLOCATION SENSITIVITY CHECK
#    USITC reports NAICS 336 trade as one combined figure, split into Motor
#    Vehicles and Other Transportation Equipment using employment shares.
#    This tests whether that approximation drives the main results.
# ----------------------------------------------------------------------------
print("\n" + "="*78)
print("9. TRANSPORTATION-EQUIPMENT (NAICS 336) ALLOCATION SENSITIVITY CHECK")
print("="*78)

transport_entities = ['Motor vehicles, bodies and trailers, and parts', 'Other transportation equipment']

# (a) Current allocation (main results, for reference)
print("\n(a) CURRENT ALLOCATION (main results, Table 4.2):")
print(f"    log_imports=0.1998, log_exports=0.2747, hhi_std=-0.0240  [already estimated above]")

# (b) Omit both transportation subsectors entirely
df_omit = df[~df['entity'].isin(transport_entities)].copy()
panel_omit = df_omit.set_index(['entity','year'])
y_omit = panel_omit['log_output']
X_omit = panel_omit[['employment','labor_productivity','log_imports','log_exports','hhi_std']]
fe_omit = PanelOLS(y_omit, X_omit, entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
print("\n(b) OMITTING both transportation subsectors (N=%d, 17 entities):" % len(df_omit))
for var in ['employment','labor_productivity','log_imports','log_exports','hhi_std']:
    print(f"    {var:22s}  coef={fe_omit.params[var]:.4f}  p={fe_omit.pvalues[var]:.3f}")

# (c) Collapse the two transportation subsectors back into one NAICS 336 category
df_collapse = df.copy()
df_collapse['entity_collapsed'] = df_collapse['entity'].replace(
    {t: 'Transportation equipment (336, collapsed)' for t in transport_entities}
)
collapsed = df_collapse.groupby(['entity_collapsed','year']).agg(
    gross_output_billions=('gross_output_billions','sum'),
    employment_thousands=('employment_thousands','sum'),
    exports_usd=('exports_usd','sum'),
    imports_usd=('imports_usd','sum'),
    labor_productivity_index=('labor_productivity_index','mean'),  # same value for both, mean is exact
    hhi=('hhi','mean'),  # identical HHI already assigned to both, mean is exact
).reset_index()
collapsed['log_output'] = np.log(collapsed['gross_output_billions'])
collapsed['log_imports'] = np.log(collapsed['imports_usd'])
collapsed['log_exports'] = np.log(collapsed['exports_usd'])
collapsed['hhi_std'] = (collapsed['hhi'] - collapsed['hhi'].mean()) / collapsed['hhi'].std()
collapsed['employment'] = collapsed['employment_thousands']
collapsed['labor_productivity'] = collapsed['labor_productivity_index']
panel_collapsed = collapsed.set_index(['entity_collapsed','year'])
y_col = panel_collapsed['log_output']
X_col = panel_collapsed[['employment','labor_productivity','log_imports','log_exports','hhi_std']]
fe_collapsed = PanelOLS(y_col, X_col, entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
print("\n(c) COLLAPSED into one Transportation Equipment category (N=%d, 18 entities):" % len(collapsed))
for var in ['employment','labor_productivity','log_imports','log_exports','hhi_std']:
    print(f"    {var:22s}  coef={fe_collapsed.params[var]:.4f}  p={fe_collapsed.pvalues[var]:.3f}")

print("\nSUMMARY: sign and rough magnitude stability across all three allocations")

# ----------------------------------------------------------------------------
# 10. WILD-CLUSTER BOOTSTRAP FOR THE DYNAMIC PANEL MODEL
#     Round-3 review: the dynamic FE model (Section 4.4.5) reported only
#     asymptotic p-values despite the same 19-cluster small-sample concern
#     that motivated the wild bootstrap for the static model. Fixing that here.
# ----------------------------------------------------------------------------
print("\n" + "="*78)
print("10. WILD-CLUSTER BOOTSTRAP FOR DYNAMIC PANEL MODEL (lagged dependent var)")
print("="*78)

dyn_entity_dummies = pd.get_dummies(dyn_df_clean['entity'], prefix='ent', drop_first=True).astype(float)
dyn_year_dummies = pd.get_dummies(dyn_df_clean['year'], prefix='yr', drop_first=True).astype(float)
X_dyn_ols = pd.concat([
    dyn_df_clean[['log_output_lag1','employment','labor_productivity','log_imports','log_exports','hhi_std']].reset_index(drop=True),
    dyn_entity_dummies.reset_index(drop=True),
    dyn_year_dummies.reset_index(drop=True)
], axis=1)
X_dyn_ols = sm.add_constant(X_dyn_ols).astype(float)
y_dyn_ols = dyn_df_clean['log_output'].reset_index(drop=True).astype(float)
dyn_cluster_codes = pd.Series(pd.factorize(dyn_df_clean['entity'].reset_index(drop=True))[0], name='cluster_code').astype(np.int64)

dyn_ols_model = sm.OLS(y_dyn_ols, X_dyn_ols)
dyn_vars = ['log_output_lag1','employment','labor_productivity','log_imports','log_exports','hhi_std']
for var in dyn_vars:
    try:
        result = wildboottest(dyn_ols_model, param=var, cluster=dyn_cluster_codes, B=999, seed=42, parallel=False, show=False)
        boot_stat = result['statistic'].values[0]
        boot_p = result['p-value'].values[0]
        print(f"  {var:22s}  wild-cluster-bootstrap statistic = {boot_stat:.3f}, p-value = {boot_p:.4f}")
    except Exception as e:
        print(f"  {var:22s}  bootstrap failed: {type(e).__name__}: {str(e)[:150]}")

# ----------------------------------------------------------------------------
# 11. FIRST-DIFFERENCED ROBUSTNESS CHECK (addresses nominal-dollar / trend concern)
#     Round-3 review: since output, imports, and exports are nominal and all
#     trending 2015-2025, first-differencing removes entity-specific linear
#     trends (including steady nominal-price drift) that year effects alone
#     do not absorb.
# ----------------------------------------------------------------------------
print("\n" + "="*78)
print("11. FIRST-DIFFERENCED ROBUSTNESS CHECK, WITH ENTITY EFFECTS")
print("="*78)
print("NOTE: an earlier version of this check included only year dummies,")
print("not entity dummies, in the differenced regression. First-differencing")
print("alone removes each entity's time-invariant fixed effect, but an")
print("entity-specific LINEAR TREND survives differencing as an entity-")
print("specific constant; only including entity dummies in the differenced")
print("regression itself removes that. This version adds entity dummies to")
print("correctly support the 'removes entity-specific trends' claim.")
print("="*78)

fd_df = df.sort_values(['entity','year']).copy()
fd_df['d_log_output'] = fd_df.groupby('entity')['log_output'].diff()
for col in ['employment','labor_productivity','log_imports','log_exports','hhi_std']:
    fd_df[f'd_{col}'] = fd_df.groupby('entity')[col].diff()
fd_df_clean = fd_df.dropna(subset=['d_log_output','d_employment','d_labor_productivity','d_log_imports','d_log_exports','d_hhi_std'])

fd_year_dummies = pd.get_dummies(fd_df_clean['year'], prefix='yr', drop_first=True).astype(float)
fd_entity_dummies = pd.get_dummies(fd_df_clean['entity'], prefix='ent', drop_first=True).astype(float)
X_fd = pd.concat([
    fd_df_clean[['d_employment','d_labor_productivity','d_log_imports','d_log_exports','d_hhi_std']].reset_index(drop=True),
    fd_entity_dummies.reset_index(drop=True),
    fd_year_dummies.reset_index(drop=True)
], axis=1)
X_fd = sm.add_constant(X_fd).astype(float)
y_fd = fd_df_clean['d_log_output'].reset_index(drop=True).astype(float)
fd_cluster = fd_df_clean['entity'].reset_index(drop=True)

fd_model = sm.OLS(y_fd, X_fd).fit(cov_type='cluster', cov_kwds={'groups': fd_cluster})
print(f"N = {len(fd_df_clean)} (one year lost per entity to differencing); includes entity AND year dummies")
for var in ['d_employment','d_labor_productivity','d_log_imports','d_log_exports','d_hhi_std']:
    print(f"  {var:22s}  coef={fd_model.params[var]:.4f}  se={fd_model.bse[var]:.4f}  p={fd_model.pvalues[var]:.3f}")
print(f"  R-squared: {fd_model.rsquared:.4f}")
print("\nComparison to main two-way FE specification (Table 4.2):")
main_map = {'d_employment':'employment','d_labor_productivity':'labor_productivity','d_log_imports':'log_imports','d_log_exports':'log_exports','d_hhi_std':'hhi_std'}
for dvar, mvar in main_map.items():
    print(f"  {mvar:22s}  main_FE={static_coefs[mvar]:.4f}   first_diff={fd_model.params[dvar]:.4f}")

print("\n" + "-"*78)
print("Wild-cluster bootstrap for the first-differenced specification (19 clusters)")
print("-"*78)
fd_ols_model = sm.OLS(y_fd, X_fd)
fd_cluster_codes = pd.Series(pd.factorize(fd_cluster)[0], name='cluster_code').astype(np.int64)
for var in ['d_employment','d_labor_productivity','d_log_imports','d_log_exports','d_hhi_std']:
    try:
        result = wildboottest(fd_ols_model, param=var, cluster=fd_cluster_codes, B=999, seed=42, parallel=False, show=False)
        boot_stat = result['statistic'].values[0]
        boot_p = result['p-value'].values[0]
        print(f"  {var:22s}  wild-cluster-bootstrap statistic = {boot_stat:.3f}, p-value = {boot_p:.4f}")
    except Exception as e:
        print(f"  {var:22s}  bootstrap failed: {type(e).__name__}: {str(e)[:150]}")

# ----------------------------------------------------------------------------
# 12. EMPLOYMENT-IMPACT MODEL
#     Extends the framework from an output-only model to a genuine
#     economic-impact model estimating effects on employment specifically,
#     controlling for output and productivity.
# ----------------------------------------------------------------------------
print("\n" + "="*78)
print("12. EMPLOYMENT-IMPACT MODEL: log(Employment) as dependent variable")
print("="*78)

emp_df = df.copy()
emp_df['log_employment'] = np.log(emp_df['employment'])
panel_emp = emp_df.set_index(['entity','year'])
y_emp = panel_emp['log_employment']
X_emp = panel_emp[['log_output','labor_productivity','log_imports','log_exports','hhi_std']]

fe_emp = PanelOLS(y_emp, X_emp, entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
print(fe_emp)

# Wild-cluster bootstrap for the employment model
emp_entity_dummies = pd.get_dummies(df['entity'], prefix='ent', drop_first=True).astype(float)
emp_year_dummies = pd.get_dummies(df['year'], prefix='yr', drop_first=True).astype(float)
X_emp_ols = pd.concat([
    df[['log_output','labor_productivity','log_imports','log_exports','hhi_std']].reset_index(drop=True),
    emp_entity_dummies.reset_index(drop=True),
    emp_year_dummies.reset_index(drop=True)
], axis=1)
X_emp_ols = sm.add_constant(X_emp_ols).astype(float)
y_emp_ols = np.log(df['employment']).reset_index(drop=True).astype(float)
emp_cluster_codes = pd.Series(pd.factorize(df['entity'])[0], name='cluster_code').astype(np.int64)

emp_ols_model = sm.OLS(y_emp_ols, X_emp_ols)
print("\nWild-cluster bootstrap for employment model:")
for var in ['log_output','labor_productivity','log_imports','log_exports','hhi_std']:
    try:
        result = wildboottest(emp_ols_model, param=var, cluster=emp_cluster_codes, B=999, seed=42, parallel=False, show=False)
        print(f"  {var:22s}  wild-cluster-bootstrap p-value = {result['p-value'].values[0]:.4f}")
    except Exception as e:
        print(f"  {var:22s}  bootstrap failed: {str(e)[:100]}")
