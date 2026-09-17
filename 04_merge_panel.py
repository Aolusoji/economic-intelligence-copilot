import pandas as pd

panel = pd.read_csv("real_manufacturing_panel_full.csv")
hhi = pd.read_csv("hhi_annual.csv")

full = panel.merge(hhi, on=['entity','year'], how='inner')
full = full.sort_values(['entity','year']).reset_index(drop=True)
full.to_csv("real_manufacturing_panel_with_hhi.csv", index=False)
print(f"Panel with HHI: {full.shape[0]} obs, missing:\n{full.isna().sum()}")
