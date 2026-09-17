import pandas as pd
import numpy as np

df = pd.read_excel('/mnt/user-data/uploads/DataWeb-Query-Export__9_.xlsx', sheet_name='Query Results')
df = df.dropna(subset=['Year', 'NAIC Number'])
df = df.rename(columns={'NAIC Number': 'naics', 'Year': 'year', 'Country': 'country', 'Customs Value': 'value'})
df['naics'] = df['naics'].astype(float).astype(int).astype(str)
df['year'] = df['year'].astype(int)
df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)

print(f"Raw rows: {len(df)}, NAICS codes: {df['naics'].nunique()}, years: {sorted(df['year'].unique())}, countries: {df['country'].nunique()}")

# Compute HHI per (naics, year): sum of squared market shares (0-10000 scale, standard convention)
def hhi(group):
    total = group['value'].sum()
    if total <= 0:
        return np.nan
    shares = group['value'] / total * 100  # percentage shares
    return (shares ** 2).sum()

hhi_df = df.groupby(['naics', 'year']).apply(hhi, include_groups=False).reset_index(name='hhi')

# Map to entity names matching our panel
naics_to_entity = {
    "311": "Food and beverage and tobacco products", "312": "Food and beverage and tobacco products",
    "313": "Textile mills and textile product mills", "314": "Textile mills and textile product mills",
    "315": "Apparel and leather and allied products", "316": "Apparel and leather and allied products",
    "321": "Wood products", "322": "Paper products", "323": "Printing and related support activities",
    "324": "Petroleum and coal products", "325": "Chemical products", "326": "Plastics and rubber products",
    "327": "Nonmetallic mineral products", "331": "Primary metals", "332": "Fabricated metal products",
    "333": "Machinery", "334": "Computer and electronic products",
    "335": "Electrical equipment, appliances, and components", "336": "Transportation equipment",
    "337": "Furniture and related products", "339": "Miscellaneous manufacturing"
}
hhi_df['entity'] = hhi_df['naics'].map(naics_to_entity)

# For combined entities (311+312, 313+314, 315+316), recompute HHI on the COMBINED value
# rather than just averaging the two sub-code HHIs (averaging would be methodologically wrong
# since it ignores relative trade volumes)
combined_naics_groups = {
    "Food and beverage and tobacco products": ["311", "312"],
    "Textile mills and textile product mills": ["313", "314"],
    "Apparel and leather and allied products": ["315", "316"],
}
recomputed = []
for entity, codes in combined_naics_groups.items():
    sub = df[df['naics'].isin(codes)]
    for y, g in sub.groupby('year'):
        agg = g.groupby('country', as_index=False)['value'].sum()
        agg['value'] = agg['value']
        total = agg['value'].sum()
        if total > 0:
            shares = agg['value'] / total * 100
            recomputed.append(dict(entity=entity, year=y, hhi=(shares**2).sum()))
recomputed_df = pd.DataFrame(recomputed)

# Single-code entities: use directly
single_entities = hhi_df[~hhi_df['entity'].isin(combined_naics_groups.keys())][['entity','year','hhi']]

# Split "Transportation equipment" using same employment-share proportional logic is NOT valid for
# HHI (concentration doesn't split proportionally by volume) -- instead, compute HHI separately
# is not possible since USITC only gives combined 336. Assign same HHI to both subsectors as proxy,
# consistent with how we handled the productivity-index split earlier.
transport = single_entities[single_entities.entity == "Transportation equipment"].copy()
mv = transport.copy(); mv['entity'] = "Motor vehicles, bodies and trailers, and parts"
ot = transport.copy(); ot['entity'] = "Other transportation equipment"
single_entities = pd.concat([single_entities[single_entities.entity != "Transportation equipment"], mv, ot], ignore_index=True)

final_hhi = pd.concat([single_entities, recomputed_df], ignore_index=True)
final_hhi = final_hhi.sort_values(['entity','year']).reset_index(drop=True)
final_hhi.to_csv("hhi_annual.csv", index=False)

print(f"\nFinal HHI panel: {final_hhi.shape[0]} obs, {final_hhi['entity'].nunique()} entities x {final_hhi['year'].nunique()} years")
print(final_hhi.pivot(index='entity', columns='year', values='hhi').round(0))
