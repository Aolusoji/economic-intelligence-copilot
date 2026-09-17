import pandas as pd

df = pd.read_excel('/mnt/user-data/uploads/labor-productivity-detailed-industries__1_.xlsx', sheet_name='MachineReadable')
sub = df[(df.Sector=='Manufacturing') & (df.Digit=='3-Digit') & (df.Measure=='Labor productivity')
         & (df.Units=='Index (2017=100)') & (df.Year.between(2015,2025))].copy()
sub['NAICS'] = sub['NAICS'].astype(str)

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
sub['entity'] = sub['NAICS'].map(naics_to_entity)

# For combined categories (311+312, 313+314, 315+316), average the index values
# (both are already index numbers on the same 2017=100 base, so simple averaging is a
# reasonable approximation when BEA/BLS report the pair as one combined line)
prod = sub.groupby(['entity','Year'], as_index=False)['Value'].mean()
prod = prod.rename(columns={'Year':'year','Value':'labor_productivity_index'})

# Split "Transportation equipment" (336) into motor vehicles vs other, using the same
# BLS employment-based share used for the trade-flow split earlier
emp = pd.read_csv("bls_employment_annual.csv")
emp_mv = emp[emp.entity == "Motor vehicles, bodies and trailers, and parts"].set_index("year")["employment_thousands"]
emp_ot = emp[emp.entity == "Other transportation equipment"].set_index("year")["employment_thousands"]
mv_share = emp_mv / (emp_mv + emp_ot)

transport = prod[prod.entity == "Transportation equipment"]
rest = prod[prod.entity != "Transportation equipment"]
split_rows = []
for _, row in transport.iterrows():
    y = row['year']
    # productivity index isn't additive like $ values, so both subsectors get the same
    # parent-category productivity index as a reasonable proxy (no finer BLS breakdown exists)
    split_rows.append(dict(entity="Motor vehicles, bodies and trailers, and parts", year=y, labor_productivity_index=row['labor_productivity_index']))
    split_rows.append(dict(entity="Other transportation equipment", year=y, labor_productivity_index=row['labor_productivity_index']))
prod_final = pd.concat([rest, pd.DataFrame(split_rows)], ignore_index=True)
prod_final.to_csv("bls_productivity_annual.csv", index=False)

print(f"Productivity panel: {prod_final.shape[0]} obs, {prod_final['entity'].nunique()} entities x {prod_final['year'].nunique()} years")
print(prod_final.pivot(index='entity', columns='year', values='labor_productivity_index').round(1))
