import pandas as pd
from bls_series import series

years = list(range(2015, 2026))

# annual average (thousand employees) per series
annual = {}
for sid, yrs in series.items():
    annual[sid] = {y: sum(v)/12 for y, v in yrs.items()}

# Build entity-level employment matching BEA subsector names
rows = []
for y in years:
    def g(sid): return annual[sid][y]

    entity_map = {
        "Wood products": g('CES3132100001'),
        "Nonmetallic mineral products": g('CES3132700001'),
        "Primary metals": g('CES3133100001'),
        "Fabricated metal products": g('CES3133200001'),
        "Machinery": g('CES3133300001'),
        "Computer and electronic products": g('CES3133400001'),
        "Electrical equipment, appliances, and components": g('CES3133500001'),
        "Motor vehicles, bodies and trailers, and parts": g('CES3133600101'),
        "Other transportation equipment": g('CES3133600001') - g('CES3133600101'),  # 336 total minus motor vehicles
        "Furniture and related products": g('CES3133700001'),
        "Miscellaneous manufacturing": g('CES3133900001'),
        "Food and beverage and tobacco products": g('CES3231100001') + g('CES3232910001'),  # food + beverage (tobacco not separately available)
        "Textile mills and textile product mills": g('CES3231300001') + g('CES3231400001'),
        "Apparel and leather and allied products": g('CES3231500001'),  # leather not separately available in this pull
        "Paper products": g('CES3232200001'),
        "Printing and related support activities": g('CES3232300001'),
        "Petroleum and coal products": g('CES3232400001'),
        "Chemical products": g('CES3232500001'),
        "Plastics and rubber products": g('CES3232600001'),
    }
    for entity, val in entity_map.items():
        rows.append(dict(entity=entity, year=y, employment_thousands=round(val, 2)))

emp = pd.DataFrame(rows)
emp.to_csv("bls_employment_annual.csv", index=False)
print(f"Employment panel: {emp.shape[0]} obs, {emp['entity'].nunique()} entities x {emp['year'].nunique()} years")
print(emp.pivot(index='entity', columns='year', values='employment_thousands').round(1))
