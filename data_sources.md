# Data Sources and Construction Methodology

## Panel structure

19 NAICS-3 manufacturing subsectors, observed annually from 2015 to 2025 (N = 209 industry-year observations).

## Variables and sources

| Variable | Source | Notes |
|---|---|---|
| Gross Output ($ billions) | U.S. Bureau of Economic Analysis, "Gross Output by Industry," Quarterly, Value Added and Gross Output Interactive Tables (bea.gov/data/gdp/gdp-industry) | Quarterly series averaged to annual |
| Employment (thousands) | U.S. Bureau of Labor Statistics, Current Employment Statistics (CES), "All Employees, Thousands" by NAICS 3-digit industry, seasonally adjusted, monthly | Averaged to annual |
| Labor Productivity (Index, 2017=100) | U.S. Bureau of Labor Statistics, Industry Productivity program, "Labor productivity by detailed industries," annual, 3-digit NAICS, All Workers basis | |
| Exports ($) | U.S. International Trade Commission (USITC) DataWeb, Total Exports, FAS Value, Actual Dollars, NAICS classification, annual, all countries aggregated | |
| Imports ($) | USITC DataWeb, Imports for Consumption, Customs Value, Actual Dollars, NAICS classification, annual, all countries aggregated | |
| Supplier Concentration (HHI, 0-10,000) | Computed from USITC DataWeb import data broken out by country of origin AND by NAICS-3 industry, annual | HHI = sum of squared percentage import shares across all countries of origin, per industry-year |

## Entity construction notes

- **Food and beverage and tobacco products**: BEA/BLS combine NAICS 311 (Food) + 312 (Beverage and Tobacco). Employment sums 311 + Beverage manufacturing (3121); tobacco is not separately available in the BLS pull used. HHI recomputed on the combined 311+312 import value by country, not averaged from two separate HHIs.
- **Textile mills and textile product mills**: Combines NAICS 313 (Textile Mills) + 314 (Textile Product Mills). HHI recomputed on the combined import value by country.
- **Apparel and leather and allied products**: Uses NAICS 315 (Apparel); Leather (316) is not separately available in the BLS employment source used, so this entity is a modest under-estimate for that subsector. HHI recomputed on the combined 315+316 import value by country.
- **Motor vehicles, bodies and trailers, and parts / Other transportation equipment**: USITC's NAICS 336 (Transportation Equipment) trade data is a single combined code. Export/import values were split into the two BEA/BLS subsector categories proportionally, using each year's employment share between the two. The BLS Industry Productivity series and the computed HHI for NAICS 336 do not break out motor vehicles vs. other transportation equipment either; both subsectors are assigned the same parent-category (336) values as an approximation.

## Documented data-infrastructure gaps

The process of assembling this panel directly surfaced several limitations in existing federal data infrastructure:

1. **No single integrated access point.** Each variable required a separate query against a separate agency system (BEA's interactive tables, BLS's Current Employment Statistics multi-screen tool, BLS's separate Industry Productivity program, and USITC's DataWeb), with no shared query interface across agencies.
2. **Inconsistent classification granularity.** BEA and BLS report several NAICS-3 categories only in combined form; USITC trade data is available at finer granularity, requiring aggregation upward to match.
3. **Incomplete native disaggregation.** USITC reports Transportation Equipment (NAICS 336) as a single combined trade figure, while BEA/BLS separately report Motor Vehicles and Other Transportation Equipment, requiring a documented approximation to reconcile.
4. **Country-level detail requires separate, larger queries.** Obtaining the country-of-origin detail needed to compute supplier concentration required a query returning nearly 40,000 rows, substantially more effort than the industry-aggregate trade data.

## Known limitations

- **Panel size**: N=209 (19 entities x 11 years) is modest by econometric standards, though it is a genuine cross-sectional panel drawn entirely from primary federal sources.
- **Transportation-equipment split**: the proportional split of USITC's combined NAICS 336 data is an approximation, not a native disaggregation from the source.
- **Endogeneity**: import/export coefficients in the regression reflect co-movement with output, not a clean causal trade elasticity.
- **No trade x concentration interaction terms**: the current model specification implies the same percentage output response to a trade shock across all subsectors, which understates the differential vulnerability of highly concentrated subsectors. See `methodology.md` for the proposed extension.
