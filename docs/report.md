# Predicting Dengue Outbreaks from Urban Environmental Conditions in Rio Grande do Sul, Brazil (2020–2024)

### INF01090 — Data Science | Final Project | 2026/1

### Team Members
- Gabriela Copetti Maccagnan
- Henrique Vieira Alves
- Jean Carlo Pizzoli

---

## 1.1 Objective

Dengue fever remains a major public health burden in Brazil, with epidemic cycles driven by complex interactions between climate, urban infrastructure, and the biology of the *Aedes aegypti* mosquito that transmits the virus. Rio Grande do Sul (RS), despite being a southern state with a more temperate climate, has experienced increasingly severe dengue seasons, including a historic outbreak in 2024.

This project investigates whether **weekly dengue outbreak events** in RS municipalities can be predicted by combining two categories of environmental signals:

1. **Meteorological conditions** — temperature, humidity, precipitation, and wind measured weekly by INMET automatic stations across the state.
2. **Urban sanitation quality** — solid waste collection coverage reported annually by municipalities through the SNIS and SINISA systems.

The target variable is a **binary outbreak indicator**: a municipality-week is classified as an outbreak (1) when the estimated incidence reaches or exceeds **20 cases per 100,000 inhabitants**, following the InfoDengue alert system threshold. The study period spans epidemiological weeks 1/2020 through 53/2024.

---

## 1.2 Dataset

Three independent data sources were collected and integrated:

### Meteorological Data — INMET (2020–2024)

Hourly records from all automatic weather stations located in Rio Grande do Sul, downloaded from the Brazilian National Meteorological Institute (INMET) historical data portal. The raw dataset contains **1,748,712 hourly observations** from multiple stations, covering variables including dry-bulb air temperature (°C), relative humidity (%), total precipitation (mm/h), wind speed (m/s), wind gust (m/s), and atmospheric pressure (mB). Station metadata — WMO code, coordinates, and altitude — is embedded in each file header.

After loading, Brazilian-format numeric strings (comma as decimal separator) were converted to float, date columns were parsed from the `YYYY/MM/DD` format, and column names were normalized to ASCII lowercase with underscores. The hourly data were then aggregated to **epidemiological weeks** (ISO week, Sunday–Saturday), producing a dataset of **10,469 station-week records** with 27 features (mean, min, max, std of temperature; mean/min/max of humidity; total/mean/max precipitation; mean/max wind speed and gust; mean/min/max pressure; and record count per week).

### Dengue Cases — InfoDengue (2020–2024)

Weekly dengue case counts for all 497 RS municipalities were fetched from the InfoDengue API (`info.dengue.mat.br`), which provides nowcasted and retrospective estimates derived from the SINAN notification system. The raw dataset contains **129,519 municipality-week records** with 31 columns, including estimated cases (`casos_est`), probable cases (`casprov_est`), confirmed cases (`casconf`), population (`pop`), incidence per 100k (`p_inc100k`), and the InfoDengue alert level (`nivel`). Municipal geocodes were obtained from the IBGE Localities API.

The target variable `outbreak` was derived as:

```
outbreak = 1  if  incidence_per_100k ≥ 20
outbreak = 0  otherwise
```

Target distribution: **10,762 outbreak weeks (8.31%)** vs. **118,757 non-outbreak weeks (91.69%)**, reflecting the expected class imbalance for epidemic surveillance tasks.

### Sanitation Data — SNIS (2020–2022) and SINISA (2023–2024)

Annual solid waste collection indicators for RS municipalities were obtained from two complementary government databases:

- **SNIS** (Sistema Nacional de Informações sobre Saneamento): covers 2020–2022, downloaded as pre-cleaned CSV files per year.
- **SINISA** (Sistema Nacional de Informações sobre Saneamento Básico): covers 2023–2024, downloaded as survey response exports.

The combined dataset contains **2,485 municipality-year records** from 497 RS municipalities across 5 years, with key indicators including total solid waste collection coverage (%), urban collection coverage (%), rural collection coverage (%), selective collection (recycling) coverage (%), number of households, area (km²), and total/urban/rural population.

Due to the difference in survey structure between SNIS and SINISA, a pre-cleaning step was performed externally to harmonize column names and align indicator definitions across the two systems before ingestion. The `IRS0001` indicator (total solid waste collection coverage, %) is the primary sanitation feature used.

### Integration Key

INMET data is organized by **meteorological station**, while dengue and sanitation data are organized by **municipality (IBGE geocode)**. Weekly climate features are joined to municipality-week records via the `semana_id` key (format: `YYYY_SEWW`). Sanitation data, being annual, are joined on `(ibge_code, year)` and therefore apply to all weeks within the corresponding year.

---

## 1.3 Methodology

The project follows the standard Data Science pipeline: data collection → cleaning → transformation → feature engineering → exploratory analysis → modeling → evaluation.

### Data Cleaning

**INMET**: Column names were normalized (Unicode normalization, lowercase, underscores). Brazilian numeric strings were auto-detected by checking that ≥50% of non-null values in object columns match a BR number pattern, then converted to float. Mojibake in string fields (UTF-8 bytes misread as Latin-1) was corrected by re-encoding. Date columns were parsed explicitly to `datetime64`.

**Dengue**: City names were standardized via a three-step pipeline: fix mojibake → strip accents → uppercase → collapse whitespace. This is required for reliable joining across sources, where the same city may appear as "São Leopoldo", "SAO LEOPOLDO", or "Sao Leopoldo". Numeric case count columns were coerced to float; missing values in `casprov_est` were filled from `casos_est` where available, and remaining NaN values were filled with 0.

**Sanitation**: Boolean columns expressed as "Sim"/"Não" were automatically detected and converted to Python booleans. Numeric columns were cast from string representations with Brazilian decimal formatting.

### Data Transformation

INMET hourly records were aggregated to epidemiological weeks by grouping on `(station, year, epidemiological_week)`. The aggregation applied `mean`, `min`, `max`, and `std` to temperature; `mean`, `min`, `max` to humidity; `sum`, `mean`, `max` to precipitation; and `mean`, `max` to wind speed and gust. The resulting weekly dataset captures both central tendency and variability of each meteorological variable within the week.

Dengue data were prepared for use as the prediction target: the `se` column (format `YYYYWW`) was decomposed into `ano` and `semana_epi` integer fields, a `semana_id` join key was constructed, and the binary `outbreak` label was created from the incidence threshold.

### Feature Engineering

A feature set was constructed from the weekly INMET data in six categories:

**Climate thresholds** — binary indicators reflecting conditions known to promote *Aedes aegypti* activity: minimum temperature above 18°C (`temp_min_acima_18`), consecutive weeks above that threshold (`semanas_consecutivas_18`), relative humidity in the ideal vector range 60–75% (`umidade_ideal`), and humidity flags for high (>75%) and low (<40%) conditions.

**Composite dengue indices** — three scalar scores inspired by vector biology literature:
- *Index P* (`index_p`): a Gaussian product score over temperature (optimum 25°C), humidity (optimum 60–75%), and precipitation (optimum 50–150 mm/week). Based on Wallau et al., 2025.
- *Transmissibility index* (`transmissibilidade`): linear ramp above 18°C minimum temperature, clipped to [0, 1].
- *Climate risk score* (`risco_climatico`): weighted combination of Index P (60%) and transmissibility (40%).

**Lag features** — past values of `temp_media`, `temp_min`, `temp_max`, `umidade_media`, `precipitacao_total`, and `index_p` at lags 1, 2, 3, 4, 6, and 8 weeks, capturing the biological latency between environmental exposure and reported cases (incubation period + notification delay).

**Rolling window features** — 2-, 4-, and 8-week moving averages, trends (current minus moving average), and standard deviations for temperature and humidity variables, plus accumulated precipitation windows.

**Extreme weather events** — binary flags for heat waves (max temp > 30°C for ≥3 consecutive weeks), prolonged drought (precipitation < 10 mm for ≥3 weeks), heavy rainfall (>100 mm/week), extreme rainfall (>150 mm/week), and sudden thermal oscillation (daily amplitude > 10°C).

**Temporal and seasonal features** — month, quarter, season dummies, a `periodo_risco` flag for March–May (the historical RS dengue peak), and cyclic sine/cosine encodings of week and month to capture seasonality without discontinuity at year boundaries.

After feature engineering, the INMET dataset expands to **9,299 station-week records × 135 features**.

**Sanitation features** derived from the SNIS/SINISA data include: population density, urban/rural population proportions, residents per household, selective collection gap, service quality score, infrastructure composite score (IDS), sanitation risk score, and temporal trend features (year-over-year coverage change, whether coverage improved).

The unified dataset is constructed by merging the climate feature matrix with the dengue target on `semana_id`, then joining sanitation features on `(ibge_code, year)`.

---

## 1.4 Results

---

## 1.5 Conclusions
