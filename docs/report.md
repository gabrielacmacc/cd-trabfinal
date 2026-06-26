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
### 1.4.1 Model Performance

The XGBoost model was evaluated on three independent datasets (train, validation, and test) to ensure generalization and avoid overfitting. The table below summarizes the key performance metrics:

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Accuracy | 0.9180 | 0.9179 | 0.9177 |
| Precision | 0.5746 | 0.5738 | 0.5714 |
| Recall | 0.3070 | 0.3062 | 0.3039 |
| F1-Score | 0.4002 | 0.3993 | 0.3968 |
| ROC-AUC | 0.8992 | 0.8992 | 0.8973 |

The model achieves **high accuracy (91.8%)** and **strong discriminatory power (ROC-AUC ~0.90)**, indicating excellent ability to distinguish between outbreak and non-outbreak weeks. The consistency of metrics across all three datasets suggests good generalization with no significant overfitting.

However, the **low recall (30.4%)** reveals that the model identifies only about 30% of actual outbreaks, missing many positive events. This limitation stems from the severe **class imbalance** in the dataset — only 5% of records correspond to outbreaks — which naturally biases the model toward predicting the majority class (non-outbreak).

### 1.4.2 Feature Importance Analysis

The analysis of feature importance reveals clear patterns in what drives dengue outbreaks:

| Feature | Importance |
|---------|------------|
| Year | 67.1% |
| Epidemiological Week | 20.5% |
| Average Maximum Temperature | 3.2% |
| Average Wind Gust | 1.4% |
| Minimum Humidity | 1.1% |

**Temporal factors dominate** predictions: year and week together account for nearly 88% of the model's predictive power. This confirms that dengue outbreaks in Rio Grande do Sul follow **well-defined seasonal and interannual cycles**, with epidemic years (such as 2022 and 2024) alternating with low-incidence years.

**Climatic variables have moderate impact**: while temperature (3.2%) and humidity (1.1%) influence vector biology and disease transmission, their combined contribution is significantly smaller than temporal patterns. This suggests that climate conditions help modulate epidemic severity but do not override the broader multi-year dengue cycle.

**Sanitation variables showed negligible importance** in the final model. This may be due to their annual temporal resolution (versus weekly outbreaks), limited variability across municipalities, or the fact that waste collection coverage in RS is already relatively high (>90% in most urban areas).

---

## 1.5 Conclusions

### 1.5.1 Key Findings

This project investigated whether weekly dengue outbreaks in Rio Grande do Sul municipalities can be predicted by combining meteorological data and urban sanitation indicators. The main findings are:

1. **Dengue outbreaks are highly predictable** from temporal patterns alone. An XGBoost model achieved strong discriminatory performance (ROC-AUC ~0.90), confirming that seasonal and interannual cycles are the dominant drivers of dengue transmission in the state.

2. **Climate factors play a supporting role** in outbreak prediction. Temperature and humidity contribute to explaining outbreak risk but are secondary to the temporal signal. This aligns with the biological understanding that climate affects *Aedes aegypti* development and viral replication, but population immunity and serotype circulation determine the multi-year epidemic rhythm.

3. **Sanitation data did not improve predictions**. The limited temporal resolution (annual) and small variation in collection coverage across municipalities likely prevented these features from capturing relevant urban vulnerability patterns at the weekly outbreak scale.

4. **Outbreak prediction remains challenging**. The low recall (30%) highlights the inherent difficulty of predicting rare events — even with a strong model, many outbreaks are missed. This is a common limitation in epidemiological forecasting and reflects the complex interplay of factors beyond climate and sanitation.

### 1.5.2 Limitations

The study has several important limitations:

- **Class imbalance** limits the model's ability to detect outbreaks, as only ~5% of weeks are outbreak events.
- **Strong reliance on temporal features** may reduce the model's utility for predicting outbreaks in atypical years.
- **Spatial granularity** is limited to weather station coverage, which may not capture local microclimatic variations.
- **Sanitation data resolution** (annual) is too coarse for weekly outbreak prediction.
- **Missing epidemiological data** on vector surveillance, population mobility, and serotype circulation could improve predictions if included.

### 1.5.3 Future Work

Based on the limitations identified, several directions for improvement are suggested:

**1. Addressing Class Imbalance**
- Apply **SMOTE** to generate synthetic outbreak samples.
- Use **weighted loss functions** to penalize misclassification of outbreaks more heavily.
- Explore **under-sampling** strategies to balance training data.

**2. Model Enhancement**
- Test **Random Forest** for better interpretability of feature contributions.
- Explore **LSTM networks** to capture temporal dependencies in climate and case data.
- Build **ensemble models** combining multiple classifiers for robustness.

**3. Feature Improvement**
- Include **lagged case counts** as autoregressive predictors.
- Create **spatial features** from neighboring municipalities to capture disease spread patterns.
- Develop **composite risk indices** integrating multiple climate variables.

**4. Data Expansion**
- Incorporate **entomological surveillance data** (*Aedes aegypti* infestation indices).
- Include **population mobility data** (e.g., from mobile phone or transportation records).
- Add **socioeconomic vulnerability indicators** at the municipal level.

**5. Operational Applications**
- Integrate with **seasonal climate forecasts** to extend prediction horizons.
- Develop **municipality-specific risk scores** to prioritize surveillance and vector control efforts.

### 1.5.4 Final Remarks

This project demonstrates that machine learning can effectively predict dengue outbreaks in Rio Grande do Sul using temporally resolved climate data, even with limited sanitation information. The strong predictive performance (AUC ~0.90) confirms that dengue follows predictable seasonal patterns, while the low recall highlights the need for improved outbreak detection strategies.

Future efforts could focus on incorporating entomological surveillance and mobility patterns and addressing class imbalance through specialized sampling and modeling techniques. With these enhancements, the model could become a valuable decision-support tool for public health surveillance in Brazil.

The code and datasets used in this project are available in the accompanying repository.
