# ✈️ YVR Winter Flight Delay Analysis

**Predicting departure delays at Vancouver International Airport using weather data and machine learning**

> A data science project by Sandaru, Laura, Julio & Diego

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data Sources](#data-sources)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [References](#references)

---

## Overview

Flight delays at Vancouver International Airport (YVR) have significant economic consequences — the average cost of aircraft block time is approximately **$100.80 per minute**. This project investigates the relationship between winter weather conditions and flight departure delays at YVR using real-world data from two public APIs.

The analysis focuses on the **2023–2024 winter season** and applies dimensionality reduction, factor analysis, and machine learning classification to predict delay severity categories.

**Delay Categories Defined:**
| Category | Delay Time |
|---|---|
| No Delay | ≤ 0 minutes |
| Short Delay | 1 – 15 minutes |
| Long Delay | 16 – 60 minutes |
| Very Long Delay | > 60 minutes |

---

## Repository Structure

```
yvr-flight-delay-analysis/
│
├── README.md
│
├── data/
│   ├── YVR_flight_data_2023-2024_departure.csv
│   ├── Weather-VAN21-24.csv
│   └── final_winter_season_data.csv          ← processed dataset used in analysis
│
├── notebooks/
│   └── DataExtraction.ipynb                  ← data collection, merging & preprocessing
│
├── analysis/
│   └── Final_Code_4830.Rmd                   ← full statistical analysis in R
|
├── output/
|   └── Final-Code-4830.html                  ← rendered analysis report
|
└── docs/
    ├── Final_Presentation_4830.pptx          ← project presentation slides
    ├── Literature_Review.docx                ← literature review on PCA & flight delays
    ├── Weather_variable_description.docx     ← description of all weather variables
    └── diagrams/
        └── Handling_missing_values.drawio    ← flowchart for missing data imputation logic
```

---

## Data Sources

This project uses **real-world data** from two public APIs:

| Source | Data | API |
|---|---|---|
| [Aviation Edge](https://aviation-edge.com/) | Flight schedules, delays, status, airline info | REST API |
| [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api) | Hourly historical weather at YVR | REST API |

The datasets were merged on **departure scheduled time (rounded to the hour)** to align weather conditions with each flight.

> **Note:** Raw API data is not included in this repository due to size and licensing constraints. The processed winter season dataset (`final_winter_season_data.csv`) is the primary input for the R analysis.

### Weather Variables Used

| Variable | Unit | Description |
|---|---|---|
| `temperature_2m` | °C | Air temperature at 2m above ground |
| `precipitation` | mm | Total precipitation (rain + snow) |
| `rain` | mm | Liquid precipitation only |
| `snowfall` | cm | Snowfall depth |
| `weather_code` | WMO code | Encoded weather condition |
| `cloud_cover` | % | Sky cloud coverage |
| `wind_speed_10m` | m/s | Wind speed at 10m |
| `wind_gusts_10m` | m/s | Peak wind gusts at 10m |
| `pressure_msl` | hPa | Atmospheric pressure at sea level |
| `surface_pressure` | hPa | Atmospheric pressure at ground |
| `relative_humidity_2m` | % | Relative humidity at 2m |
| `wind_direction_10m` | degrees | Wind direction at 10m |

---

## Methodology

### 1. Data Collection & Preprocessing (`DataExtraction.ipynb`)
- Fetched flight departure data from Aviation Edge API and weather data from Open-Meteo API
- Merged datasets on hourly timestamps
- Filtered to the **winter season** (Dec, Jan, Feb) — 278,000+ records narrowed to a seasonal subset
- Handled missing values using a logic-based imputation strategy:
  - If `departure.actualTime` is missing but `departure.delay` is available → impute actual time
  - If `departure.delay` is missing but `departure.actualTime` is available and later than scheduled → compute delay
  - Records where actual time precedes scheduled time were removed
  - See `docs/diagrams/Handling_missing_values.drawio` for the full flowchart

### 2. Exploratory Data Analysis (R)
- Univariate distributions of delay and weather variables
- Bivariate scatter plots: delay vs. each weather variable
- Time series of departure delays over the winter season
- Correlation heatmap — strong correlations found: precipitation/rain (0.96), wind speed/gusts (0.95), pressure_msl/surface_pressure (1.0)

### 3. Principal Component Analysis — Proposal 1
- Standardized 12 numeric variables and applied PCA
- Retained **PC1–PC6** (explaining ~84% of total variance)
- Used PCs as inputs to **Multinomial Logistic Regression** (AIC: 27,397 vs. 111,977 for raw variables)

### 4. Factor Analysis — Proposal 2
- Applied KMO and Bartlett's tests to assess suitability
- Removed highly correlated variables (precipitation, wind_gusts_10m, pressure_msl) to improve KMO
- Extracted **4 factors** via varimax rotation:
  - **MR1**: Temperature & humidity (15.6% variance)
  - **MR2**: Wind direction (8.6% variance)
  - **MR3**: Wind speed (10.2% variance)
  - **MR4**: Snowfall (5.0% variance)
- Regression using factor scores explained 13.4% of variance in departure delay

### 5. Stepwise Regression — Proposal 3
- Applied `stepAIC()` with bidirectional selection
- Final model retained 10 predictors; R² = 1.67% (very weak linear fit)
- Confirmed that non-linear models are more appropriate

### 6. Correspondence Analysis — Proposal 4
- Categorized snowfall, rain, and wind speed into ordinal buckets
- CA biplots revealed: **snowfall > wind speed > rain** in association with longer delays

### 7. Random Forest Classification
- Input: PC1–PC6 from PCA
- Split: 70% training / 30% testing
- **Accuracy: 98.43%**, Kappa: 0.9728
- Most important features: PC5 and PC6 (highest Mean Decrease Accuracy)

---

## Key Findings

- Weather variables alone explain only **~1.7% of variance** in raw delay time (linear models), but classify delay *categories* with **98.4% accuracy** (Random Forest on PCA components)
- **Snowfall** has the strongest association with longer delays, followed by wind speed and rain
- Strong multicollinearity exists in weather data — PCA and FA are essential for reducing redundancy
- The Random Forest model on PCA-reduced features significantly outperforms logistic regression
- Other factors (airline operations, air traffic, human error) likely drive much of the unexplained variance

---

## How to Run

### Python — Data Extraction (`DataExtraction.ipynb`)

```bash
# Install dependencies
pip install pandas jupyter

# Launch Jupyter
jupyter notebook notebooks/DataExtraction.ipynb
```

Place your raw CSV files in the `data/` folder before running.

### R — Statistical Analysis (`Final_Code_4830.Rmd`)

Open `analysis/Final_Code_4830.Rmd` in RStudio and knit to HTML or PDF.

**Required R packages:**
```r
install.packages(c(
  "dplyr", "ggplot2", "tidyr", "reshape2", "nnet", "caTools",
  "randomForest", "caret", "psych", "GPArotation",
  "FactoMineR", "factoextra", "pROC", "corrplot", "MASS"
))
```

Ensure the processed dataset is available at: `Dataset/final_winter_season_data.csv`

---

## Requirements

**Python (notebook):**
```
pandas
jupyter
```

**R (analysis):**
See package list above. R version 4.0+ recommended.

---

## 📊 Rendered Analysis

A fully rendered HTML report of the analysis is available in the `output/` folder.
To view it properly, download `Final-Code-4830.html` and open it in your browser,
or view it via [GitHub Pages / nbviewer / htmlpreview].

---

## References

- Aviation Edge. (n.d.). *Aviation Edge - Database and API*. https://aviation-edge.com/
- Open-Meteo. (n.d.). *Open-Meteo Historical Weather API*. https://open-meteo.com/en/docs/historical-weather-api
- Liu, H., Zhang, Y., & Li, J. (2019). Flight delay prediction with weather data using PCA and machine learning. *Journal of Air Transport Management*.
- Zhang, J., Wang, S., & Chen, Y. (2020). A PCA-based approach for predicting flight delays using weather data. *Transportation Research Part C: Emerging Technologies*.
- Mamdouh, M., Ezzat, M., & Hefny, H. A. (2023). A novel intelligent approach for flight delay prediction. *Journal of Big Data, 10*, Article 179.
- Sternberg, A., et al. (2017). A review on flight delay prediction. *arXiv:1703.06118*.
- Tang, Y. (2021). Airline flight delay prediction using machine learning models. *Proceedings of ICEI 2021*.
- Yazdi, M. F., et al. (2020). Flight delay prediction based on deep learning and Levenberg-Marquardt algorithm. *Journal of Big Data, 7*, Article 106.

---
