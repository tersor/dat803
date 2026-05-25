# London Smart Meter — Project Summary

## Overview

Analysis of household electricity consumption (MAC000002) combined with London weather data.  
The project covers exploratory analysis, time-series decomposition, stationarity testing, and
forecasting using AR, MA, ARMA, and ARIMA models.

---

## Data Sources

| File | Description | Resolution |
|------|-------------|------------|
| `MAC000002_energy_halfhourly.csv` | Electricity consumption (kWh) for one household | Half-hourly |
| `London_weather_hourly.csv` | Temperature, wind speed, humidity, pressure | Hourly |
| `london_weather.csv` | Sunshine hours, precipitation, cloud cover, snow depth | Daily |

**Date range:** October 2012 – February 2014  
**Energy dataset:** ~24 000 half-hourly rows → resampled to ~12 000 hourly rows  
**Merge strategy:** Left join (energy ← hourly weather ← daily weather on date key).  
No rows from the energy dataset are lost.  
Daily weather columns are broadcast to every hour of the same day.

---

## Files Delivered

```
analysis.ipynb      Jupyter Notebook — full analysis (6 sections)
app.py              Streamlit interactive dashboard (4 tabs)
requirements.txt    Python package dependencies
london_weather.csv  Downloaded from Kaggle (emmanuelfwerr/london-weather-data)
```

---

## Notebook — `analysis.ipynb`

### Section 1 — Data Loading & Cleaning
- Load all three CSVs with pandas, parse timestamps
- Strip whitespace from energy values, coerce to float
- Drop 1 null energy row
- Resample half-hourly energy to hourly sums (kWh/hour)
- Parse daily weather integer date (`YYYYMMDD` → datetime)
- Fill snow depth nulls with 0; forward-fill remaining nulls
- Merge all three datasets via left joins
- Report final shape, date range, and missing value counts

### Section 2 — Exploratory Data Analysis (EDA)
- **Line chart:** daily total consumption over the full period + one sample week
- **Heatmap (seaborn):** average hourly consumption by hour-of-day × day-of-week
- **Box/violin plots:** consumption distribution by month and by season
- **Correlation heatmap (seaborn):** Pearson correlation matrix of all numeric features
- **Scatter plots:** temperature, wind speed, sunshine, and precipitation vs. consumption, each with an OLS trend line

Key observations from EDA:
- Consumption peaks at 07:00–09:00 and 17:00–21:00
- Winter months consume significantly more than summer months
- Temperature has the strongest inverse correlation with consumption
- Weekday peaks are sharper than weekend peaks

### Section 3 — Feature Engineering
| Feature | Description |
|---------|-------------|
| `hour`, `dayofweek`, `month` | Calendar time features |
| `is_weekend` | Binary flag (0/1) |
| `season` | Winter / Spring / Summer / Autumn |
| `temp_lag1h`, `temp_lag24h`, `temp_lag48h` | Lagged temperature |
| `energy_ma24h`, `energy_ma48h`, `energy_ma168h` | Rolling averages of consumption (24h, 48h, 1 week) |
| `energy_lag24h`, `energy_lag168h` | Lagged consumption (same hour yesterday / last week) |

- Rolling averages plotted over a 3-month sample window
- **Cross-correlation plots** for temperature, wind speed, sunshine, and precipitation vs. consumption at lags −48h to +48h

### Section 4 — Time Series Analysis

**Stationarity (ADF test):**
- Raw consumption: non-stationary (daily and weekly seasonality + winter trend)
- 1st-order differenced series: stationary at 5% significance level
- Seasonal-differenced (lag=24): also stationary

**Seasonal decomposition:**
- Additive model, period = 24 (daily cycle)
- Extracts: Observed / Trend / Seasonal / Residual components
- Plotted over a 4-week window in January 2013

**ACF and PACF plots:**
- Computed on the 1st-order differenced series (lags 0–72)
- ACF: identifies MA(q) order — lag at which autocorrelation drops inside confidence bands
- PACF: identifies AR(p) order — lag at which partial autocorrelation cuts off

### Section 5 — Forecasting Models

**Train/test split:** last 48 hours of the series = test set

| Model | Implementation | Order selection |
|-------|---------------|-----------------|
| AR | `statsmodels.tsa.ar_model.AutoReg` | AIC over p ∈ {1..49} |
| MA | `statsmodels.tsa.arima.ARIMA(0,0,q)` | AIC over q ∈ {1..9} |
| ARMA | `ARIMA(p,0,q)` | AIC grid search p,q ∈ {1..6} |
| ARIMA | `ARIMA(p,1,q)` | Same (p,q) as best ARMA + d=1 |

All predictions clipped at 0 (consumption cannot be negative).

**Output:**
- Forecast plot: last 96h training context + actual vs. all model predictions over the 48h test window
- Metrics table: MAE, RMSE, MAPE for each model
- Bar chart comparison of all three metrics

### Section 6 — Summary & Conclusions
- Temperature is the dominant weather driver (inverse relationship)
- Wind speed and precipitation have secondary positive effects
- Sunshine hours (daily) correlate negatively with consumption
- Cross-correlation confirms weather effects persist 24–48h ahead
- ARMA typically achieves the best AIC on the stationary series
- ARIMA handles drift/trend via differencing
- Limitations: single household, daily weather broadcast to all hours, no SARIMA or exogenous regressors

---

## Streamlit App — `app.py`

Run with: `streamlit run app.py`

### Sidebar controls
- **Date range filter** applied to Overview and EDA tabs
- **Forecast horizon slider** (12 / 24 / 36 / 48 hours)
- **Model selection** checkboxes (AR, MA, ARMA, ARIMA)

### Tab 1 — Overview
- KPI metrics: total observations, date range, average and peak consumption
- Raw data preview (first 200 rows)
- Descriptive statistics table
- Interactive Plotly line chart of daily total consumption

### Tab 2 — EDA
- Hourly consumption line chart with 24h and 48h rolling averages (date-filtered)
- Hour × day-of-week heatmap (Plotly imshow)
- Consumption by season (box plot)
- Pearson correlation matrix heatmap (lower triangle only)
- Scatter plot with OLS trendline — selectable weather variable
- Cross-correlation bar chart — selectable variable, adjustable max lag

### Tab 3 — Time Series
- ADF test results table (raw / differenced / seasonal-differenced)
- Seasonal decomposition — selectable 4-week window (Jan / Apr / Jul / Nov 2013)
- ACF and PACF plots — adjustable lag count (rendered via matplotlib)

### Tab 4 — Forecast
- Models fitted with a spinner (takes 20–60 seconds for full grid search)
- Interactive forecast plot: training context + actual + all selected model predictions
- Forecast window highlighted in yellow
- Metrics table (MAE / RMSE / MAPE)
- Bar charts for each metric
- Model interpretation table

---

## Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.13
statsmodels>=0.14
scikit-learn>=1.3
jupyter>=1.0
notebook>=7.0
streamlit>=1.35
plotly>=5.20
```

Install: `pip install -r requirements.txt`

---

## Techniques Used (course alignment)

| Topic | Where used |
|-------|-----------|
| **Pandas** | Data loading, resampling, merging, feature engineering throughout |
| **Seaborn** | Heatmap, correlation matrix, violin/box plots, scatter plots (Section 2) |
| **Time series decomposition** | `seasonal_decompose` additive model, period=24 (Section 4) |
| **Stationarity testing** | ADF test on raw, differenced, and seasonal-differenced series (Section 4) |
| **ACF / PACF** | `plot_acf`, `plot_pacf` for order identification (Section 4) |
| **AR model** | `AutoReg` with AIC-based order selection (Section 5) |
| **MA model** | `ARIMA(0,0,q)` with AIC-based order selection (Section 5) |
| **ARMA / ARIMA** | Combined models with grid search, 48h forecast and evaluation (Section 5) |
