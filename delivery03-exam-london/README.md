# DAT803 Delivery 03 – London Energy & Weather Analysis

Time series analysis of household electricity consumption combined with London weather data. Delivered as a Jupyter Notebook and an interactive Streamlit dashboard.

## Requirements

- Python 3.9+
- pip

## Setup

```bash
pip install -r requirements.txt
```

## Running

### Jupyter Notebook (full analysis)

```bash
jupyter notebook analysis.ipynb
```

Opens the notebook in your browser. Work through the sections sequentially:
1. Data loading & cleaning
2. Exploratory data analysis
3. Feature engineering
4. Time series analysis
5. Forecasting (AR/MA/ARMA/ARIMA)
6. Conclusions

### Streamlit Dashboard (interactive app)

```bash
streamlit run app.py
```

Opens a web app at `http://localhost:8501` with tabs for Overview, EDA, Time Series, and Forecast.

## Data

| File | Description |
|------|-------------|
| `london_weather.csv` | Daily weather (sunshine, precipitation, cloud cover, snow) |
| `London_weather_hourly.csv` | Hourly weather (temperature, wind, humidity, pressure) |
| `MAC000002_energy_halfhourly.csv` | Half-hourly electricity consumption for one household |

Data covers October 2012 – February 2014, sourced from UK Power Networks LCL and Heathrow weather records.
