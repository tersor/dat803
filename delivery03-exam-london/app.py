"""
London Smart Meter — Interactive Energy & Weather Dashboard
Household: MAC000002

Run with:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="London Smart Meter Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Data loading (cached)
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and merging datasets…")
def load_data():
    # --- Energy ---
    energy_raw = pd.read_csv("MAC000002_energy_halfhourly.csv", parse_dates=["time"])
    energy_raw["energy(kWh/hh)"] = pd.to_numeric(
        energy_raw["energy(kWh/hh)"].astype(str).str.strip(), errors="coerce"
    )
    energy_raw = energy_raw.dropna(subset=["energy(kWh/hh)"]).sort_values("time")

    energy_hourly = (
        energy_raw.set_index("time")["energy(kWh/hh)"]
        .resample("h")
        .sum()
        .rename("energy_kwh")
        .reset_index()
    )

    # --- Hourly weather ---
    wh = pd.read_csv("London_weather_hourly.csv", parse_dates=["time"])
    hourly_cols = ["time", "temperature", "windSpeed", "humidity", "pressure"]
    wh = wh[hourly_cols].copy()
    wh["pressure"] = wh["pressure"].ffill()

    # --- Daily weather ---
    wd = pd.read_csv("london_weather.csv")
    wd["date"] = pd.to_datetime(wd["date"].astype(str), format="%Y%m%d")
    daily_cols = ["date", "sunshine", "precipitation", "cloud_cover", "snow_depth"]
    wd = wd[daily_cols].copy()
    wd["snow_depth"] = wd["snow_depth"].fillna(0)
    wd = wd.ffill()

    # --- Merge ---
    df = energy_hourly.merge(wh, on="time", how="left")
    df["date"] = df["time"].dt.normalize()
    df = df.merge(wd, on="date", how="left")
    df = df.drop(columns=["date"])
    df = df.set_index("time").sort_index()
    df = df.dropna()

    # --- Feature engineering ---
    df["hour"]        = df.index.hour
    df["dayofweek"]   = df.index.dayofweek
    df["day_name"]    = df.index.day_name()
    df["month"]       = df.index.month
    df["is_weekend"]  = (df["dayofweek"] >= 5).astype(int)
    df["season"]      = df["month"].map({
        12: "Winter", 1: "Winter",  2: "Winter",
        3:  "Spring", 4: "Spring",  5: "Spring",
        6:  "Summer", 7: "Summer",  8: "Summer",
        9:  "Autumn", 10: "Autumn", 11: "Autumn",
    })
    df["energy_ma24h"]  = df["energy_kwh"].rolling(24,  min_periods=1).mean()
    df["energy_ma48h"]  = df["energy_kwh"].rolling(48,  min_periods=1).mean()
    df["energy_ma168h"] = df["energy_kwh"].rolling(168, min_periods=1).mean()
    df["temp_lag24h"]   = df["temperature"].shift(24)
    df["energy_lag24h"] = df["energy_kwh"].shift(24)

    return df


df = load_data()
series = df["energy_kwh"].copy()

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
st.sidebar.title("⚡ Smart Meter Dashboard")
st.sidebar.markdown("**Household:** MAC000002  \n**Location:** London, UK")

date_min = df.index.min().date()
date_max = df.index.max().date()

st.sidebar.markdown("### Date filter (EDA & Overview)")
date_range = st.sidebar.date_input(
    "Select date range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_dt, end_dt = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_dt, end_dt = pd.Timestamp(date_min), pd.Timestamp(date_max)

df_filtered = df.loc[start_dt:end_dt]

st.sidebar.markdown("---")
st.sidebar.markdown("### Forecast settings")
horizon = st.sidebar.slider("Forecast horizon (hours)", min_value=12, max_value=48, value=48, step=12)
model_choice = st.sidebar.multiselect(
    "Models to run",
    options=["AR", "MA", "ARMA", "ARIMA"],
    default=["AR", "MA", "ARMA", "ARIMA"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Data: UK Power Networks LCL + Heathrow weather")

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab_overview, tab_eda, tab_ts, tab_forecast = st.tabs([
    "📋 Overview", "📊 EDA", "📈 Time Series", "🔮 Forecast"
])

# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tab_overview:
    st.header("Data Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total observations", f"{len(df):,}")
    col2.metric("Date range", f"{date_min} → {date_max}")
    col3.metric("Avg consumption", f"{df['energy_kwh'].mean():.3f} kWh/h")
    col4.metric("Peak consumption", f"{df['energy_kwh'].max():.3f} kWh/h")

    st.markdown("---")
    st.subheader("Raw data preview (first 200 rows)")
    st.dataframe(
        df.reset_index()[
            ["time", "energy_kwh", "temperature", "windSpeed", "sunshine",
             "precipitation", "cloud_cover", "snow_depth"]
        ].head(200),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Descriptive statistics")
    desc_cols = ["energy_kwh", "temperature", "windSpeed", "sunshine", "precipitation"]
    st.dataframe(df[desc_cols].describe().round(3), use_container_width=True)

    st.markdown("---")
    st.subheader("Daily total consumption — full period")
    daily = df["energy_kwh"].resample("D").sum().reset_index()
    daily.columns = ["date", "daily_kwh"]
    fig = px.line(daily, x="date", y="daily_kwh",
                  title="Daily Total Electricity Consumption — MAC000002",
                  labels={"daily_kwh": "Energy (kWh/day)", "date": "Date"},
                  color_discrete_sequence=["steelblue"])
    fig.update_traces(line_width=1.2)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════
with tab_eda:
    st.header("Exploratory Data Analysis")
    st.caption(f"Showing filtered range: **{start_dt.date()}** → **{end_dt.date()}**")

    # --- Hourly line chart ---
    st.subheader("Hourly Consumption (filtered range)")
    dfp = df_filtered[["energy_kwh", "energy_ma24h", "energy_ma48h"]].reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfp["time"], y=dfp["energy_kwh"],
                             name="Hourly", opacity=0.35,
                             line=dict(color="steelblue", width=0.8)))
    fig.add_trace(go.Scatter(x=dfp["time"], y=dfp["energy_ma24h"],
                             name="24h MA", line=dict(color="orange", width=2)))
    fig.add_trace(go.Scatter(x=dfp["time"], y=dfp["energy_ma48h"],
                             name="48h MA", line=dict(color="red", width=2)))
    fig.update_layout(
        title="Hourly Consumption with Rolling Averages",
        xaxis_title="Date", yaxis_title="Energy (kWh/hour)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    # --- Heatmap ---
    with col_left:
        st.subheader("Avg Consumption by Hour × Day of Week")
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pivot = df_filtered.pivot_table(
            values="energy_kwh", index="hour", columns="day_name", aggfunc="mean"
        ).reindex(columns=day_order)
        fig_hm = px.imshow(
            pivot,
            labels=dict(x="Day of Week", y="Hour of Day", color="kWh"),
            color_continuous_scale="YlOrRd",
            title="Heatmap — Avg kWh/hour",
            aspect="auto",
            height=420,
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # --- Season box ---
    with col_right:
        st.subheader("Consumption by Season")
        season_order = ["Winter", "Spring", "Summer", "Autumn"]
        fig_box = px.box(
            df_filtered.reset_index(),
            x="season", y="energy_kwh",
            category_orders={"season": season_order},
            color="season",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Distribution by Season",
            labels={"energy_kwh": "kWh/hour"},
            height=420,
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation Heatmap — Energy & Weather Features")
    corr_cols = ["energy_kwh", "temperature", "windSpeed", "humidity",
                 "pressure", "sunshine", "precipitation", "cloud_cover", "snow_depth"]
    corr = df_filtered[corr_cols].corr().round(2)
    mask_upper = np.triu(np.ones(corr.shape), k=1).astype(bool)
    corr_masked = corr.where(~mask_upper)
    fig_corr = px.imshow(
        corr_masked,
        text_auto=True,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title="Pearson Correlation Matrix",
        zmin=-1, zmax=1,
        height=520,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    st.subheader("Scatter — Weather Variable vs. Consumption")
    scatter_var = st.selectbox(
        "Select weather variable",
        ["temperature", "windSpeed", "sunshine", "precipitation"],
    )
    fig_sc = px.scatter(
        df_filtered.reset_index().sample(min(5000, len(df_filtered)), random_state=42),
        x=scatter_var, y="energy_kwh",
        opacity=0.3,
        trendline="ols",
        labels={"energy_kwh": "kWh/hour"},
        title=f"{scatter_var} vs. Electricity Consumption",
        color_discrete_sequence=["steelblue"],
        height=400,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")
    st.subheader("Cross-Correlation — Weather Variable vs. Consumption at Multiple Lags")
    ccf_var = st.selectbox(
        "Select variable for cross-correlation",
        ["temperature", "windSpeed", "sunshine", "precipitation"],
        key="ccf_select",
    )
    max_lag = st.slider("Max lag (hours)", min_value=12, max_value=72, value=48, step=12)

    lags = list(range(-max_lag, max_lag + 1))
    x_full = df[ccf_var].dropna()
    y_full = df["energy_kwh"].dropna()
    shared_idx = x_full.index.intersection(y_full.index)
    x_s, y_s = x_full[shared_idx], y_full[shared_idx]

    corrs = []
    for lag in lags:
        if lag == 0:
            corrs.append(x_s.corr(y_s))
        elif lag > 0:
            corrs.append(x_s.shift(lag).corr(y_s))
        else:
            corrs.append(x_s.corr(y_s.shift(-lag)))

    ccf_df = pd.DataFrame({"lag": lags, "correlation": corrs})
    fig_ccf = px.bar(
        ccf_df, x="lag", y="correlation",
        title=f"Cross-Correlation: {ccf_var} ↔ energy_kwh",
        labels={"lag": "Lag (hours)  [negative = weather leads]", "correlation": "Pearson r"},
        color="correlation",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        height=380,
    )
    fig_ccf.add_hline(y=0, line_width=1, line_color="black")
    fig_ccf.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")
    st.plotly_chart(fig_ccf, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — TIME SERIES
# ══════════════════════════════════════════════
with tab_ts:
    st.header("Time Series Analysis")

    # --- ADF test ---
    st.subheader("Stationarity — Augmented Dickey-Fuller Test")

    def run_adf(s, label):
        res = adfuller(s.dropna(), autolag="AIC")
        stationary = res[1] < 0.05
        return {
            "Series": label,
            "Test Statistic": round(res[0], 4),
            "p-value": round(res[1], 6),
            "Lags Used": res[2],
            "Stationary (5%)": "✓ Yes" if stationary else "✗ No",
        }

    adf_results = [
        run_adf(series, "Raw consumption"),
        run_adf(series.diff().dropna(), "1st-order differenced"),
        run_adf(series.diff(24).dropna(), "Seasonal-differenced (lag=24)"),
    ]
    st.dataframe(pd.DataFrame(adf_results).set_index("Series"), use_container_width=True)
    st.markdown(
        """
**Interpretation:**
- If p-value < 0.05 → series is **stationary** (no unit root)
- Raw consumption is typically **non-stationary** due to daily/weekly seasonality
- 1st-order or seasonal differencing usually achieves stationarity
        """
    )

    st.markdown("---")
    st.subheader("Seasonal Decomposition (period = 24h, additive)")
    st.markdown("Select a 2–8 week window for decomposition:")

    decomp_opts = {
        "Jan 2013 (4 weeks)": ("2013-01-01", "2013-01-28"),
        "Apr 2013 (4 weeks)": ("2013-04-01", "2013-04-28"),
        "Jul 2013 (4 weeks)": ("2013-07-01", "2013-07-28"),
        "Nov 2013 (4 weeks)": ("2013-11-01", "2013-11-28"),
    }
    decomp_sel = st.selectbox("Decomposition window", list(decomp_opts.keys()))
    d_start, d_end = decomp_opts[decomp_sel]
    decomp_series = series[d_start:d_end]

    decomp = seasonal_decompose(decomp_series, model="additive", period=24, extrapolate_trend="freq")

    fig_d = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
        vertical_spacing=0.06,
    )
    components = [
        (decomp.observed,  "steelblue"),
        (decomp.trend,     "orange"),
        (decomp.seasonal,  "green"),
        (decomp.resid,     "red"),
    ]
    for i, (comp, col) in enumerate(components, start=1):
        fig_d.add_trace(
            go.Scatter(x=comp.index, y=comp.values, line=dict(color=col, width=1.2), showlegend=False),
            row=i, col=1,
        )
    fig_d.update_layout(height=700, title_text=f"Additive Decomposition — {decomp_sel}")
    st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("---")
    st.subheader("ACF and PACF Plots")
    st.markdown(
        "**ACF** identifies MA(q) order (lag after which autocorrelation cuts off).  \n"
        "**PACF** identifies AR(p) order (lag after which partial autocorrelation cuts off)."
    )

    acf_lags = st.slider("Number of lags to display", min_value=24, max_value=96, value=72, step=24)

    series_d1 = series.diff().dropna()
    fig_acf, axes = plt.subplots(2, 1, figsize=(12, 7))
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    plot_acf(series_d1,  lags=acf_lags, ax=axes[0], alpha=0.05)
    axes[0].set_title("ACF — 1st-Order Differenced Consumption")
    axes[0].set_xlabel("Lag (hours)")
    plot_pacf(series_d1, lags=acf_lags, ax=axes[1], alpha=0.05, method="ywm")
    axes[1].set_title("PACF — 1st-Order Differenced Consumption")
    axes[1].set_xlabel("Lag (hours)")
    plt.tight_layout()
    st.pyplot(fig_acf)
    plt.close(fig_acf)

# ══════════════════════════════════════════════
# TAB 4 — FORECAST
# ══════════════════════════════════════════════
with tab_forecast:
    st.header(f"48-Hour Electricity Consumption Forecast")
    st.markdown(
        f"**Forecast horizon:** {horizon} hours  \n"
        f"**Models:** {', '.join(model_choice) if model_choice else 'None selected'}"
    )

    if not model_choice:
        st.warning("Please select at least one model in the sidebar.")
        st.stop()

    train = series.iloc[:-horizon]
    test  = series.iloc[-horizon:]

    col1, col2 = st.columns(2)
    col1.metric("Training observations", f"{len(train):,}")
    col2.metric("Test observations", f"{len(test):,} ({horizon}h)")

    # ─── Model fitting ───────────────────────
    predictions = {}
    metrics_list = []

    def mape(actual, predicted):
        mask = actual != 0
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

    with st.spinner("Fitting models… this may take 20–60 seconds"):

        if "AR" in model_choice:
            ar_aic = {}
            for p in range(1, 50):
                try:
                    m = AutoReg(train, lags=p, old_names=False).fit()
                    ar_aic[p] = m.aic
                except:
                    pass
            best_p = min(ar_aic, key=ar_aic.get)
            ar_model = AutoReg(train, lags=best_p, old_names=False).fit()
            ar_pred = ar_model.predict(start=len(train), end=len(train) + horizon - 1)
            ar_pred.index = test.index
            ar_pred = ar_pred.clip(lower=0)
            predictions[f"AR(p={best_p})"] = ar_pred
            metrics_list.append({
                "Model": f"AR(p={best_p})",
                "MAE":  round(mean_absolute_error(test, ar_pred), 4),
                "RMSE": round(np.sqrt(mean_squared_error(test, ar_pred)), 4),
                "MAPE": round(mape(test.values, ar_pred.values), 2),
            })

        if "MA" in model_choice:
            ma_aic = {}
            for q in range(1, 10):
                try:
                    m = ARIMA(train, order=(0, 0, q)).fit()
                    ma_aic[q] = m.aic
                except:
                    pass
            best_q = min(ma_aic, key=ma_aic.get)
            ma_model = ARIMA(train, order=(0, 0, best_q)).fit()
            ma_pred = ma_model.get_forecast(steps=horizon).predicted_mean
            ma_pred.index = test.index
            ma_pred = ma_pred.clip(lower=0)
            predictions[f"MA(q={best_q})"] = ma_pred
            metrics_list.append({
                "Model": f"MA(q={best_q})",
                "MAE":  round(mean_absolute_error(test, ma_pred), 4),
                "RMSE": round(np.sqrt(mean_squared_error(test, ma_pred)), 4),
                "MAPE": round(mape(test.values, ma_pred.values), 2),
            })

        best_arma = (2, 2)  # sensible default
        if "ARMA" in model_choice or "ARIMA" in model_choice:
            arma_aic = {}
            for p in range(1, 6):
                for q in range(1, 6):
                    try:
                        m = ARIMA(train, order=(p, 0, q)).fit()
                        arma_aic[(p, q)] = m.aic
                    except:
                        pass
            if arma_aic:
                best_arma = min(arma_aic, key=arma_aic.get)

        if "ARMA" in model_choice:
            arma_model = ARIMA(train, order=(best_arma[0], 0, best_arma[1])).fit()
            arma_pred = arma_model.get_forecast(steps=horizon).predicted_mean
            arma_pred.index = test.index
            arma_pred = arma_pred.clip(lower=0)
            predictions[f"ARMA({best_arma[0]},{best_arma[1]})"] = arma_pred
            metrics_list.append({
                "Model": f"ARMA({best_arma[0]},{best_arma[1]})",
                "MAE":  round(mean_absolute_error(test, arma_pred), 4),
                "RMSE": round(np.sqrt(mean_squared_error(test, arma_pred)), 4),
                "MAPE": round(mape(test.values, arma_pred.values), 2),
            })

        if "ARIMA" in model_choice:
            arima_model = ARIMA(train, order=(best_arma[0], 1, best_arma[1])).fit()
            arima_pred = arima_model.get_forecast(steps=horizon).predicted_mean
            arima_pred.index = test.index
            arima_pred = arima_pred.clip(lower=0)
            predictions[f"ARIMA({best_arma[0]},1,{best_arma[1]})"] = arima_pred
            metrics_list.append({
                "Model": f"ARIMA({best_arma[0]},1,{best_arma[1]})",
                "MAE":  round(mean_absolute_error(test, arima_pred), 4),
                "RMSE": round(np.sqrt(mean_squared_error(test, arima_pred)), 4),
                "MAPE": round(mape(test.values, arima_pred.values), 2),
            })

    # ─── Forecast plot ───────────────────────
    st.markdown("---")
    st.subheader("Forecast vs. Actual")

    context = series.iloc[-(96 + horizon):-horizon]
    colors_map = ["steelblue", "coral", "green", "purple", "orange"]

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=context.index, y=context.values,
        name="Training (last 96h)", line=dict(color="lightgrey", width=1.0),
    ))
    fig_fc.add_trace(go.Scatter(
        x=test.index, y=test.values,
        name="Actual", line=dict(color="black", width=2.5),
    ))
    for (label, pred), color in zip(predictions.items(), colors_map):
        fig_fc.add_trace(go.Scatter(
            x=pred.index, y=pred.values,
            name=label, line=dict(color=color, width=1.8, dash="dash"),
        ))

    # Shade forecast window
    fig_fc.add_vrect(
        x0=test.index[0], x1=test.index[-1],
        fillcolor="yellow", opacity=0.07,
        layer="below", line_width=0,
        annotation_text="Forecast window", annotation_position="top left",
    )
    fig_fc.update_layout(
        title=f"{horizon}-Hour Electricity Forecast — MAC000002",
        xaxis_title="Date/Time",
        yaxis_title="Energy (kWh/hour)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=480,
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # ─── Metrics table ───────────────────────
    st.markdown("---")
    st.subheader("Model Performance Metrics")
    metrics_df = pd.DataFrame(metrics_list).set_index("Model")
    st.dataframe(metrics_df, use_container_width=True)

    # Bar charts for each metric
    metric_cols = st.columns(3)
    for col, metric in zip(metric_cols, ["MAE", "RMSE", "MAPE"]):
        with col:
            fig_m = px.bar(
                metrics_df.reset_index(),
                x="Model", y=metric,
                color="Model",
                color_discrete_sequence=colors_map,
                title=metric,
                labels={metric: metric + (" (%)" if metric == "MAPE" else " (kWh)")},
                height=320,
                text=metric,
            )
            fig_m.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_m.update_layout(showlegend=False, uniformtext_minsize=8)
            st.plotly_chart(fig_m, use_container_width=True)

    st.markdown("---")
    st.subheader("Model Interpretation")
    st.markdown(
        """
| Model | Mechanism | Typical use case |
|-------|-----------|-----------------|
| **AR** | Uses past consumption values weighted by learned coefficients | Stable, persistent series |
| **MA** | Models the series as a function of past forecast *errors* | Short-term shock propagation |
| **ARMA** | Combines AR and MA terms | General stationary series |
| **ARIMA** | ARMA + differencing to handle trends/non-stationarity | Trending or drifting series |

**Lower MAE / RMSE / MAPE = better forecast.**  
MAPE (%) is scale-independent and allows direct comparison across different periods.
        """
    )
