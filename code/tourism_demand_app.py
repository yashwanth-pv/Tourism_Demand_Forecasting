"""
Tourism Demand Forecasting — Streamlit App
--------------------------------------------
Upload historical tourism data (or use the built-in sample dataset),
choose a forecasting method, and visualize predicted future demand.

Run locally with:
    pip install -r requirements.txt
    streamlit run tourism_demand_app.py
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Tourism Demand Forecasting", page_icon="🧳", layout="wide")

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

@st.cache_data
def generate_sample_data(n_years: int = 6, seed: int = 42) -> pd.DataFrame:
    """Synthetic monthly tourist-arrivals dataset with trend + seasonality + noise."""
    rng = np.random.default_rng(seed)
    periods = n_years * 12
    dates = pd.date_range(end=pd.Timestamp.today().replace(day=1), periods=periods, freq="MS")

    trend = np.linspace(8000, 15000, periods)
    seasonality = 3000 * np.sin(2 * np.pi * (dates.month / 12) + 1.2) + 1500 * (
        (dates.month.isin([6, 7, 8, 12])).astype(float)
    )
    noise = rng.normal(0, 500, periods)
    # simulate a pandemic-style dip
    dip = np.zeros(periods)
    if periods > 40:
        dip[30:42] = -np.linspace(0, 9000, 12)
        if periods > 42:
            dip[42:54] = np.linspace(-9000, 0, min(12, periods - 42))

    visitors = np.clip(trend + seasonality + noise + dip, 0, None).round()
    return pd.DataFrame({"date": dates, "visitors": visitors})


def load_uploaded(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df


def moving_average_forecast(series: pd.Series, horizon: int, window: int) -> np.ndarray:
    history = list(series.values.astype(float))
    preds = []
    for _ in range(horizon):
        window_vals = history[-window:]
        next_val = float(np.mean(window_vals))
        preds.append(next_val)
        history.append(next_val)
    return np.array(preds)


def linear_trend_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    x = np.arange(len(series)).reshape(-1, 1)
    y = series.values.astype(float)
    model = LinearRegression().fit(x, y)
    future_x = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
    return model.predict(future_x)


def holt_winters_forecast(series: pd.Series, horizon: int, seasonal_periods: int):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    seasonal = "add" if len(series) >= 2 * seasonal_periods else None
    model = ExponentialSmoothing(
        series.values.astype(float),
        trend="add",
        seasonal=seasonal,
        seasonal_periods=seasonal_periods if seasonal else None,
        initialization_method="estimated",
    ).fit()
    return model.forecast(horizon)


def sarima_forecast(series: pd.Series, horizon: int, seasonal_periods: int):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    order = (1, 1, 1)
    seasonal_order = (1, 1, 0, seasonal_periods) if len(series) >= 2 * seasonal_periods else (0, 0, 0, 0)
    model = SARIMAX(
        series.values.astype(float),
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    return model.forecast(horizon)


def _make_lag_features(values: np.ndarray, lag: int):
    X, y = [], []
    for i in range(lag, len(values)):
        X.append(values[i - lag:i])
        y.append(values[i])
    return np.array(X), np.array(y)


def mlp_epoch_forecast(train_values: np.ndarray, test_values, horizon: int, epochs: int, lag: int = 6):
    """Trains a small MLP one epoch at a time (partial_fit), recording R^2 'accuracy'
    per epoch on the training set and, if a hold-out is supplied, on validation too.
    Returns (forecast, train_acc_history, val_acc_history)."""
    lag = max(2, min(lag, len(train_values) // 3))
    X, y = _make_lag_features(train_values, lag)
    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    Xs, ys = scaler_X.transform(X), scaler_y.transform(y.reshape(-1, 1)).ravel()

    val_Xs = val_ys = None
    if test_values is not None and len(test_values) > 0:
        combined = np.concatenate([train_values[-lag:], test_values])
        vX, vy = _make_lag_features(combined, lag)
        val_Xs, val_ys = scaler_X.transform(vX), scaler_y.transform(vy.reshape(-1, 1)).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=(16, 8), activation="relu", solver="adam",
        warm_start=True, max_iter=1, random_state=42,
    )

    train_acc_hist, val_acc_hist = [], []
    for _ in range(epochs):
        model.partial_fit(Xs, ys)
        train_acc_hist.append(r2_score(ys, model.predict(Xs)))
        if val_Xs is not None:
            val_acc_hist.append(r2_score(val_ys, model.predict(val_Xs)))

    # Recursive multi-step forecast using the trained model
    window = list(train_values[-lag:])
    preds = []
    for _ in range(horizon):
        x_in = scaler_X.transform([window[-lag:]])
        pred = scaler_y.inverse_transform(model.predict(x_in).reshape(-1, 1))[0][0]
        preds.append(pred)
        window.append(pred)

    return np.array(preds), train_acc_hist, val_acc_hist


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    mape = np.mean(np.abs((actual - predicted) / np.where(actual == 0, 1, actual))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


# ----------------------------------------------------------------------
# Sidebar — data & settings
# ----------------------------------------------------------------------

st.sidebar.header("⚙️ Settings")

data_source = st.sidebar.radio("Data source", ["Use sample dataset", "Upload CSV"])

if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV (needs a date column and a numeric demand column)", type=["csv"])
    if uploaded is not None:
        raw_df = load_uploaded(uploaded)
    else:
        st.sidebar.info("No file uploaded yet — showing sample dataset instead.")
        raw_df = generate_sample_data()
else:
    n_years = st.sidebar.slider("Years of sample history", 3, 10, 6)
    raw_df = generate_sample_data(n_years)

cols = list(raw_df.columns)
date_col = st.sidebar.selectbox("Date column", cols, index=cols.index("date") if "date" in cols else 0)
numeric_cols = [c for c in cols if c != date_col]
value_col = st.sidebar.selectbox(
    "Demand column (e.g. visitors, arrivals)", numeric_cols,
    index=numeric_cols.index("visitors") if "visitors" in numeric_cols else 0
)

df = raw_df[[date_col, value_col]].copy()
df.columns = ["date", "value"]
df["date"] = pd.to_datetime(df["date"])
df = df.dropna().sort_values("date").reset_index(drop=True)

freq_choice = st.sidebar.selectbox("Data frequency", ["Monthly", "Weekly", "Daily"], index=0)
seasonal_periods = {"Monthly": 12, "Weekly": 52, "Daily": 7}[freq_choice]

method = st.sidebar.selectbox(
    "Forecasting method",
    ["Linear Trend", "Moving Average", "Holt-Winters (trend+seasonality)", "SARIMA", "Neural Network (MLP, epoch training)"],
)

horizon = st.sidebar.slider("Forecast horizon (periods ahead)", 1, 36, 12)

ma_window = 6
epochs = 30
if method == "Moving Average":
    ma_window = st.sidebar.slider("Moving average window", 2, 24, 6)
if method == "Neural Network (MLP, epoch training)":
    epochs = st.sidebar.slider("Training epochs", 10, 300, 60, step=10)

test_size = st.sidebar.slider("Hold-out size for validation (periods)", 3, min(24, max(4, len(df) // 3)), min(6, max(4, len(df)//4)))

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

st.title("🧳 Tourism Demand Forecasting")
st.caption("Forecast tourist arrivals / bookings / demand using classical time-series methods.")

st.subheader("📊 Historical Data")
st.dataframe(df.tail(10), use_container_width=True)

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=df["date"], y=df["value"], mode="lines", name="Historical", line=dict(color="#2563eb")))
fig_hist.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=350, title="Historical Demand")
st.plotly_chart(fig_hist, use_container_width=True)

if len(df) < test_size + 5:
    st.warning("Not enough data points for a reliable validation split — try reducing hold-out size.")
    st.stop()

# ----------------------------------------------------------------------
# Validation on hold-out set
# ----------------------------------------------------------------------

train = df["value"].iloc[: -test_size]
test = df["value"].iloc[-test_size:]

def run_method(train_series: pd.Series, horizon: int):
    if method == "Moving Average":
        return moving_average_forecast(train_series, horizon, ma_window)
    elif method == "Linear Trend":
        return linear_trend_forecast(train_series, horizon)
    elif method == "Holt-Winters (trend+seasonality)":
        return np.asarray(holt_winters_forecast(train_series, horizon, seasonal_periods))
    elif method == "SARIMA":
        return np.asarray(sarima_forecast(train_series, horizon, seasonal_periods))
    elif method == "Neural Network (MLP, epoch training)":
        preds, _, _ = mlp_epoch_forecast(train_series.values, None, horizon, epochs)
        return preds

train_acc_hist, val_acc_hist = [], []
with st.spinner("Validating model on hold-out data..."):
    if method == "Neural Network (MLP, epoch training)":
        val_preds, train_acc_hist, val_acc_hist = mlp_epoch_forecast(
            train.values, test.values, test_size, epochs
        )
    else:
        val_preds = run_method(train, test_size)

metrics = evaluate(test.values, val_preds)

st.subheader("✅ Validation on Recent Hold-out Period")
m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{metrics['MAE']:.1f}")
m2.metric("RMSE", f"{metrics['RMSE']:.1f}")
m3.metric("MAPE", f"{metrics['MAPE (%)']:.1f}%")

fig_val = go.Figure()
fig_val.add_trace(go.Scatter(x=df["date"], y=df["value"], mode="lines", name="Actual", line=dict(color="#2563eb")))
fig_val.add_trace(go.Scatter(
    x=df["date"].iloc[-test_size:], y=val_preds, mode="lines+markers",
    name="Predicted (hold-out)", line=dict(color="#f97316", dash="dash")
))
fig_val.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=350, title="Actual vs. Predicted (Validation)")
st.plotly_chart(fig_val, use_container_width=True)

# ----------------------------------------------------------------------
# Model accuracy over epochs (Neural Network method only)
# ----------------------------------------------------------------------

if method == "Neural Network (MLP, epoch training)" and train_acc_hist:
    st.subheader("📉 Model Accuracy over Epochs")
    epoch_x = list(range(1, len(train_acc_hist) + 1))
    fig_epoch = go.Figure()
    fig_epoch.add_trace(go.Scatter(x=epoch_x, y=train_acc_hist, mode="lines", name="Train Accuracy (R²)", line=dict(color="#2563eb")))
    if val_acc_hist:
        fig_epoch.add_trace(go.Scatter(x=epoch_x, y=val_acc_hist, mode="lines", name="Validation Accuracy (R²)", line=dict(color="#f97316")))
    fig_epoch.update_layout(
        margin=dict(l=10, r=10, t=30, b=10), height=350,
        title="Training vs. Validation Accuracy per Epoch",
        xaxis_title="Epoch", yaxis_title="Accuracy (R²)",
    )
    st.plotly_chart(fig_epoch, use_container_width=True)
    st.caption("R² compares predictions to actuals on scaled lag-window features; it can start negative and climb as the network learns, and a widening gap from the training curve signals overfitting.")

# ----------------------------------------------------------------------
# Future forecast (train on full data)
# ----------------------------------------------------------------------

st.subheader(f"🔮 Forecast — Next {horizon} Periods")

with st.spinner("Generating future forecast..."):
    if method == "Neural Network (MLP, epoch training)":
        future_preds, _, _ = mlp_epoch_forecast(df["value"].values, None, horizon, epochs)
    else:
        future_preds = run_method(df["value"], horizon)

freq_map = {"Monthly": "MS", "Weekly": "W", "Daily": "D"}
future_dates = pd.date_range(
    start=df["date"].iloc[-1], periods=horizon + 1, freq=freq_map[freq_choice]
)[1:]

forecast_df = pd.DataFrame({"date": future_dates, "forecast": np.round(future_preds, 1)})

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(x=df["date"], y=df["value"], mode="lines", name="Historical", line=dict(color="#2563eb")))
fig_fc.add_trace(go.Scatter(
    x=forecast_df["date"], y=forecast_df["forecast"], mode="lines+markers",
    name="Forecast", line=dict(color="#16a34a", dash="dash")
))
fig_fc.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=400, title=f"Demand Forecast ({method})")
st.plotly_chart(fig_fc, use_container_width=True)

st.dataframe(forecast_df, use_container_width=True)

csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download forecast as CSV", data=csv_bytes, file_name="tourism_demand_forecast.csv", mime="text/csv")

st.caption(
    "Methods: Moving Average (simple baseline), Linear Trend (regression on time), "
    "Holt-Winters (captures trend + seasonality), SARIMA (autoregressive with seasonal differencing), "
    "Neural Network (learns nonlinear patterns from lagged windows over training epochs). "
    "Validation metrics are computed on the most recent hold-out periods, not seen during that fit."
)

# ----------------------------------------------------------------------
# Scalability & Performance Analysis
# ----------------------------------------------------------------------

st.subheader("🚀 Scalability & Performance Analysis")
st.caption("Wall-clock benchmark of each method fitting on your current dataset "
           f"({len(df)} rows) and forecasting {horizon} periods ahead, run just now on this machine.")


@st.cache_data(show_spinner=False)
def run_scalability_benchmark(values: np.ndarray, horizon: int, seasonal_periods: int, ma_window: int, epochs: int) -> pd.DataFrame:
    series = pd.Series(values)
    benchmarks = [
        {
            "Method": "Moving Average",
            "fn": lambda: moving_average_forecast(series, horizon, ma_window),
            "Time Complexity": "O(n) — O(1) per forecast step",
            "Space Complexity": "O(window)",
            "Scalability Rating": "⭐⭐⭐⭐⭐ Excellent",
            "Best Suited For": "Very large datasets, real-time / streaming updates",
        },
        {
            "Method": "Linear Trend",
            "fn": lambda: linear_trend_forecast(series, horizon),
            "Time Complexity": "O(n) — closed-form regression",
            "Space Complexity": "O(n)",
            "Scalability Rating": "⭐⭐⭐⭐⭐ Excellent",
            "Best Suited For": "Large datasets with a stable long-term trend",
        },
        {
            "Method": "Holt-Winters",
            "fn": lambda: holt_winters_forecast(series, horizon, seasonal_periods),
            "Time Complexity": "O(n) per optimization iteration",
            "Space Complexity": "O(n)",
            "Scalability Rating": "⭐⭐⭐⭐ Good",
            "Best Suited For": "Medium datasets with clear trend + seasonality",
        },
        {
            "Method": "SARIMA",
            "fn": lambda: sarima_forecast(series, horizon, seasonal_periods),
            "Time Complexity": "O(n · iterations) — iterative MLE, slows on long/seasonal series",
            "Space Complexity": "O(n)",
            "Scalability Rating": "⭐⭐⭐ Moderate",
            "Best Suited For": "Small-to-medium datasets needing strong statistical rigor",
        },
        {
            "Method": "Neural Network (MLP)",
            "fn": lambda: mlp_epoch_forecast(series.values, None, horizon, min(epochs, 30))[0],
            "Time Complexity": "O(epochs · n · hidden units)",
            "Space Complexity": "O(n + parameters)",
            "Scalability Rating": "⭐⭐⭐ Moderate (GPU/batch-scalable at larger scale)",
            "Best Suited For": "Large datasets with complex nonlinear patterns",
        },
    ]

    rows = []
    for b in benchmarks:
        try:
            t0 = time.perf_counter()
            b["fn"]()
            elapsed = round(time.perf_counter() - t0, 4)
        except Exception:
            elapsed = float("nan")
        rows.append({
            "Method": b["Method"],
            "Run Time (s)": elapsed,
            "Time Complexity": b["Time Complexity"],
            "Space Complexity": b["Space Complexity"],
            "Scalability Rating": b["Scalability Rating"],
            "Best Suited For": b["Best Suited For"],
        })
    return pd.DataFrame(rows)


with st.spinner("Benchmarking all methods on your dataset..."):
    perf_df = run_scalability_benchmark(df["value"].values, horizon, seasonal_periods, ma_window, epochs)

st.dataframe(perf_df, use_container_width=True, hide_index=True)
st.caption(
    "Run times are measured live on this dataset size and machine, so they'll change with more rows, "
    "a longer horizon, or more epochs. Complexity/scalability columns are general guidance, not measurements."
)
