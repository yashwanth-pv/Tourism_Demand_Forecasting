# 🧳 Tourism Demand Forecasting

An interactive [Streamlit](https://streamlit.io) app that forecasts tourism demand (visitor arrivals, bookings, etc.) from historical time-series data. Upload your own CSV or explore a built-in synthetic dataset, compare forecasting methods, and inspect validation accuracy, training curves, and performance benchmarks.

## Features

- **Flexible data input** – upload a CSV (pick your date and demand columns) or use a generated sample dataset with trend, seasonality, noise, and a demand dip.
- **Five forecasting methods**
  - Moving Average (baseline)
  - Linear Trend (regression on time)
  - Holt-Winters (trend + seasonality)
  - SARIMA (seasonal ARIMA)
  - Neural Network (MLP trained epoch by epoch on lagged windows)
- **Hold-out validation** – MAE, RMSE, and MAPE on the most recent periods, with an actual-vs-predicted chart.
- **Model accuracy over epochs** – line chart of training vs. validation R² per epoch (Neural Network method only).
- **Future forecast** – interactive Plotly chart, data table, and CSV download.
- **Scalability & Performance Analysis** – live benchmark table comparing run time, time/space complexity, scalability rating, and best-fit use case for every method.

## Project Structure

```
.
├── tourism_demand_app.py   # The Streamlit app
├── requirements.txt        # Python dependencies
└── README.md
```

## Installation

Python 3.9+ is recommended.

```bash
# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Running the App

```bash
streamlit run tourism_demand_app.py
```

Streamlit will open the app in your browser (default: http://localhost:8501).

## Usage

1. **Choose a data source** in the sidebar: the sample dataset or your own CSV.
2. **Select the date column and demand column** (for uploads).
3. **Set the data frequency** (Monthly, Weekly, or Daily). This determines the seasonal period (12, 52, or 7).
4. **Pick a forecasting method** and, if applicable, its settings (moving-average window or training epochs).
5. **Set the forecast horizon** and the hold-out size used for validation.
6. Review the validation metrics, forecast chart, and benchmark table, then download the forecast as CSV.

### CSV format

Any CSV with one date column and one numeric demand column works, for example:

| date       | visitors |
|------------|----------|
| 2022-01-01 | 8450     |
| 2022-02-01 | 8120     |
| 2022-03-01 | 9010     |

Rows with missing values are dropped, and data is sorted by date automatically.

## Methods at a Glance

| Method | Idea | Works best when |
|--------|------|-----------------|
| Moving Average | Averages the last *n* periods | You need a fast, simple baseline |
| Linear Trend | Fits a line to demand over time | Demand has a stable long-term trend |
| Holt-Winters | Exponential smoothing with trend and seasonality | Clear seasonal patterns, medium-sized data |
| SARIMA | Autoregressive model with seasonal differencing | Small-to-medium data where statistical rigor matters |
| Neural Network (MLP) | Learns nonlinear patterns from lagged windows | Larger datasets with complex patterns |

## Notes and Limitations

- **Seasonal models need history.** Holt-Winters and SARIMA fall back to non-seasonal fits when there are fewer than two full seasonal cycles of data.
- **The epoch accuracy chart is sensitive to dataset size.** With small datasets and short hold-out windows, validation R² can be very noisy or negative, and it is not clipped in the app. Larger hold-outs, fewer epochs, or more regularization give steadier curves.
- **Neural Network forecasts are recursive.** Each future step feeds on earlier predictions, so errors can compound over long horizons.
- **Benchmark timings are machine-specific.** Run times in the scalability table are measured live and change with hardware, dataset size, horizon, and epochs. The complexity and scalability columns are general guidance, not measurements.
- The sample dataset is synthetic and for demonstration only.

## Dependencies

`streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn`, `statsmodels`. See `requirements.txt` for versions.
