# Task 03 . Predictive Analytics Using Historical Data

**Internship:** Thiranex Data Analytics Internship  
**Task:** Build a predictive model to forecast future trends

---

## Objective

Use historical sales data (reused from Task 01) to build and compare predictive
models that forecast future revenue trends, evaluate their accuracy, and
visualize predictions against actuals.

## Features

- Aggregate transaction-level sales data into a daily revenue time series
- Clean and preprocess: fill missing calendar dates, log-transform to handle
  exponential growth
- Exploratory analysis: trend, weekly seasonality, monthly totals
- **Time-series model:** SARIMA (statsmodels) — trend + weekly seasonality
- **Regression model:** Log-Linear Regression (scikit-learn) — trend + day-of-week features
- Model evaluation with MAE, RMSE, and MAPE on a held-out test period
- Side-by-side visualization of actual vs. predicted values
- 30-day future forecast beyond the historical data range

## Tech Stack

- Language: Python 3
- Libraries: pandas, numpy, statsmodels, scikit-learn, matplotlib, seaborn
- Notebook: Jupyter Notebook

## Project Structure

```
Task-03-Predictive-Analytics/
├── notebooks/
│   └── predictive_analytics.ipynb   # Main analysis & modeling notebook
├── data/
│   └── raw_sales_data.csv           # Historical dataset (reused from Task 01)
├── outputs/
│   ├── 01_revenue_trend.png
│   ├── 02_weekly_seasonality.png
│   ├── 03_monthly_revenue.png
│   ├── 04_actual_vs_predicted.png
│   ├── 05_prediction_scatter.png
│   ├── 06_future_forecast.png
│   └── future_30_day_forecast.csv   # Exported 30-day forecast from both models
└── README.md
```

## How to Run

1. Clone / open the project folder
2. Install dependencies:
   ```
   pip install pandas numpy statsmodels scikit-learn matplotlib seaborn jupyter
   ```
3. Launch Jupyter:
   ```
   jupyter notebook notebooks/predictive_analytics.ipynb
   ```
4. Run all cells top to bottom

## Analysis Steps

| Step | Description |
|---|---|
| Data Loading | Import Task 01 transaction data, aggregate to daily revenue |
| Preprocessing | Fill missing dates, log-transform for exponential growth |
| EDA | Trend plot, day-of-week boxplot, monthly totals |
| Train/Test Split | Last 30 days held out, chronological (no shuffling) |
| Model A | SARIMA — time-series forecasting with weekly seasonality |
| Model B | Log-Linear Regression — trend + day-of-week dummies |
| Evaluation | MAE, RMSE, MAPE compared side by side |
| Visualization | Actual vs. predicted line & scatter plots |
| Forecasting | 30-day forecast beyond available historical data |

## Results Summary (Held-Out 30-Day Test Period)

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| SARIMA (time-series) | ~58.2M | ~66.5M | ~32.7% |
| Log-Linear Regression | ~34.7M | ~40.1M | ~20.6% |

**Winner: Log-Linear Regression** — lower error across all three metrics.
The log-linear regression model generalized better on this dataset because
daily revenue grows close to exponentially across the year — once
log-transformed, that growth becomes nearly linear, which plays directly to
linear regression's strengths. SARIMA still captures useful local
autocorrelation and weekly seasonality, and both models agree directionally
on continued growth in the 30-day future forecast.

### Notes on the Forecast Divergence

The two models project a noticeably different range for the 30-day future
forecast (Dec 29 – Jan 27): the regression model trends toward the higher end
(~$178M–$382M), while SARIMA stays more moderate (~$179M–$305M). This is the
expected signature of their different assumptions:

- **Log-linear regression** extrapolates a *constant* exponential growth rate
  indefinitely — it keeps compounding upward with no resistance, which can
  overshoot if the underlying growth rate itself is decelerating.
- **SARIMA** is autoregressive, so it partially "remembers" the more recent
  (somewhat slower) daily values and tempers its slope — more conservative,
  but can lag behind a strong trend.

In practice this is a classic regression-vs-time-series tradeoff worth
stating explicitly in any forecast handed to stakeholders: regression gives
a cleaner trend extrapolation, SARIMA gives a more cautious one that adapts
faster if the trend bends.

### Resolved: SARIMA Convergence Warning

The full-data SARIMA refit (used for the 30-day future forecast) initially
emitted a *"Maximum Likelihood optimization failed to converge"* warning from
`statsmodels` — a numerical optimizer settling for a near-optimum within its
default iteration budget. This did not invalidate the forecast, but it has
been resolved by raising the iteration limit and switching to the Powell
optimizer: `fit(disp=False, maxiter=200, method='powell')`. Both SARIMA fit
cells in the notebook now use this, and re-running produces no convergence
warning.

## Key Learnings

- Predictive modeling using both time-series (SARIMA) and regression approaches
- Log-transformation for stabilizing variance on exponentially growing data
- Handling missing dates in time-series data via interpolation
- Chronological train/test splitting (no data leakage from shuffling)
- Model evaluation with MAE, RMSE, and MAPE
- Communicating forecast accuracy visually (actual vs. predicted, scatter plots)
- Forecasting beyond the historical data range for business planning
- Diagnosing and resolving optimizer convergence warnings in statistical models
# thiranex-task03
