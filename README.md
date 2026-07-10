# GridWise — Energy Intelligence Platform

A modern energy management dashboard for monitoring, analyzing, and predicting renewable energy generation. Built with FastAPI + React, featuring ML-based forecast refinement, real-time streaming, and anomaly detection.

![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-blue?style=flat-square)
![React](https://img.shields.io/badge/React-18.2.0-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-yellow?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange?style=flat-square)

---

## Features

### Dashboard & Monitoring
- **Real-time streaming** — WebSocket-fed live power, grid frequency, and load indicators
- **Wind & Solar tracking** — Hourly actual vs predicted generation charts
- **Machine consumption** — 5-machine energy and temperature monitoring
- **Anomaly detection** — Threshold-based alerts (temp > 47°C, energy > 9 kWh)

### Model Performance & Training
- **Metrics dashboard** — MAE, RMSE, MAPE, and MAPE Score (100 − MAPE%) per day and 30-day trends
- **Interactive model training** — Train Gradient Boosting or Random Forest regressors from the UI
- **R² reporting** — Train/test R² (coefficient of determination) for trained models
- **Original vs improved comparison** — Side-by-side prediction charts with MAE improvement %
- **Feature importance** — Horizontal bar charts showing learned feature weights
- **Residual analysis** — Per-hour error breakdown with color-coded severity

### AI Insights
- Rule-based energy analysis panel (generation strength, prediction quality)

---

## Architecture

```
GridWise/
├── backend/
│   ├── server.py                  # FastAPI app (all endpoints + ML training)
│   ├── models/                    # Persisted joblib models + feature importance JSON
│   ├── test_predictions.csv       # Wind: 19,656 samples (time, ActualPower, PredictedPower)
│   ├── test_solar_predictions.csv # Solar: 19,656 samples
│   ├── machine_test_data.csv      # 5 machines × (energy, temperature)
│   └── requirements.txt
├── notebooks/
│   └── eda.ipynb                  # Exploratory Data Analysis (distributions, correlations, patterns)
├── frontend/
│   ├── src/App.js                 # Single-file React SPA (all views)
│   ├── src/App.css                # Tailwind + custom styles
│   ├── package.json
│   └── public/index.html
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 14+

### Install

```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

### Run

```bash
# Terminal 1 — Backend (port 8001)
cd backend
python3 -m uvicorn server:app --host 0.0.0.0 --port 8001

# Terminal 2 — Frontend (port 3000)
cd frontend
PORT=3000 npm start
```

### Access
- Dashboard: http://localhost:3000
- API Docs (Swagger): http://localhost:8001/docs
- Health check: http://localhost:8001/api/health

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Server health check |
| GET | `/api/dates` | Available dates in dataset |
| GET | `/api/model-status` | Trained model status + metrics |
| GET | `/api/feature-importance` | Feature importance from trained models |
| GET | `/api/experiments` | Training experiment history log |
| POST | `/api/wind-prediction` | Wind power data for a date |
| POST | `/api/solar-prediction` | Solar power data for a date |
| POST | `/api/machine-consumption` | Machine energy/temp data |
| POST | `/api/dashboard-summary` | Aggregated wind + solar stats |
| POST | `/api/model-performance` | MAE, RMSE, MAPE trends + residuals |
| POST | `/api/ai-insights` | Rule-based energy insights |
| POST | `/api/train-model` | Train ML models (GB, RF, XGBoost, Linear) |
| POST | `/api/compare-models` | Train all 4 algorithms and return comparison |
| POST | `/api/train-machine-models` | Train Isolation Forest + consumption forecaster |
| POST | `/api/machine-anomalies` | Detect anomalies on a date (Isolation Forest) |
| GET | `/api/machine-model-status` | Machine model training status |
| POST | `/api/predict-with-trained` | Predictions from trained models |
| WS | `/api/ws/realtime` | Live data stream (2s interval) |

### Train Model Request

```json
{
  "model_type": "wind|solar|both",
  "algorithm": "gradient_boosting|random_forest|xgboost|linear_regression",
  "test_size": 0.2
}
```

---

## Machine Learning

### Model Approach

The trained models function as a **correction layer** on top of an existing baseline forecast. Rather than predicting power output from raw weather or time data alone, they take the baseline model's predicted power as one of their input features and learn to refine that prediction using additional time-based signals. The system's value is in reducing the residual error of a pre-existing forecast, not in producing a standalone forecast from scratch.

### Feature Engineering (16 features)

| Feature | Description |
|---------|-------------|
| `hour` | Hour of day (0–23) |
| `hour_sin`, `hour_cos` | Cyclical hour encoding (24h cycle) |
| `day_sin`, `day_cos` | Cyclical day-of-month encoding |
| `month_sin`, `month_cos` | Cyclical month encoding (seasonal) |
| `baseline_predicted_power` | Original model's prediction (key input for solar) |
| `hour_squared` | Polynomial feature |
| `lag_1`, `lag_2`, `lag_3` | Previous 1/2/3 hours' actual power |
| `lag_24` | Same hour previous day |
| `rolling_mean_6h` | Rolling mean of last 6 hours |
| `rolling_std_6h` | Rolling std of last 6 hours (volatility) |
| `rolling_mean_24h` | Rolling mean of last 24 hours |

### Model Performance

| Model | Best Algorithm | Test R² | Test MAE | Why |
|-------|---------------|---------|----------|-----|
| Wind | Linear Regression | 99.8% | 0.77 kW | Lag-1 dominance → linear is optimal |
| Solar | Gradient Boosting | 98.8% | 1.0 kW | Nonlinear weather/hour interactions |

The comparison includes 4 algorithms: Linear Regression (baseline), Gradient Boosting, Random Forest, and XGBoost. Each is evaluated on the same temporal split with training time, overfit gap, and all standard metrics.

The train/test split is **chronological** — the first 80% of timesteps are used for training, and the last 20% form the held-out test set. This prevents future data from leaking into training, which is critical for time-series forecasting.

Solar MAPE is computed on **daylight hours only** (actual > 0.5 kW) to avoid the well-known zero-division problem with nighttime solar observations.

### Outputs
- **Persisted models**: `backend/models/*.joblib` (wind, solar, isolation forest, consumption forecaster)
- **Feature importance**: `backend/models/*_feature_importance.json`
- **Evaluation metrics**: R², MAE, RMSE returned via API
- **Anomaly detection**: Isolation Forest scores per hour, multivariate (energy + temperature)

---

## Tech Stack

### Backend
| Library | Purpose |
|---------|---------|
| FastAPI 0.109 | Web framework + WebSocket |
| Uvicorn | ASGI server |
| Pandas 2.1 | Data loading and filtering |
| NumPy | Numerical computation |
| scikit-learn 1.5 | ML models (GBR, RFR, LinearRegression), metrics |
| XGBoost | Gradient boosted trees (industry standard) |
| Joblib | Model serialization |

### Frontend
| Library | Purpose |
|---------|---------|
| React 18 | UI framework |
| Recharts 2.10 | Charts (Line, Area, Pie, Bar, Composed) |
| Lucide React | Icons |
| Axios | HTTP client |
| Tailwind CSS | Utility-first styling |

---

## Configuration

| Setting | How |
|---------|-----|
| Backend port | Change in `server.py` `__main__` block or CLI flag |
| Frontend port | `PORT=3000 npm start` |
| Backend URL from frontend | Set `REACT_APP_BACKEND_URL` env var (defaults to proxy) |
| Frontend proxy | `"proxy": "http://localhost:8001"` in `package.json` |

---

## Model Training Guide

1. Navigate to the **Model Performance** tab
2. Select algorithm (Gradient Boosting recommended)
3. Click **Train Models**
4. Review R², MAE, RMSE in the training results panel
5. Compare original vs improved predictions in the charts below
6. Check **Feature Importance** bar charts to understand what the model learned

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Backend won't start | `lsof -i :8001` to check port; reinstall deps |
| Frontend won't start | `rm -rf node_modules && npm install` |
| API connection errors | Ensure backend on :8001, check CORS in server.py |
| Empty charts | Verify CSV files exist in `backend/` directory |
| Feature importance empty | Train models first via the UI or API |

---

## Sample API Call

```bash
curl -X POST http://localhost:8001/api/wind-prediction \
  -H "Content-Type: application/json" \
  -d '{"date": "02-01-2022"}'
```

```json
{
  "data": [
    { "time": "00:00", "actual": 1234.56, "predicted": 1200.34 }
  ],
  "stats": {
    "average": 1250.45,
    "maximum": 1500.23,
    "total": 30010.80
  }
}
```

---

## License

This project is provided as-is for educational and portfolio purposes.

---

**Last Updated**: July 10, 2026
**Version**: 1.1.0
