# GridWise - Energy Intelligence Platform

## Original Problem Statement
Build a modern, showcase-worthy energy management dashboard for a Data Analyst/Engineer job interview. The existing project had wind/solar power prediction, machine consumption monitoring with anomaly detection using Horizon UI/Chakra template.

**User Requirements:**
- Modern look with real-time updates
- AI-powered insights using Claude
- Full redesign with React + Tailwind
- Model Performance tab for showcasing data analytics skills
- Model Training to improve prediction accuracy

## Architecture

### Tech Stack
- **Frontend**: React 18 + Tailwind CSS + Recharts + Framer Motion
- **Backend**: FastAPI (Python) with Pandas/NumPy/Scikit-learn
- **AI**: Claude Sonnet 4.5 via Emergent LLM Key
- **ML**: Gradient Boosting & Random Forest Regressors
- **Real-time**: WebSocket for live streaming data simulation

### Data Sources (CSV)
- `test_predictions.csv` - Wind power actual vs predicted (19,656 samples)
- `test_solar_predictions.csv` - Solar power actual vs predicted (19,656 samples)
- `machine_test_data.csv` - 5 machines energy & temperature data

## User Personas
1. **Data Analyst** - Analyze energy patterns, compare predictions, train models
2. **Grid Operator** - Monitor real-time grid status, anomalies
3. **ML Engineer** - Train and evaluate prediction models
4. **Management** - High-level dashboards, energy distribution

## Core Requirements (Implemented)

### Pages
1. **Dashboard** - Combined overview with live stats, distribution chart, anomaly alerts
2. **Wind Power** - Actual vs Predicted line chart, statistics
3. **Solar Power** - Actual vs Predicted line chart, statistics
4. **Machine Consumption** - Multi-machine energy charts, temperature monitoring
5. **Model Performance** - R² and error metrics, training, trend analysis, residual charts

### Features
- [x] Real-time streaming data (WebSocket simulation)
- [x] AI-powered insights panel using Claude
- [x] Date picker for historical data
- [x] Anomaly detection alerts
- [x] Modern dark theme UI
- [x] Responsive navigation sidebar
- [x] Model Performance analytics
- [x] **Model Training** (Gradient Boosting & Random Forest)
- [x] **Original vs Improved predictions comparison**

## What's Been Implemented (Jan 31, 2026)

### Backend API Endpoints
- `GET /api/health` - Health check
- `GET /api/dates` - Available dates in dataset
- `POST /api/wind-prediction` - Wind power data
- `POST /api/solar-prediction` - Solar power data
- `POST /api/machine-consumption` - Machine energy/temp data
- `POST /api/dashboard-summary` - Aggregated stats
- `POST /api/ai-insights` - Claude AI analysis
- `POST /api/model-performance` - Model metrics (MAE, RMSE, MAPE, R² scores)
- `POST /api/train-model` - **NEW** Train ML models
- `GET /api/model-status` - **NEW** Check trained model status
- `POST /api/predict-with-trained` - **NEW** Get improved predictions
- `WebSocket /api/ws/realtime` - Live data streaming

### Model Training Features (NEW)
- Algorithm selection: Gradient Boosting or Random Forest
- 80/20 train/test split
- R² (coefficient of determination) calculation
- Wind model achieves **99.7% R²** on test set with temporal split
- Solar model achieves **98.8% R²** on test set with temporal split
- Solar model needs more feature engineering
- Original vs Improved prediction comparison charts
- MAE improvement percentage display
- Model persistence with joblib

### ML Feature Engineering
- Hour of day (0-23)
- Hour sin/cos encoding (cyclical)
- Day of month sin encoding
- Month sin encoding
- Original predicted power as feature
- Hour squared (polynomial)

## P0/P1/P2 Features Remaining

### P1 (Nice to Have)
- [ ] Export data to CSV/PDF
- [ ] Add more features for solar model (weather data)
- [ ] Hyperparameter tuning UI
- [ ] Cross-validation support

### P2 (Future)
- [ ] User authentication
- [ ] Alert notification system
- [ ] LSTM/Neural Network models
- [ ] Auto-ML model selection

## Next Tasks
1. Improve solar model with more features
2. Add weather data integration
3. Implement hyperparameter tuning
4. Add model export/import functionality
