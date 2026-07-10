from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import random
import json
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

load_dotenv(override=False)

app = FastAPI(title="GridWise API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Store trained models in memory
trained_models = {
    'wind': None,
    'solar': None,
    'wind_metrics': None,
    'solar_metrics': None
}

def load_csv_data():
    wind_df = pd.read_csv(os.path.join(BASE_DIR, 'test_predictions.csv'))
    solar_df = pd.read_csv(os.path.join(BASE_DIR, 'test_solar_predictions.csv'))
    machine_df = pd.read_csv(os.path.join(BASE_DIR, 'machine_test_data.csv'))
    return wind_df, solar_df, machine_df

wind_data, solar_data, machine_data = load_csv_data()

def filter_by_date(df, date_str):
    df_copy = df.copy()
    df_copy['parsed_date'] = df_copy['time'].apply(lambda x: x.split(' ')[0] if ' ' in str(x) else x)
    filtered = df_copy[df_copy['parsed_date'] == date_str]
    return filtered

def extract_features(df):
    """Extract features for ML training"""
    features = []
    targets = []
    
    for _, row in df.iterrows():
        time_str = str(row['time'])
        if ' ' in time_str:
            date_part, time_part = time_str.split(' ')
            hour = int(time_part.split(':')[0])
            day_parts = date_part.split('-')
            day = int(day_parts[0])
            month = int(day_parts[1])
        else:
            hour = 0
            day = 1
            month = 1
        
        # Feature engineering
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 31)
        month_sin = np.sin(2 * np.pi * month / 12)
        
        # Use predicted power as a feature (existing model's prediction)
        predicted = float(row['PredictedPower'])
        
        features.append([hour, hour_sin, hour_cos, day_sin, month_sin, predicted, hour**2])
        targets.append(float(row['ActualPower']) * 100)
    
    return np.array(features), np.array(targets)

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "GridWise API"}

@app.get("/api/dates")
async def get_available_dates():
    """Get available dates from the dataset"""
    dates = wind_data['time'].apply(lambda x: x.split(' ')[0] if ' ' in str(x) else x).unique().tolist()[:30]
    return {"dates": dates}

@app.post("/api/wind-prediction")
async def get_wind_prediction(data: dict):
    date_str = data.get('date', '02-01-2022')
    filtered = filter_by_date(wind_data, date_str)
    
    if filtered.empty:
        return {"error": "No data for date", "data": [], "stats": {}}
    
    records = []
    for _, row in filtered.iterrows():
        time_parts = str(row['time']).split(' ')
        hour = time_parts[1] if len(time_parts) > 1 else '00:00'
        records.append({
            "time": hour,
            "actual": round(float(row['ActualPower']) * 100, 2),
            "predicted": round(float(row['PredictedPower']) * 100, 2)
        })
    
    actual_values = [r['actual'] for r in records]
    return {
        "data": records,
        "stats": {
            "average": round(sum(actual_values) / len(actual_values), 2) if actual_values else 0,
            "maximum": round(max(actual_values), 2) if actual_values else 0,
            "total": round(sum(actual_values), 2) if actual_values else 0
        }
    }

@app.post("/api/solar-prediction")
async def get_solar_prediction(data: dict):
    date_str = data.get('date', '02-01-2022')
    filtered = filter_by_date(solar_data, date_str)
    
    if filtered.empty:
        return {"error": "No data for date", "data": [], "stats": {}}
    
    records = []
    for _, row in filtered.iterrows():
        time_parts = str(row['time']).split(' ')
        hour = time_parts[1] if len(time_parts) > 1 else '00:00'
        records.append({
            "time": hour,
            "actual": round(float(row['ActualPower']) * 100, 2),
            "predicted": round(float(row['PredictedPower']) * 100, 2)
        })
    
    actual_values = [r['actual'] for r in records]
    return {
        "data": records,
        "stats": {
            "average": round(sum(actual_values) / len(actual_values), 2) if actual_values else 0,
            "maximum": round(max(actual_values), 2) if actual_values else 0,
            "total": round(sum(actual_values), 2) if actual_values else 0
        }
    }

@app.post("/api/machine-consumption")
async def get_machine_consumption(data: dict):
    date_str = data.get('date', '01-01-2023')
    filtered = filter_by_date(machine_data, date_str)
    
    if filtered.empty:
        return {"error": "No data for date", "data": [], "stats": {}}
    
    records = []
    total_consumption = 0
    anomalies = []
    
    for _, row in filtered.iterrows():
        time_parts = str(row['time']).split(' ')
        hour = time_parts[1] if len(time_parts) > 1 else '00:00'
        
        record = {"time": hour}
        for i in range(1, 6):
            energy_col = f'Machine_{i} Energy Consumed (kWh)'
            temp_col = f'Machine_{i} Temperature (C)'
            if energy_col in row and temp_col in row:
                energy = float(row[energy_col])
                temp = float(row[temp_col])
                record[f'machine{i}_energy'] = round(energy, 2)
                record[f'machine{i}_temp'] = round(temp, 2)
                total_consumption += energy
                
                # Check for anomalies (high temp > 47C or high energy > 9 kWh)
                if temp > 47 or energy > 9:
                    anomalies.append({
                        "machine": f"Machine {i}",
                        "time": hour,
                        "type": "High Temperature" if temp > 47 else "High Energy",
                        "value": round(temp if temp > 47 else energy, 2)
                    })
        records.append(record)
    
    return {
        "data": records,
        "stats": {
            "total_consumption": round(total_consumption, 2),
            "anomaly_count": len(anomalies),
            "anomalies": anomalies[:5]  # Top 5 anomalies
        }
    }

@app.post("/api/dashboard-summary")
async def get_dashboard_summary(data: dict):
    date_str = data.get('date', '02-01-2022')
    
    # Get wind data
    wind_filtered = filter_by_date(wind_data, date_str)
    wind_total = wind_filtered['ActualPower'].sum() * 100 if not wind_filtered.empty else 0
    
    # Get solar data
    solar_filtered = filter_by_date(solar_data, date_str)
    solar_total = solar_filtered['ActualPower'].sum() * 100 if not solar_filtered.empty else 0
    
    combined_total = wind_total + solar_total
    
    return {
        "wind_total": round(wind_total, 2),
        "solar_total": round(solar_total, 2),
        "combined_total": round(combined_total, 2),
        "wind_percentage": round((wind_total / combined_total * 100) if combined_total > 0 else 0, 1),
        "solar_percentage": round((solar_total / combined_total * 100) if combined_total > 0 else 0, 1)
    }

@app.post("/api/model-performance")
async def get_model_performance(data: dict):
    """Calculate comprehensive model performance metrics"""
    # Get all available dates for trend analysis
    dates = wind_data['time'].apply(lambda x: x.split(' ')[0] if ' ' in str(x) else x).unique().tolist()[:30]
    
    wind_metrics_trend = []
    solar_metrics_trend = []
    
    for date_str in dates:
        wind_filtered = filter_by_date(wind_data, date_str)
        solar_filtered = filter_by_date(solar_data, date_str)
        
        if not wind_filtered.empty:
            actual = wind_filtered['ActualPower'].values * 100
            predicted = wind_filtered['PredictedPower'].values * 100
            
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mape = np.mean(np.abs((actual - predicted) / (actual + 0.001))) * 100
            mape_score = max(0, 100 - mape)
            
            wind_metrics_trend.append({
                "date": date_str,
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "mape": round(mape, 2),
                "mape_score": round(mape_score, 1)
            })
        
        if not solar_filtered.empty:
            actual = solar_filtered['ActualPower'].values * 100
            predicted = solar_filtered['PredictedPower'].values * 100
            
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mape = np.mean(np.abs((actual - predicted) / (actual + 0.001))) * 100
            mape_score = max(0, 100 - mape)
            
            solar_metrics_trend.append({
                "date": date_str,
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "mape": round(mape, 2),
                "mape_score": round(mape_score, 1)
            })
    
    # Calculate overall metrics
    wind_overall = {
        "avg_mae": round(np.mean([m['mae'] for m in wind_metrics_trend]), 2) if wind_metrics_trend else 0,
        "avg_rmse": round(np.mean([m['rmse'] for m in wind_metrics_trend]), 2) if wind_metrics_trend else 0,
        "avg_mape": round(np.mean([m['mape'] for m in wind_metrics_trend]), 2) if wind_metrics_trend else 0,
        "avg_mape_score": round(np.mean([m['mape_score'] for m in wind_metrics_trend]), 1) if wind_metrics_trend else 0,
        "best_day": max(wind_metrics_trend, key=lambda x: x['mape_score'])['date'] if wind_metrics_trend else "",
        "worst_day": min(wind_metrics_trend, key=lambda x: x['mape_score'])['date'] if wind_metrics_trend else ""
    }
    
    solar_overall = {
        "avg_mae": round(np.mean([m['mae'] for m in solar_metrics_trend]), 2) if solar_metrics_trend else 0,
        "avg_rmse": round(np.mean([m['rmse'] for m in solar_metrics_trend]), 2) if solar_metrics_trend else 0,
        "avg_mape": round(np.mean([m['mape'] for m in solar_metrics_trend]), 2) if solar_metrics_trend else 0,
        "avg_mape_score": round(np.mean([m['mape_score'] for m in solar_metrics_trend]), 1) if solar_metrics_trend else 0,
        "best_day": max(solar_metrics_trend, key=lambda x: x['mape_score'])['date'] if solar_metrics_trend else "",
        "worst_day": min(solar_metrics_trend, key=lambda x: x['mape_score'])['date'] if solar_metrics_trend else ""
    }
    
    # Residuals for selected date (hourly breakdown)
    selected_date = data.get('date', '02-01-2022')
    wind_filtered = filter_by_date(wind_data, selected_date)
    solar_filtered = filter_by_date(solar_data, selected_date)
    
    wind_residuals = []
    solar_residuals = []
    
    for _, row in wind_filtered.iterrows():
        time_parts = str(row['time']).split(' ')
        hour = time_parts[1] if len(time_parts) > 1 else '00:00'
        actual = float(row['ActualPower']) * 100
        predicted = float(row['PredictedPower']) * 100
        wind_residuals.append({
            "time": hour,
            "actual": round(actual, 2),
            "predicted": round(predicted, 2),
            "residual": round(actual - predicted, 2),
            "error_pct": round(abs(actual - predicted) / (actual + 0.001) * 100, 1)
        })
    
    for _, row in solar_filtered.iterrows():
        time_parts = str(row['time']).split(' ')
        hour = time_parts[1] if len(time_parts) > 1 else '00:00'
        actual = float(row['ActualPower']) * 100
        predicted = float(row['PredictedPower']) * 100
        solar_residuals.append({
            "time": hour,
            "actual": round(actual, 2),
            "predicted": round(predicted, 2),
            "residual": round(actual - predicted, 2),
            "error_pct": round(abs(actual - predicted) / (actual + 0.001) * 100, 1)
        })
    
    return {
        "wind_trend": wind_metrics_trend,
        "solar_trend": solar_metrics_trend,
        "wind_overall": wind_overall,
        "solar_overall": solar_overall,
        "wind_residuals": wind_residuals,
        "solar_residuals": solar_residuals,
        "selected_date": selected_date
    }

@app.post("/api/train-model")
async def train_model(data: dict):
    """Train improved prediction models using Gradient Boosting"""
    global trained_models
    
    model_type = data.get('model_type', 'both')  # 'wind', 'solar', or 'both'
    algorithm = data.get('algorithm', 'gradient_boosting')  # 'gradient_boosting' or 'random_forest'
    test_size = data.get('test_size', 0.2)
    
    results = {}
    
    # Train Wind Model
    if model_type in ['wind', 'both']:
        X_wind, y_wind = extract_features(wind_data)
        X_train, X_test, y_train, y_test = train_test_split(X_wind, y_wind, test_size=test_size, random_state=42)
        
        if algorithm == 'gradient_boosting':
            wind_model = GradientBoostingRegressor(
                n_estimators=100, 
                max_depth=5, 
                learning_rate=0.1,
                random_state=42
            )
        else:
            wind_model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                random_state=42
            )
        
        wind_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = wind_model.predict(X_train)
        y_pred_test = wind_model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Calculate R² (coefficient of determination)
        train_ss_res = np.sum((y_train - y_pred_train) ** 2)
        train_ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
        train_r2 = max(0, 1 - (train_ss_res / train_ss_tot)) * 100
        
        test_ss_res = np.sum((y_test - y_pred_test) ** 2)
        test_ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        test_r2 = max(0, 1 - (test_ss_res / test_ss_tot)) * 100
        
        # Store model
        trained_models['wind'] = wind_model
        trained_models['wind_metrics'] = {
            'train_mae': round(train_mae, 2),
            'test_mae': round(test_mae, 2),
            'train_rmse': round(train_rmse, 2),
            'test_rmse': round(test_rmse, 2),
            'train_r2': round(train_r2, 1),
            'test_r2': round(test_r2, 1),
            'samples_train': len(X_train),
            'samples_test': len(X_test)
        }
        
        # Save model to disk
        joblib.dump(wind_model, os.path.join(MODELS_DIR, 'wind_model.joblib'))
        
        # Save feature importance
        wind_importances = [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in sorted(zip(FEATURE_NAMES, wind_model.feature_importances_), key=lambda x: x[1], reverse=True)
        ]
        with open(os.path.join(MODELS_DIR, 'wind_feature_importance.json'), 'w') as f:
            json.dump(wind_importances, f, indent=2)
        
        results['wind'] = trained_models['wind_metrics']
    
    # Train Solar Model
    if model_type in ['solar', 'both']:
        X_solar, y_solar = extract_features(solar_data)
        X_train, X_test, y_train, y_test = train_test_split(X_solar, y_solar, test_size=test_size, random_state=42)
        
        if algorithm == 'gradient_boosting':
            solar_model = GradientBoostingRegressor(
                n_estimators=100, 
                max_depth=5, 
                learning_rate=0.1,
                random_state=42
            )
        else:
            solar_model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                random_state=42
            )
        
        solar_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = solar_model.predict(X_train)
        y_pred_test = solar_model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Calculate R² (coefficient of determination)
        train_ss_res = np.sum((y_train - y_pred_train) ** 2)
        train_ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
        train_r2 = max(0, 1 - (train_ss_res / train_ss_tot)) * 100
        
        test_ss_res = np.sum((y_test - y_pred_test) ** 2)
        test_ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        test_r2 = max(0, 1 - (test_ss_res / test_ss_tot)) * 100
        
        # Store model
        trained_models['solar'] = solar_model
        trained_models['solar_metrics'] = {
            'train_mae': round(train_mae, 2),
            'test_mae': round(test_mae, 2),
            'train_rmse': round(train_rmse, 2),
            'test_rmse': round(test_rmse, 2),
            'train_r2': round(train_r2, 1),
            'test_r2': round(test_r2, 1),
            'samples_train': len(X_train),
            'samples_test': len(X_test)
        }
        
        # Save model to disk
        joblib.dump(solar_model, os.path.join(MODELS_DIR, 'solar_model.joblib'))
        
        # Save feature importance
        solar_importances = [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in sorted(zip(FEATURE_NAMES, solar_model.feature_importances_), key=lambda x: x[1], reverse=True)
        ]
        with open(os.path.join(MODELS_DIR, 'solar_feature_importance.json'), 'w') as f:
            json.dump(solar_importances, f, indent=2)
        
        results['solar'] = trained_models['solar_metrics']
    
    return {
        "status": "success",
        "algorithm": algorithm,
        "results": results,
        "message": f"Models trained successfully using {algorithm.replace('_', ' ').title()}"
    }

@app.get("/api/model-status")
async def get_model_status():
    """Get status of trained models"""
    return {
        "wind_trained": trained_models['wind'] is not None,
        "solar_trained": trained_models['solar'] is not None,
        "wind_metrics": trained_models['wind_metrics'],
        "solar_metrics": trained_models['solar_metrics']
    }

@app.post("/api/predict-with-trained")
async def predict_with_trained(data: dict):
    """Get predictions using the trained models"""
    date_str = data.get('date', '02-01-2022')
    
    wind_filtered = filter_by_date(wind_data, date_str)
    solar_filtered = filter_by_date(solar_data, date_str)
    
    wind_predictions = []
    solar_predictions = []
    
    # Wind predictions
    if trained_models['wind'] is not None and not wind_filtered.empty:
        X_wind, y_actual = extract_features(wind_filtered)
        y_pred = trained_models['wind'].predict(X_wind)
        
        for i, (_, row) in enumerate(wind_filtered.iterrows()):
            time_parts = str(row['time']).split(' ')
            hour = time_parts[1] if len(time_parts) > 1 else '00:00'
            wind_predictions.append({
                "time": hour,
                "actual": round(y_actual[i], 2),
                "original_pred": round(float(row['PredictedPower']) * 100, 2),
                "improved_pred": round(y_pred[i], 2)
            })
    
    # Solar predictions
    if trained_models['solar'] is not None and not solar_filtered.empty:
        X_solar, y_actual = extract_features(solar_filtered)
        y_pred = trained_models['solar'].predict(X_solar)
        
        for i, (_, row) in enumerate(solar_filtered.iterrows()):
            time_parts = str(row['time']).split(' ')
            hour = time_parts[1] if len(time_parts) > 1 else '00:00'
            solar_predictions.append({
                "time": hour,
                "actual": round(y_actual[i], 2),
                "original_pred": round(float(row['PredictedPower']) * 100, 2),
                "improved_pred": round(y_pred[i], 2)
            })
    
    # Calculate improvement metrics
    wind_improvement = None
    solar_improvement = None
    
    if wind_predictions:
        original_mae = np.mean([abs(p['actual'] - p['original_pred']) for p in wind_predictions])
        improved_mae = np.mean([abs(p['actual'] - p['improved_pred']) for p in wind_predictions])
        wind_improvement = {
            "original_mae": round(original_mae, 2),
            "improved_mae": round(improved_mae, 2),
            "improvement_pct": round((original_mae - improved_mae) / original_mae * 100, 1) if original_mae > 0 else 0
        }
    
    if solar_predictions:
        original_mae = np.mean([abs(p['actual'] - p['original_pred']) for p in solar_predictions])
        improved_mae = np.mean([abs(p['actual'] - p['improved_pred']) for p in solar_predictions])
        solar_improvement = {
            "original_mae": round(original_mae, 2),
            "improved_mae": round(improved_mae, 2),
            "improvement_pct": round((original_mae - improved_mae) / original_mae * 100, 1) if original_mae > 0 else 0
        }
    
    return {
        "wind_predictions": wind_predictions,
        "solar_predictions": solar_predictions,
        "wind_improvement": wind_improvement,
        "solar_improvement": solar_improvement,
        "date": date_str
    }

FEATURE_NAMES = ["hour", "hour_sin", "hour_cos", "day_sin", "month_sin", "baseline_predicted_power", "hour_squared"]

@app.get("/api/feature-importance")
async def get_feature_importance():
    """Return feature importances from trained models"""
    result = {}
    
    if trained_models['wind'] is not None:
        importances = trained_models['wind'].feature_importances_
        result['wind'] = [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
        ]
    
    if trained_models['solar'] is not None:
        importances = trained_models['solar'].feature_importances_
        result['solar'] = [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
        ]
    
    if not result:
        return {"error": "No trained models available. Train a model first via /api/train-model.", "wind": None, "solar": None}
    
    return result

@app.post("/api/ai-insights")
async def get_ai_insights(data: dict):
    """Generate AI insights - simplified fallback version"""
    try:
        date_str = data.get('date', '02-01-2022')
        context = data.get('context', 'general')
        
        # Gather data for analysis
        wind_filtered = filter_by_date(wind_data, date_str)
        solar_filtered = filter_by_date(solar_data, date_str)
        
        wind_avg = wind_filtered['ActualPower'].mean() * 100 if not wind_filtered.empty else 0
        solar_avg = solar_filtered['ActualPower'].mean() * 100 if not solar_filtered.empty else 0
        wind_pred_accuracy = 0
        solar_pred_accuracy = 0
        
        if not wind_filtered.empty:
            wind_pred_accuracy = 100 - (abs(wind_filtered['ActualPower'] - wind_filtered['PredictedPower']).mean() / wind_filtered['ActualPower'].mean() * 100)
        if not solar_filtered.empty:
            solar_pred_accuracy = 100 - (abs(solar_filtered['ActualPower'] - solar_filtered['PredictedPower']).mean() / solar_filtered['ActualPower'].mean() * 100)
        
        # Generate rule-based insights
        insights = []
        
        if wind_avg > 60:
            insights.append(f"Wind generation is strong at {wind_avg:.1f} kW.")
        elif wind_avg < 30:
            insights.append(f"Wind generation is low at {wind_avg:.1f} kW.")
        else:
            insights.append(f"Wind generation is moderate at {wind_avg:.1f} kW.")
        
        if solar_avg > 60:
            insights.append(f"Solar generation is strong at {solar_avg:.1f} kW.")
        elif solar_avg < 30:
            insights.append(f"Solar generation is low at {solar_avg:.1f} kW.")
        else:
            insights.append(f"Solar generation is moderate at {solar_avg:.1f} kW.")
        
        if wind_pred_accuracy > 80:
            insights.append("Wind prediction model is performing well.")
        elif wind_pred_accuracy < 60:
            insights.append("Wind prediction model needs improvement.")
        
        insight_text = " ".join(insights)
        
        return {"insight": insight_text, "date": date_str}
    except Exception as e:
        return {"insight": "Analysis suggests wind and solar generation are within normal parameters. Consider reviewing prediction models for improved accuracy.", "error": str(e)}

# WebSocket for real-time data streaming
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/api/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Simulate real-time data updates
            live_data = {
                "timestamp": datetime.now().isoformat(),
                "wind_power": round(random.uniform(20, 80), 2),
                "solar_power": round(random.uniform(30, 70), 2),
                "grid_frequency": round(random.uniform(49.95, 50.05), 3),
                "total_load": round(random.uniform(100, 200), 2)
            }
            await websocket.send_json(live_data)
            await asyncio.sleep(2)  # Update every 2 seconds
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
