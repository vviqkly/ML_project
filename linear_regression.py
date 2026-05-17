import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

def add_lags(df, lag_days=[7, 14, 30]):
    df = df.copy()
    for lag in lag_days:
        df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
    df = df.dropna()
    return df

def preprocess_for_lr(df, scaler=None):
    df = df.copy()
    
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    
    features = ['sales_lag_7', 'sales_lag_14', 'sales_lag_30', 
                'day_of_week', 'month', 'weekend']
    
    X = df[features].values
    
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, df['sales'].values, scaler

def calculate_forecast(forecast_days=30):
    # Загрузка и подготовка
    df = pd.read_csv('sales_data.csv', parse_dates=['date'])
    df_with_lags = add_lags(df, lag_days=[7, 14, 30])
    
    # Подготовка данных
    X_all, y_all, scaler = preprocess_for_lr(df_with_lags)
    
    # КРОСС-ВАЛИДАЦИЯ (TimeSeriesSplit для временных рядов)
    tscv = TimeSeriesSplit(n_splits=5, test_size=50)
    model_cv = Ridge(alpha=0.5)
    
    cv_r2_scores = []
    cv_mae_scores = []
    cv_rmse_scores = []
    cv_mape_scores = []
    
    for train_idx, val_idx in tscv.split(X_all):
        X_train_fold = X_all[train_idx]
        y_train_fold = y_all[train_idx]
        X_val_fold = X_all[val_idx]
        y_val_fold = y_all[val_idx]
        
        model_cv.fit(X_train_fold, y_train_fold)
        y_val_pred = model_cv.predict(X_val_fold)
        
        cv_r2_scores.append(r2_score(y_val_fold, y_val_pred))
        cv_mae_scores.append(mean_absolute_error(y_val_fold, y_val_pred))
        cv_rmse_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_val_pred)))
        cv_mape_scores.append(np.mean(np.abs((y_val_fold - y_val_pred) / y_val_fold)) * 100)
    
    # разделение на train/test
    split_idx = int(len(X_all) * 0.7)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]
    
    # Обучение модели на train
    model = Ridge(alpha=0.5)
    model.fit(X_train, y_train)
    
    # Метрики на test
    y_test_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    # Прогноз на будущее (на всех данных)

    # Переобучаем модель на всех данных
    model_full = Ridge(alpha=0.01)
    model_full.fit(X_all, y_all)
    
    # Прогноз на будущее
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    
    # Берем последние доступные лаги из данных
    current_lags = {
        7: df['sales'].iloc[-7],
        14: df['sales'].iloc[-14],
        30: df['sales'].iloc[-30],
    }
    
    # Прогнозируем последовательно
    future_predictions = []
    
    for future_date in future_dates:
        row = {
            'sales_lag_7': current_lags[7],
            'sales_lag_14': current_lags[14],
            'sales_lag_30': current_lags[30],
            'day_of_week': future_date.dayofweek,
            'month': future_date.month,
            'weekend': 1 if future_date.dayofweek >= 5 else 0
        }
        
        features_array = np.array([[
            row['sales_lag_7'],
            row['sales_lag_14'],
            row['sales_lag_30'],
            row['day_of_week'], 
            row['month'], 
            row['weekend']
        ]])
        features_scaled = scaler.transform(features_array)
        
        pred = model_full.predict(features_scaled)[0]
        # pred = min(pred, max(y_all) * 3)
        pred = max(0, pred)
        future_predictions.append(pred)
        
        # Обновляем лаги для следующего шага
        current_lags = {
            7: pred,
            14: current_lags[7],
            30: current_lags[14],
        }
    
    return {
        'df': df,
        'future_dates': future_dates,
        'future_predictions': future_predictions,
        'metrics': {'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape},
        'cv_metrics': {
            'r2_mean': np.mean(cv_r2_scores),
            'r2_std': np.std(cv_r2_scores),
            'mae_mean': np.mean(cv_mae_scores),
            'mae_std': np.std(cv_mae_scores),
            'rmse_mean': np.mean(cv_rmse_scores),
            'rmse_std': np.std(cv_rmse_scores),
            'mape_mean': np.mean(cv_mape_scores),
            'mape_std': np.std(cv_mape_scores)
        },
        'model': model_full,
        'scaler': scaler
    }