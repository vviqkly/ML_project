import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

st.set_page_config("Прогноз спроса на товары", layout="wide")
st.title("Прогноз спроса на товары")

# Функция генерации датасета

def generate_dataset():
    np.random.seed(None)
    
    n_days = 1500
    dates = pd.date_range(start="2022-01-01", periods=n_days)
    
    global_trend_strength = np.random.uniform(0.5, 1.5)
    global_noise_level = np.random.uniform(3, 8)
    global_holiday_strength = np.random.uniform(10, 25)
    
    trend_start = np.random.uniform(50, 100)
    trend_end = np.random.uniform(150, 300)
    seasonality_amp = np.random.uniform(20, 60)
    
    trend = np.linspace(trend_start, trend_start + trend_end * global_trend_strength, n_days)
    
    yearly_amp = seasonality_amp * np.random.uniform(0.8, 1.2)
    phase_shift = np.random.uniform(0, 2 * np.pi)
    yearly_seasonality = yearly_amp * np.sin(np.arange(n_days) * 2 * np.pi / 365 + phase_shift)
    
    monthly_amp = seasonality_amp * np.random.uniform(0.3, 0.6)
    monthly_seasonality = monthly_amp * np.sin(np.arange(n_days) * 2 * np.pi / 30)
    
    weekly_amp = seasonality_amp * np.random.uniform(0.2, 0.4)
    weekly_seasonality = weekly_amp * np.sin(np.arange(n_days) * 2 * np.pi / 7)
    
    seasonality = yearly_seasonality + monthly_seasonality + weekly_seasonality
    
    weekend_strength = np.random.uniform(10, 30)
    is_weekend = (dates.dayofweek >= 5).astype(int)
    weekend_effect = is_weekend * weekend_strength
    
    holiday_strength = global_holiday_strength * np.random.uniform(0.8, 1.2)
    is_new_year = (dates.month == 1) & (dates.day <= 9)
    is_women_day = (dates.month == 3) & (dates.day == 8)
    is_may_holidays = (dates.month == 5) & (dates.day >= 1) & (dates.day <= 10)
    holiday_effect = (is_new_year | is_women_day | is_may_holidays).astype(int) * holiday_strength
    
    noise = np.random.normal(0, global_noise_level, n_days)
    
    sales = trend + seasonality + weekend_effect + holiday_effect + noise
    sales = np.maximum(sales, 0)
    sales = np.round(sales, 2)
    
    df = pd.DataFrame({
        "date": dates,
        "sales": sales
    })
    
    df.to_csv("sales_data.csv", index=False)
    return df

# Функции для прогноза

def add_lags(df, lag_days=[7, 14, 30, 60]):
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
    
    features = ['sales_lag_7', 'sales_lag_14', 'sales_lag_30', 'sales_lag_60', 
                'day_of_week', 'month', 'weekend']
    
    X = df[features].values
    
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, df['sales'].values, scaler

# Функция прогнозирования 

def calculate_forecast(forecast_days=30, history_days=150):
    # Загрузка и подготовка
    df = pd.read_csv('sales_data.csv', parse_dates=['date'])
    df_with_lags = add_lags(df, lag_days=[7, 14, 30, 60])
    
    # Обучение на всех данных
    X_all, y_all, scaler = preprocess_for_lr(df_with_lags)
    
    # Обучение модели
    model = LinearRegression()
    model.fit(X_all, y_all)
    
    # Метрики
    y_all_pred = model.predict(X_all)
    
    r2 = r2_score(y_all, y_all_pred)
    mae = mean_absolute_error(y_all, y_all_pred)
    rmse = np.sqrt(mean_squared_error(y_all, y_all_pred))
    mape = np.mean(np.abs((y_all - y_all_pred) / y_all)) * 100
    
    # Прогноз на будущее
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    
    # Берем последние доступные лаги из данных
    current_lags = {
        7: df['sales'].iloc[-7],
        14: df['sales'].iloc[-14],
        30: df['sales'].iloc[-30],
        60: df['sales'].iloc[-60]
    }
    
    # Прогнозируем последовательно
    future_predictions = []
    
    for future_date in future_dates:
        row = {
            'sales_lag_7': current_lags[7],
            'sales_lag_14': current_lags[14],
            'sales_lag_30': current_lags[30],
            'sales_lag_60': current_lags[60],
            'day_of_week': future_date.dayofweek,
            'month': future_date.month,
            'weekend': 1 if future_date.dayofweek >= 5 else 0
        }
        
        features_array = np.array([[
            row['sales_lag_7'],
            row['sales_lag_14'],
            row['sales_lag_30'],
            row['sales_lag_60'],
            row['day_of_week'], 
            row['month'], 
            row['weekend']
        ]])
        features_scaled = scaler.transform(features_array)
        
        pred = model.predict(features_scaled)[0]
        future_predictions.append(pred)
        
        # Обновляем лаги для следующего шага
        current_lags = {
            7: pred,
            14: current_lags[7],
            30: current_lags[14],
            60: current_lags[30]
        }
    
    return {
        'df': df,
        'future_dates': future_dates,
        'future_predictions': future_predictions,
        'metrics': {'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape},
        'model': model,
        'scaler': scaler
    }

# Инициализация состояния

if "generated_df" not in st.session_state:
    st.session_state.generated_df = None
    
if "show_viz" not in st.session_state:
    st.session_state.show_viz = False
    
if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None

# Блок генерации датасета

st.header("1. Генерация данных")

col_btn, col_save = st.columns(2)

with col_btn:
    if st.button("Сгенерировать датасет"):
        with st.spinner("Генерация данных..."):
            st.session_state.generated_df = generate_dataset()
            st.session_state.show_viz = False
            st.session_state.forecast_result = None
        st.success("Датасет сгенерирован!")

with col_save:
    if st.session_state.generated_df is not None:
        csv_data = st.session_state.generated_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Сохранить датасет (CSV)",
            data=csv_data,
            file_name="sales_data.csv",
            mime="text/csv"
        )

# Визуализация данных

if st.session_state.generated_df is not None:
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        if st.button("Показать визуализацию данных"):
            st.session_state.show_viz = True
    
    with col_viz2:
        if st.button("Скрыть визуализацию"):
            st.session_state.show_viz = False
    
    if st.session_state.show_viz:
        df_viz = st.session_state.generated_df
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_viz["date"], df_viz["sales"], linewidth=0.5, alpha=0.7, color='green')
        ax.set_title("Сгенерированные продажи")
        ax.set_xlabel("Дата")
        ax.set_ylabel("Продажи")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.dataframe(df_viz)


st.header("2. Прогнозирование")
st.markdown("---")

# Параметры прогноза
col_params1, col_params2 = st.columns(2)

with col_params1:
    forecast_days = st.slider("Горизонт прогноза (дней)", 10, 150, 30)

with col_params2:
    history_days = st.slider("Показать историю (дней)", 30, 500, 150)

# Кнопка расчета прогноза
if st.button("Рассчитать прогноз", type="primary"):
    with st.spinner("Расчет прогноза..."):
        try:
            st.session_state.forecast_result = calculate_forecast(forecast_days, history_days)
            st.success("Прогноз рассчитан!")
        except Exception as e:
            st.error(f"Ошибка: {e}")
            st.session_state.forecast_result = None

# Отображение результатов прогноза

if st.session_state.forecast_result is not None:
    result = st.session_state.forecast_result
    
    # Метрики качества
    st.subheader("Качество модели")
    metrics = result['metrics']
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("R²", f"{metrics['r2']:.4f}")
    with col_m2:
        st.metric("MAE", f"{metrics['mae']:.2f}")
    # with col_m3:
    #     st.metric("RMSE", f"{metrics['rmse']:.2f}")
    # with col_m4:
    #     st.metric("MAPE", f"{metrics['mape']:.1f}%")
    
    # График
    st.subheader("График прогноза")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    history_df = result['df'].iloc[-history_days:]
    
    ax.plot(history_df['date'], history_df['sales'], label="Исторические данные", 
            color="blue", linewidth=1.5)
    ax.plot(result['future_dates'], result['future_predictions'], 
            label=f"Прогноз на {forecast_days} дней", 
            color="red", linestyle="--", linewidth=2, marker="o", markersize=3)
    
    ax.axvline(x=result['df']['date'].iloc[-1], color='black', linestyle=':', alpha=0.7, label='Граница')
    
    ax.set_title(f"Прогноз продаж на {forecast_days} дней")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Продажи")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Таблица с прогнозом
    st.subheader("Таблица прогноза")
    
    forecast_table = pd.DataFrame({
        'Дата': result['future_dates'],
        'Прогноз продаж': [round(x, 2) for x in result['future_predictions']]
    })
    
    st.dataframe(forecast_table)
    
    # Кнопка сохранения прогноза
    csv_forecast = forecast_table.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Сохранить прогноз (CSV)",
        data=csv_forecast,
        file_name="future_forecast.csv",
        mime="text/csv"
    )