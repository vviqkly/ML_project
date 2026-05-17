import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sint import generate_dataset
from linear_regression import calculate_forecast

st.set_page_config("Прогноз спроса на товары", layout="wide")
st.title("Прогноз спроса на товары")

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

# Блок прогнозирования

st.header("2. Прогнозирование")
st.markdown("---")

# Параметры прогноза
col_params1, col_params2 = st.columns(2)

with col_params1:
    forecast_days = st.slider("Горизонт прогноза (дней)", 10, 100, 30)

with col_params2:
    history_days = st.slider("Показать историю (дней)", 30, 500, 150)

# Кнопка расчета прогноза
if st.button("Рассчитать прогноз", type="primary"):
    with st.spinner("Расчет прогноза..."):
        try:
            st.session_state.forecast_result = calculate_forecast(forecast_days)
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
    
    # График прогноза
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