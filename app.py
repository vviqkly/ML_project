import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

history_df = pd.read_csv("sales_data.csv", parse_dates=["date"])
forecast_df = pd.read_csv("future_forecast.csv", parse_dates=["date"])

st.set_page_config("Прогноз спроса на товары")
st.title("Прогноз спроса на товары")

# Слайдеры для выбора временного отрезка
history_days = st.sidebar.slider("История (дней)", 30, 200, 60)
forecast_days = st.sidebar.slider("Прогноз (дней)", 7, 100, 14)

# Загрузка собственного csv-файла
uploaded = st.file_uploader("Загрузите свой CSV", type="csv")

if uploaded is not None:
    history_df = pd.read_csv(uploaded, parse_dates=["date"])
    st.success("Используем ваш файл")
else:
    st.info("Используем стандартные данные")

history_slice = history_df.tail(history_days)
forecast_slice = forecast_df.head(forecast_days)

# График
fig, ax = plt.subplots(figsize=(16, 7))

ax.plot(history_slice["date"], history_slice["sales"], linewidth = 3, color = "blue", label = "История продаж")

ax.plot(forecast_slice["date"], forecast_slice["forecast"], linewidth = 3, color = "red", label = "Прогноз")

ax.axvline(x=history_df["date"].iloc[-1], linewidth = 3, color="black", linestyle=":")

ax.set_xlabel("Дата")
ax.set_ylabel("Продажи (шт)")
ax.grid(True, alpha=0.3)
ax.legend()
 
st.pyplot(fig)

# Таблица
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Прогноз на завтра", f"{forecast_df['forecast'].iloc[0]:.0f} шт")

with col2:
    st.metric("Среднее за период", f"{forecast_slice['forecast'].mean():.0f} шт")

with col3:
    st.metric("Последние продажи", f"{history_df['sales'].iloc[-1]:.0f} шт")
