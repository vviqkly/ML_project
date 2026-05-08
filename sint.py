import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(None)  # убираем фиксированный seed для случайности

n_days = 750
dates = pd.date_range(start="2024-01-01", periods=n_days)

# Рандомизация глобальных параметров
global_trend_strength = np.random.uniform(0.5, 1.5)   # сила общего тренда
global_noise_level = np.random.uniform(3, 8)          # уровень шума
global_holiday_strength = np.random.uniform(10, 25)   # сила праздничного эффекта

# Параметры для временного ряда
trend_start = np.random.uniform(50, 100)
trend_end = np.random.uniform(150, 300)
seasonality_amp = np.random.uniform(20, 60)

# ========== 1. ТРЕНД ==========
trend = np.linspace(trend_start, trend_start + trend_end * global_trend_strength, n_days)

# ========== 2. СЕЗОННОСТЬ ==========
# Годовая сезонность
yearly_amp = seasonality_amp * np.random.uniform(0.8, 1.2)
phase_shift = np.random.uniform(0, 2 * np.pi)
yearly_seasonality = yearly_amp * np.sin(np.arange(n_days) * 2 * np.pi / 365 + phase_shift)

# Месячная сезонность
monthly_amp = seasonality_amp * np.random.uniform(0.3, 0.6)
monthly_seasonality = monthly_amp * np.sin(np.arange(n_days) * 2 * np.pi / 30)

# Недельная сезонность
weekly_amp = seasonality_amp * np.random.uniform(0.2, 0.4)
weekly_seasonality = weekly_amp * np.sin(np.arange(n_days) * 2 * np.pi / 7)

# Общая сезонность
seasonality = yearly_seasonality + monthly_seasonality + weekly_seasonality

# ========== 3. ВЫХОДНЫЕ (эффект) ==========
weekend_strength = np.random.uniform(10, 30)
is_weekend = (dates.dayofweek >= 5).astype(int)  # суббота=5, воскресенье=6
weekend_effect = is_weekend * weekend_strength

# ========== 4. ПРАЗДНИКИ ==========
holiday_strength = global_holiday_strength * np.random.uniform(0.8, 1.2)

# Новый год (1-7 января)
is_new_year = (dates.month == 1) & (dates.day <= 7)

# 8 Марта
is_women_day = (dates.month == 3) & (dates.day == 8)

# Можно добавить другие праздники
is_may_holidays = (dates.month == 5) & (dates.day >= 1) & (dates.day <= 9)

holiday_effect = (is_new_year | is_women_day | is_may_holidays).astype(int) * holiday_strength

# ========== 5. ШУМ ==========
noise = np.random.normal(0, global_noise_level, n_days)

# ========== 6. ИТОГОВЫЕ ПРОДАЖИ ==========
sales = trend + seasonality + weekend_effect + holiday_effect + noise
sales = np.maximum(sales, 0)  # продажи не могут быть отрицательными
sales = np.round(sales, 2)

# ========== 7. СОЗДАНИЕ DATAFRAME (только 2 колонки) ==========
df = pd.DataFrame({
    "date": dates,
    "sales": sales
})

# Сохраняем в CSV
df.to_csv("sales_data.csv", index=False)