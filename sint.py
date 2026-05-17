import numpy as np
import pandas as pd

def generate_dataset():
    np.random.seed(None)
    
    n_days = 750
    dates = pd.date_range(start="2024-01-01", periods=n_days)
    
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