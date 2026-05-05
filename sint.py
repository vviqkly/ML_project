import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_days = 1500
dates = pd.date_range(start="2026-01-01", periods=n_days)
data = []

for product_id in range(3):  # 3 товара
    trend = np.linspace(10, 50, n_days)  # рост продаж
    seasonality = 10 * np.sin(np.arange(n_days) * 2 * np.pi / 30)  # сезонность
    noise = np.random.normal(0, 3, n_days)  # шум
    
    quantity = trend + seasonality + noise
    quantity = np.maximum(0, quantity)  # убираем отрицательные значения
    
    price = np.random.uniform(5, 15, n_days)
    sales = price * quantity
    
    df = pd.DataFrame({
        "date": dates,
        "product": f"product_{product_id}",
        "price": price,
        "quantity": quantity,
        "sales": sales
    })
    
    data.append(df)

final_df = pd.concat(data)
final_df.to_csv("sales_data.csv", index=False)
final_df.head()

# фильтр по одному товару
product_0 = final_df[final_df["product"] == "product_0"]
plt.figure(figsize=(12, 5))
plt.plot(product_0["date"], product_0["sales"])
plt.title("Продажи товара product_0")
plt.xlabel("Дата")
plt.ylabel("Продажи")
plt.show()

final_df.describe()