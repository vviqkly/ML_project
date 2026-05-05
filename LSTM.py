import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn

df = pd.read_csv("sales_data.csv")
sales = df["sales"].values.reshape(-1, 1)

#нормализация

scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(sales)

#создание последовательностей
def create_sequences(data, seq_length=7):
    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    return np.array(X), np.array(y)


X, y = create_sequences(sales_scaled, seq_length=7)

split = int(X.shape[0] * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

#LSTM
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size =1, hidden_size = 50, batch_first = True)

        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out[:, -1, :])
        return out

#подготовка модели

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

#обучение

model = LSTM()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50

for epoch in range(epochs):
    model.train()

    pred = model(X_train)
    loss = loss_fn(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

    #предсказание
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

    #ркальность
y_pred = scaler.inverse_transform(y_pred.numpy())
y_test_real = scaler.inverse_transform(y_test.numpy())

#график
import matplotlib.pyplot as plt

plt.plot(y_test_real, label="Fact")
plt.plot(y_pred, label="Prediction")
plt.legend()
plt.show()

torch.save(model.state_dict(), "lstm_sales.pt")

