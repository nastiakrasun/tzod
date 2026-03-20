import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 1. Генерація часового ряду
# =========================================================
np.random.seed(42)

n = 200
dates = pd.date_range(start='2024-01-01', periods=n, freq='D')

# базовий сигнал + шум
values = 50 + np.random.normal(0, 5, n)

# додаємо аномалії
anomaly_indices = np.random.choice(n, size=10, replace=False)
values[anomaly_indices] += np.random.choice([30, -30], size=10)

df = pd.DataFrame({
    'date': dates,
    'value': values
}).set_index('date')

print("Перші значення:")
print(df.head())


# =========================================================
# 2. Виявлення аномалій через IQR
# =========================================================
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['anomaly'] = (df['value'] < lower_bound) | (df['value'] > upper_bound)

print("\nМежі IQR:")
print(f"Lower bound: {lower_bound:.2f}")
print(f"Upper bound: {upper_bound:.2f}")

print("\nКількість аномалій:", df['anomaly'].sum())


# =========================================================
# 3. Візуалізація
# =========================================================
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['value'], label='Часовий ряд')
plt.scatter(
    df.index[df['anomaly']],
    df['value'][df['anomaly']],
    label='Аномалії'
)

plt.axhline(lower_bound, linestyle='--', label='Нижня межа')
plt.axhline(upper_bound, linestyle='--', label='Верхня межа')

plt.title('Виявлення аномалій у часовому ряді (IQR)')
plt.xlabel('Дата')
plt.ylabel('Значення')
plt.legend()
plt.grid(True)
plt.show()


# =========================================================
# 4. Вивід аномалій
# =========================================================
print("\nАномальні значення:")
print(df[df['anomaly']])