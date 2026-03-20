import pandas as pd
import numpy as np

# 1. Генеруємо штучний DataFrame
np.random.seed(42)

n = 100

df = pd.DataFrame({
    'product': np.random.choice(['A', 'B', 'C', 'D'], size=n),
    'quantity': np.random.randint(1, 10, size=n).astype(float),
    'price': np.random.randint(50, 500, size=n).astype(float),
    'discount': np.round(np.random.uniform(0, 0.3, size=n), 2)
})

# Додаємо пропуски (~20%)
for col in ['quantity', 'discount']:
    df.loc[df.sample(frac=0.2).index, col] = np.nan

print("Перші рядки датафрейму:")
print(df.head())

# 2. Звіт по пропусках
missing_report = pd.DataFrame({
    'missing_count': df.isna().sum(),
    'missing_percent': (df.isna().mean() * 100).round(2)
}).sort_values(by='missing_percent', ascending=False)

print("\nЗвіт по пропусках:")
print(missing_report)

# 3. Середній чек ДО імпутації
df['total_before'] = df['quantity'] * df['price'] * (1 - df['discount'])
mean_check_before = df['total_before'].mean()

# 4. Імпутація медіаною по product
df['quantity'] = df['quantity'].fillna(
    df.groupby('product')['quantity'].transform('median')
)
df['discount'] = df['discount'].fillna(
    df.groupby('product')['discount'].transform('median')
)

# Дозаповнення, якщо вся група була NaN
df['quantity'] = df['quantity'].fillna(df['quantity'].median())
df['discount'] = df['discount'].fillna(df['discount'].median())

# 5. Середній чек ПІСЛЯ імпутації
df['total_after'] = df['quantity'] * df['price'] * (1 - df['discount'])
mean_check_after = df['total_after'].mean()

# 6. Порівняння
comparison = pd.DataFrame({
    'mean_check_before': [round(mean_check_before, 2)],
    'mean_check_after': [round(mean_check_after, 2)],
    'difference': [round(mean_check_after - mean_check_before, 2)]
})

print("\nПорівняння середнього чеку:")
print(comparison)