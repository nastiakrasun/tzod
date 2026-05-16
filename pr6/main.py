import pandas as pd
import matplotlib.pyplot as plt

# === 1. Завантаження даних ===
file_path = "/Users/nastiakrasun/Desktop/Studying/3 курс/tzod/pr6/data.csv"   # назва CSV-файлу

df = pd.read_csv(file_path)

# === 2. Перевірка назв колонок ===
print("Колонки у файлі:")
print(df.columns)

# === 3. Вибір потрібних колонок ===
time = df["Time (s)"]
acceleration = df["Absolute acceleration (m/s^2)"]

# === 4. Фільтрація шуму методом ковзного середнього ===
window_size = 5

df["Filtered acceleration"] = (
    acceleration
    .rolling(window=window_size, center=True)
    .mean()
)

# === 5. Поділ експерименту на ділянки ===
# 0–15 с — спокійна ходьба
# 15–25 с — швидка ходьба
# 25–30.84 с — біг

walking = df[(time >= 0) & (time < 15)]
fast_walking = df[(time >= 15) & (time < 25)]
running = df[(time >= 25) & (time <= 30.84)]

# === 6. Обчислення середнього прискорення ===
mean_walking = walking["Filtered acceleration"].mean()
mean_fast_walking = fast_walking["Filtered acceleration"].mean()
mean_running = running["Filtered acceleration"].mean()

print("\nСереднє прискорення:")
print(f"Спокійна ходьба: {mean_walking:.3f} м/с²")
print(f"Швидка ходьба: {mean_fast_walking:.3f} м/с²")
print(f"Біг: {mean_running:.3f} м/с²")

# === 7. Побудова графіка прискорення від часу ===
plt.figure(figsize=(12, 6))

plt.plot(
    time,
    acceleration,
    label="Сире прискорення",
    alpha=0.4
)

plt.plot(
    time,
    df["Filtered acceleration"],
    label="Згладжене прискорення",
    linewidth=2
)

plt.axvspan(0, 15, alpha=0.2, label="Спокійна ходьба")
plt.axvspan(15, 25, alpha=0.2, label="Швидка ходьба")
plt.axvspan(25, 30.84, alpha=0.2, label="Біг")

plt.xlabel("Час, с")
plt.ylabel("Прискорення, м/с²")
plt.title("Зміна прискорення під час ходьби та бігу")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("acceleration_time_graph.png", dpi=300)
plt.show()

# === 8. Побудова стовпчикової діаграми ===
activities = ["Спокійна ходьба", "Швидка ходьба", "Біг"]
means = [mean_walking, mean_fast_walking, mean_running]

plt.figure(figsize=(8, 5))
plt.bar(activities, means)

plt.ylabel("Середнє прискорення, м/с²")
plt.title("Порівняння середнього прискорення")
plt.grid(axis="y")
plt.tight_layout()

plt.savefig("mean_acceleration_bar_chart.png", dpi=300)
plt.show()

# === 9. Збереження результатів у CSV ===
summary = pd.DataFrame({
    "Режим руху": activities,
    "Середнє прискорення, м/с²": means
})

summary.to_csv("acceleration_summary.csv", index=False)

print("\nРезультати збережено у файл acceleration_summary.csv")