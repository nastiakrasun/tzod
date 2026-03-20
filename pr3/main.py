import math
import time
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# =========================================================
# 1. Генерація даних з датчиків
# =========================================================
np.random.seed(42)

def generate_sensor_data(n):
    power = np.random.uniform(50, 500, n)         # потужність
    temperature = np.random.uniform(10, 80, n)    # температура
    vibration = np.random.uniform(0, 5, n)        # вібрація
    load_factor = np.random.uniform(0.5, 1.5, n)  # коефіцієнт навантаження
    return power, temperature, vibration, load_factor


# =========================================================
# 2. Чиста реалізація Python
# =========================================================
def energy_needs_python(power, temperature, vibration, load_factor):
    result = []
    for i in range(len(power)):
        temp_coeff = 1 + 0.01 * abs(temperature[i] - 25)
        vib_coeff = 1 + 0.05 * vibration[i]
        load_coeff = load_factor[i] ** 1.2

        energy = power[i] * temp_coeff * vib_coeff * load_coeff + math.log1p(power[i])
        result.append(energy)
    return result


# =========================================================
# 3. Реалізація з Numba
# =========================================================
@njit
def energy_needs_numba(power, temperature, vibration, load_factor):
    result = np.empty(len(power))
    for i in range(len(power)):
        temp_coeff = 1 + 0.01 * abs(temperature[i] - 25)
        vib_coeff = 1 + 0.05 * vibration[i]
        load_coeff = load_factor[i] ** 1.2

        energy = power[i] * temp_coeff * vib_coeff * load_coeff + math.log1p(power[i])
        result[i] = energy
    return result


# =========================================================
# 4. Функція для вимірювання часу
# =========================================================
def measure_time(func, *args, repeat=3):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)
    return sum(times) / len(times)


# =========================================================
# 5. Порівняння на різних розмірах даних
# =========================================================
sizes = [1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
python_times = []
numba_times = []

# Попередній виклик для компіляції Numba
p, t, v, l = generate_sensor_data(10)
energy_needs_numba(p, t, v, l)

for size in sizes:
    power, temperature, vibration, load_factor = generate_sensor_data(size)

    t_python = measure_time(
        energy_needs_python,
        power, temperature, vibration, load_factor
    )

    t_numba = measure_time(
        energy_needs_numba,
        power, temperature, vibration, load_factor
    )

    python_times.append(t_python)
    numba_times.append(t_numba)

    print(f"Розмір даних: {size}")
    print(f"  Python: {t_python:.6f} с")
    print(f"  Numba : {t_numba:.6f} с")
    print("-" * 40)


# =========================================================
# 6. Таблиця результатів
# =========================================================
print("\nПідсумкова таблиця:")
print(f"{'Розмір':>10} | {'Python (с)':>12} | {'Numba (с)':>12} | {'Прискорення':>12}")
print("-" * 56)
for size, py_t, nb_t in zip(sizes, python_times, numba_times):
    speedup = py_t / nb_t if nb_t != 0 else float('inf')
    print(f"{size:>10} | {py_t:>12.6f} | {nb_t:>12.6f} | {speedup:>12.2f}x")


# =========================================================
# 7. Побудова графіка
# =========================================================
plt.figure(figsize=(10, 6))
plt.plot(sizes, python_times, marker='o', label='Чистий Python')
plt.plot(sizes, numba_times, marker='o', label='Numba')
plt.xlabel('Розмір даних')
plt.ylabel('Час виконання (с)')
plt.title('Порівняння часу виконання: Python vs Numba')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.show()