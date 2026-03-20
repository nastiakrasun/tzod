import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =========================================================
# 1. Модель енергетичних потоків
# =========================================================
# Нехай:
# x(t) — запас енергії в системі
# dx/dt = generation - demand - losses
#
# generation = alpha
# demand = beta * x
# losses = gamma * x^2
#
# alpha  - інтенсивність генерації
# beta   - коефіцієнт споживання
# gamma  - нелінійні втрати
#
# Така модель дозволяє оцінити, чи виходить система на стабільний стан,
# чи спостерігаються сильні коливання / перевантаження.


def energy_flow_ode(t, x, alpha, beta, gamma):
    dxdt = alpha - beta * x[0] - gamma * x[0]**2
    return [dxdt]


# =========================================================
# 2. Функція симуляції
# =========================================================
def simulate_energy(alpha, beta, gamma, x0=10, t_max=50):
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, 500)

    sol = solve_ivp(
        energy_flow_ode,
        t_span,
        [x0],
        args=(alpha, beta, gamma),
        t_eval=t_eval,
        method='RK45'
    )

    return sol.t, sol.y[0]


# =========================================================
# 3. Оцінка стабільності
# =========================================================
# Вважаємо, що система стабільна, якщо наприкінці симуляції
# значення енергії виходить на майже сталий рівень:
# невелика дисперсія на останньому відрізку часу.
def stability_metric(x, tail_fraction=0.2):
    tail_size = int(len(x) * tail_fraction)
    tail = x[-tail_size:]
    return np.std(tail)


# =========================================================
# 4. Набори параметрів для порівняння
# =========================================================
param_sets = [
    {"alpha": 20, "beta": 0.8, "gamma": 0.03, "label": "Базовий режим"},
    {"alpha": 25, "beta": 0.8, "gamma": 0.03, "label": "Вища генерація"},
    {"alpha": 20, "beta": 0.5, "gamma": 0.03, "label": "Менше споживання"},
    {"alpha": 20, "beta": 0.8, "gamma": 0.08, "label": "Вищі втрати"},
]

results = []

plt.figure(figsize=(10, 6))

for params in param_sets:
    t, x = simulate_energy(params["alpha"], params["beta"], params["gamma"])
    stab = stability_metric(x)

    results.append({
        "label": params["label"],
        "alpha": params["alpha"],
        "beta": params["beta"],
        "gamma": params["gamma"],
        "mean_energy": np.mean(x),
        "final_energy": x[-1],
        "stability_std": stab
    })

    plt.plot(t, x, label=f'{params["label"]} (std={stab:.3f})')

plt.title("Симуляція енергетичних потоків у мережі")
plt.xlabel("Час")
plt.ylabel("Рівень енергії")
plt.legend()
plt.grid(True)
plt.show()


# =========================================================
# 5. Таблиця результатів моделювання
# =========================================================
print("Результати симуляції:")
print(f"{'Режим':<20} | {'alpha':>6} | {'beta':>6} | {'gamma':>6} | {'Сер.енергія':>12} | {'Фінал':>10} | {'STD стаб.':>10}")
print("-" * 90)

for r in results:
    print(f"{r['label']:<20} | {r['alpha']:>6.2f} | {r['beta']:>6.2f} | {r['gamma']:>6.2f} | "
          f"{r['mean_energy']:>12.3f} | {r['final_energy']:>10.3f} | {r['stability_std']:>10.3f}")


# =========================================================
# 6. Порівняння з методом середніх значень
# =========================================================
# Попередній статистичний підхід:
# оцінюємо мережу лише через середнє значення енергії.
#
# Це простіше, але не показує динаміку:
# - чи є вихід на стабільний режим
# - чи є коливання
# - як поводиться система в часі

labels = [r["label"] for r in results]
mean_values = [r["mean_energy"] for r in results]
stability_values = [r["stability_std"] for r in results]

x_pos = np.arange(len(labels))

plt.figure(figsize=(10, 6))
plt.bar(x_pos, mean_values)
plt.xticks(x_pos, labels, rotation=15)
plt.title("Порівняння за середнім значенням енергії")
plt.ylabel("Середній рівень енергії")
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(x_pos, stability_values)
plt.xticks(x_pos, labels, rotation=15)
plt.title("Порівняння стабільності мережі")
plt.ylabel("STD на кінцевій ділянці")
plt.grid(axis='y')
plt.show()


# =========================================================
# 7. Автоматичний текстовий висновок
# =========================================================
print("\nВисновок:")
print("Диференціальна модель дозволяє дослідити зміну енергетичного стану системи в часі.")
print("На відміну від статистичного підходу, що базується лише на середніх значеннях,")
print("цей метод показує динаміку процесу, перехідні режими та стабільність мережі.")
print("Малі значення stability_std свідчать про вихід системи на стабільний режим.")