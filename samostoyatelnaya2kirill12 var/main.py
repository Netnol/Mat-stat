import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =========================
# ЧТЕНИЕ CSV-ФАЙЛА
# =========================

file_path = "variant_12_data.csv"

df = pd.read_csv(file_path)

# Проверяем колонки
print("Колонки в файле:")
print(df.columns)

# Берем x и y
x = df["x"].values
y = df["y"].values

# =========================
# ПРОСТАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ
# =========================

n = len(x)
alpha = 0.05
dfree = n - 2

x_mean = np.mean(x)
y_mean = np.mean(y)

Sxx = np.sum((x - x_mean) ** 2)
Sxy = np.sum((x - x_mean) * (y - y_mean))

# Коэффициенты
b1 = Sxy / Sxx
b0 = y_mean - b1 * x_mean

# Предсказания
y_hat = b0 + b1 * x

# Остатки
residuals = y - y_hat

# Остаточная дисперсия
RSS = np.sum(residuals ** 2)
s2 = RSS / dfree
s = np.sqrt(s2)

# t-критическое
t_crit = stats.t.ppf(1 - alpha / 2, dfree)

# Стандартные ошибки коэффициентов
se_b1 = s / np.sqrt(Sxx)
se_b0 = s * np.sqrt(1 / n + (x_mean ** 2) / Sxx)

# Доверительные интервалы
b1_ci = (
    b1 - t_crit * se_b1,
    b1 + t_crit * se_b1
)

b0_ci = (
    b0 - t_crit * se_b0,
    b0 + t_crit * se_b0
)

# =========================
# ВЫВОД РЕЗУЛЬТАТОВ
# =========================

print("\n=========================")
print("УРАВНЕНИЕ РЕГРЕССИИ")
print("=========================")

print(f"y = {b0:.4f} + ({b1:.4f}) * x")

print("\n=========================")
print("ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ")
print("=========================")

print(f"b0 = {b0:.4f}")
print(f"95% ДИ для b0: ({b0_ci[0]:.4f}; {b0_ci[1]:.4f})")

print()

print(f"b1 = {b1:.4f}")
print(f"95% ДИ для b1: ({b1_ci[0]:.4f}; {b1_ci[1]:.4f})")

# =========================
# ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ
# ДЛЯ СРЕДНЕЙ ФУНКЦИИ РЕГРЕССИИ
# =========================

x_grid = np.linspace(min(x), max(x), 200)
y_grid_hat = b0 + b1 * x_grid

se_mean = s * np.sqrt(
    1 / n + ((x_grid - x_mean) ** 2) / Sxx
)

lower_mean = y_grid_hat - t_crit * se_mean
upper_mean = y_grid_hat + t_crit * se_mean

# =========================
# ТАБЛИЦА ПО ТОЧКАМ
# =========================

se_points = s * np.sqrt(
    1 / n + ((x - x_mean) ** 2) / Sxx
)

lower_points = y_hat - t_crit * se_points
upper_points = y_hat + t_crit * se_points

result_df = pd.DataFrame({
    "x": x,
    "y": y,
    "y_hat": y_hat,
    "lower_CI": lower_points,
    "upper_CI": upper_points
})

print("\n=========================")
print("ТАБЛИЦА")
print("=========================")

print(result_df.round(4))

# =========================
# ГРАФИК
# =========================

plt.figure(figsize=(10, 6))

plt.scatter(x, y, label="Наблюдения")

plt.plot(
    x_grid,
    y_grid_hat,
    label="Линия регрессии"
)

plt.plot(
    x_grid,
    lower_mean,
    linestyle="--",
    label="Нижняя граница 95% ДИ"
)

plt.plot(
    x_grid,
    upper_mean,
    linestyle="--",
    label="Верхняя граница 95% ДИ"
)

plt.xlabel("x")
plt.ylabel("y")

plt.title("Простая линейная регрессия")

plt.legend()
plt.grid(True)

plt.show()