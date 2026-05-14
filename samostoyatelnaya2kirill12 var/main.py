import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# =========================
# 1. Загрузка данных
# =========================

file_path = "variant_12_data.csv"

df = pd.read_csv(file_path)

# На всякий случай убираем лишние пробелы в названиях колонок
df.columns = df.columns.str.strip()

# В файле должны быть колонки: i, x, y
x = df["x"].to_numpy()
y = df["y"].to_numpy()

n = len(df)
alpha = 0.05

print("Исходные данные:")
print(df)


# =========================
# 2. Диаграмма рассеяния
# =========================

plt.figure(figsize=(8, 5))
plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Диаграмма рассеяния")
plt.grid(True)
plt.show()

print("\nПо диаграмме видно, что зависимость между x и y скорее отрицательная.")


# =========================
# 3. Расчёт коэффициентов регрессии
# =========================

x_mean = np.mean(x)
y_mean = np.mean(y)

Sxx = np.sum((x - x_mean) ** 2)
Sxy = np.sum((x - x_mean) * (y - y_mean))

beta_1 = Sxy / Sxx
beta_0 = y_mean - beta_1 * x_mean

print("\nОценки коэффициентов регрессии:")
print(f"beta_0 = {beta_0:.4f}")
print(f"beta_1 = {beta_1:.4f}")

print("\nУравнение регрессии:")
print(f"y_hat = {beta_0:.4f} + ({beta_1:.4f}) * x")


# =========================
# 4. Прогнозные значения и остатки
# =========================

df["y_hat"] = beta_0 + beta_1 * x
df["e"] = y - df["y_hat"]
df["e^2"] = df["e"] ** 2

print("\nТаблица с прогнозными значениями и остатками:")
print(df.round(4))


# =========================
# 5. Остаточная дисперсия
# =========================

RSS = np.sum(df["e^2"])
s2 = RSS / (n - 2)
s = np.sqrt(s2)

TSS = np.sum((y - y_mean) ** 2)
R2 = 1 - RSS / TSS

print("\nОстаточная сумма квадратов и дисперсия:")
print(f"RSS = {RSS:.4f}")
print(f"s^2 = {s2:.4f}")
print(f"s = {s:.4f}")
print(f"R^2 = {R2:.4f}")


# =========================
# 6. Доверительные интервалы для коэффициентов
# =========================

t_crit = stats.t.ppf(1 - alpha / 2, n - 2)

SE_beta_1 = s / np.sqrt(Sxx)
SE_beta_0 = s * np.sqrt(1 / n + x_mean ** 2 / Sxx)

beta_0_left = beta_0 - t_crit * SE_beta_0
beta_0_right = beta_0 + t_crit * SE_beta_0

beta_1_left = beta_1 - t_crit * SE_beta_1
beta_1_right = beta_1 + t_crit * SE_beta_1

print("\n95% доверительные интервалы для коэффициентов:")
print(f"beta_0: [{beta_0_left:.4f}; {beta_0_right:.4f}]")
print(f"beta_1: [{beta_1_left:.4f}; {beta_1_right:.4f}]")

if beta_1_left <= 0 <= beta_1_right:
    print("Ноль входит в интервал для beta_1, значит зависимость нельзя считать значимой.")
else:
    print("Ноль не входит в интервал для beta_1, значит зависимость статистически значима.")


# =========================
# 7. Доверительный интервал для средней функции регрессии
# =========================

x_grid = np.linspace(x.min(), x.max(), 200)

y_grid_hat = beta_0 + beta_1 * x_grid

SE_mean = s * np.sqrt(
    1 / n + ((x_grid - x_mean) ** 2) / Sxx
)

lower_bound = y_grid_hat - t_crit * SE_mean
upper_bound = y_grid_hat + t_crit * SE_mean


# =========================
# 8. Итоговый график
# =========================

plt.figure(figsize=(10, 6))

plt.scatter(x, y, label="Исходные точки")
plt.plot(x_grid, y_grid_hat, label="Прямая регрессии")
plt.plot(x_grid, lower_bound, linestyle="--", label="Нижняя граница 95% ДИ")
plt.plot(x_grid, upper_bound, linestyle="--", label="Верхняя граница 95% ДИ")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Простая линейная регрессия с доверительным интервалом")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# 9. Краткие выводы
# =========================

print("\nВыводы:")
print(f"1. Уравнение регрессии: y_hat = {beta_0:.4f} + ({beta_1:.4f}) * x.")

if beta_1 < 0:
    print("2. Зависимость между x и y отрицательная.")
else:
    print("2. Зависимость между x и y положительная.")

if beta_1_left <= 0 <= beta_1_right:
    print("3. Ноль попадает в доверительный интервал для beta_1.")
    print("4. Линейную зависимость нельзя считать статистически значимой на уровне 5%.")
else:
    print("3. Ноль не попадает в доверительный интервал для beta_1.")
    print("4. Линейную зависимость можно считать статистически значимой на уровне 5%.")

print("5. Доверительный интервал уже всего около среднего значения x.")
print(f"6. Коэффициент R^2 = {R2:.4f}, он показывает долю изменчивости y, объяснённую моделью.")