import csv
import math
from scipy import stats

# Чтение данных из CSV файлов
def read_csv(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Определяем название столбца со значениями (x_i или y_j)
            for key, value in row.items():
                if key != 'i' and key != 'j':
                    data.append(float(value))
                    break
    return data

# Чтение выборок
X = read_csv('variant_8_sample_X.csv')
Y = read_csv('variant_8_sample_Y.csv')

# Размеры выборок
m = len(X)  # 56
n = len(Y)  # 42

# Выборочные средние
X_bar = sum(X) / m
Y_bar = sum(Y) / n

# Выборочные дисперсии (несмещенные)
s1_sq = sum((x - X_bar)**2 for x in X) / (m - 1)
s2_sq = sum((y - Y_bar)**2 for y in Y) / (n - 1)

# Объединенная оценка дисперсии
s_sq = ((m - 1) * s1_sq + (n - 1) * s2_sq) / (m + n - 2)
s = math.sqrt(s_sq)

# Наблюдаемое значение статистики критерия
t_obs = (X_bar - Y_bar) / (s * math.sqrt(1/m + 1/n))

# Степени свободы
k = m + n - 2

# Уровень значимости
alpha = 0.05

# Критическое значение (двусторонний критерий)
t_crit = stats.t.ppf(1 - alpha/2, k)

# Вывод результатов
print("=" * 60)
print("ПРОВЕРКА ГИПОТЕЗЫ О РАВЕНСТВЕ МАТЕМАТИЧЕСКИХ ОЖИДАНИЙ")
print("=" * 60)
print(f"\nНулевая гипотеза H0: μ1 = μ2")
print(f"Альтернативная гипотеза H1: μ1 ≠ μ2")
print(f"\nРазмеры выборок:")
print(f"  m = {m}")
print(f"  n = {n}")
print(f"\nВыборочные характеристики:")
print(f"  X̄ (среднее первой выборки) = {X_bar:.4f}")
print(f"  Ȳ (среднее второй выборки) = {Y_bar:.4f}")
print(f"  s₁² (дисперсия первой выборки) = {s1_sq:.4f}")
print(f"  s₂² (дисперсия второй выборки) = {s2_sq:.4f}")
print(f"  s² (объединенная дисперсия) = {s_sq:.4f}")
print(f"  s (объединенное среднеквадратичное) = {s:.4f}")
print(f"\nСтатистика критерия:")
print(f"  t_набл = {t_obs:.4f}")
print(f"\nПараметры критерия:")
print(f"  Число степеней свободы k = {k}")
print(f"  Уровень значимости α = {alpha}")
print(f"  t_крит = ±{t_crit:.4f}")
print(f"\nКритическая область:")
print(f"  W = (-∞; {-t_crit:.4f}] ∪ [{t_crit:.4f}; +∞)")
print(f"\nРешение:")
print(f"  |t_набл| = {abs(t_obs):.4f}")
print(f"  t_крит = {t_crit:.4f}")

if abs(t_obs) > t_crit:
    print(f"\n{'=' * 60}")
    print("ВЫВОД: H0 ОТВЕРГАЕТСЯ")
    print("На уровне значимости 0.05 есть статистически значимое")
    print("различие между математическими ожиданиями выборок")
    print("=" * 60)
else:
    print(f"\n{'=' * 60}")
    print("ВЫВОД: H0 НЕ ОТВЕРГАЕТСЯ")
    print("На уровне значимости 0.05 нет оснований утверждать, что")
    print("математические ожидания выборок различны")
    print("=" * 60)
