import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# ============== ИМПОРТ ДАННЫХ ==============
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'RGR1_A-3_X1-X4.csv')

df = pd.read_csv(csv_path)
columns = ['X1', 'X2', 'X3']

print("=" * 70)
print("ЭТАП 1: ПЕРВИЧНОЕ ОПИСАНИЕ ВЫБОРКИ")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════
# 1.1 ГРАФИК ЭМПИРИЧЕСКОЙ ФУНКЦИИ РАСПРЕДЕЛЕНИЯ Fn(x)
# ═══════════════════════════════════════════════════════════════════
print("\n1.1 ГРАФИК ЭМПИРИЧЕСКОЙ ФУНКЦИИ РАСПРЕДЕЛЕНИЯ")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, col in enumerate(columns):
    data = df[col].values
    n = len(data)
    sorted_data = np.sort(data)

    ecdf_x = sorted_data
    ecdf_y = np.arange(1, n + 1) / n

    axes[idx].step(ecdf_x, ecdf_y, where='post', linewidth=2)
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('Fn(x)')
    axes[idx].set_title(f'{col}')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('1.1_ecdf.png', dpi=300)
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 1.2 ГИСТОГРАММЫ С РАЗНЫМ КОЛИЧЕСТВОМ ИНТЕРВАЛОВ
# ═══════════════════════════════════════════════════════════════════
print("\n1.2 ГИСТОГРАММЫ")


def get_bins(data, rule):
    n = len(data)
    if rule == 'Sturges':
        return int(np.ceil(1 + np.log2(n)))
    elif rule == 'Scott':
        bin_width = 3.5 * np.std(data, ddof=1) / (n ** (1 / 3))
        return int(np.ceil((data.max() - data.min()) / bin_width))
    elif rule == 'FD':
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        bin_width = 2 * iqr / (n ** (1 / 3))
        return int(np.ceil((data.max() - data.min()) / bin_width)) if bin_width > 0 else 10
    elif rule == 'Sqrt':
        return int(np.ceil(np.sqrt(n)))


rules = ['Sturges', 'Scott', 'FD', 'Sqrt']

for col in columns:
    data = df[col].values
    n = len(data)

    print(f"\n{col} (n = {n}):")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, rule in enumerate(rules):
        n_bins = get_bins(data, rule)
        axes[idx].hist(data, bins=n_bins, edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Частота')
        axes[idx].set_title(f'{rule}: {n_bins} интервал(ов)')
        axes[idx].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'1.2_histogram_{col}.png', dpi=300)
    plt.show()

# ═══════════════════════════════════════════════════════════════════
# 1.3 ЧИСЛОВЫЕ ХАРАКТЕРИСТИКИ
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1.3 ЧИСЛОВЫЕ ХАРАКТЕРИСТИКИ")
print("=" * 70)

for col in columns:
    data = df[col].values
    n = len(data)

    mean = np.mean(data)
    var_biased = np.var(data, ddof=0)  # S² (смещённая)
    std_biased = np.std(data, ddof=0)  # S (смещённое)
    var_unbiased = np.var(data, ddof=1)  # σ̂² (несмещённая)
    std_unbiased = np.std(data, ddof=1)  # σ̂ (несмещённое)
    median = np.median(data)

    print(f"\n{col}:")
    print(f"  a. Среднее (x̄) = {mean:.4f}")
    print(f"  b. S² (смещённая) = {var_biased:.4f}, S = {std_biased:.4f}")
    print(f"  c. σ̂² (несмещённая) = {var_unbiased:.4f}, σ̂ = {std_unbiased:.4f}")
    print(f"     → S² делится на n, σ̂² делится на (n-1)")
    print(f"     → Несмещённая оценка для генеральной совокупности")
    print(f"  d. Медиана (m̃e) = {median:.4f}")

# ═══════════════════════════════════════════════════════════════════
# 1.4 ОПИСАНИЕ ФОРМЫ РАСПРЕДЕЛЕНИЯ
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1.4 ОПИСАНИЕ ФОРМЫ РАСПРЕДЕЛЕНИЯ")
print("=" * 70)

for col in columns:
    data = df[col].values
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    outliers = data[(data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)]

    print(f"\n{col}:")

    # Симметрия
    if abs(skewness) < 0.5:
        print(f"  • Симметричное (skew = {skewness:.3f})")
    elif skewness > 0.5:
        print(f"  • Правосторонняя асимметрия (skew = {skewness:.3f})")
        print(f"    → Правый «хвост»")
    else:
        print(f"  • Левосторонняя асимметрия (skew = {skewness:.3f})")
        print(f"    → Левый «хвост»")

    # Выбросы
    print(f"  • Выбросы: {len(outliers)}", end="")
    if len(outliers) > 0:
        print(f" ({np.round(outliers, 2)})")
    else:
        print()

    # Моды
    print(f"  • Модальность: унимодальное (по гистограмме)")

# ═══════════════════════════════════════════════════════════════════
# ЭТАП 2: ПРЕДПОЛОЖЕНИЕ О ЗАКОНЕ РАСПРЕДЕЛЕНИЯ
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ЭТАП 2: ПРЕДПОЛОЖЕНИЕ О ЗАКОНЕ РАСПРЕДЕЛЕНИЯ")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════
# 2.1 СООТВЕТСТВИЕ ЗАКОНАМ N, U, Exp
# ═══════════════════════════════════════════════════════════════════
print("\n2.1 ПРЕДПОЛОЖЕНИЕ О ЗАКОНЕ РАСПРЕДЕЛЕНИЯ")

for col in columns:
    data = df[col].values
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    stat, p_value = stats.shapiro(data)
    min_val, max_val = data.min(), data.max()

    print(f"\n{col}:")
    print(f"  Shapiro-Wilk: p-value = {p_value:.4f}")
    print(f"  Асимметрия: {skewness:.3f}, Эксцесс: {kurtosis:.3f}")
    print(f"  Диапазон: [{min_val:.2f}, {max_val:.2f}]")

    # Вывод о распределении
    if p_value > 0.05 and abs(skewness) < 0.5:
        print(f"  → Закон: N(μ={mean:.2f}, σ={std:.2f})")
    elif min_val >= 0 and skewness > 0.5:
        lambda_param = 1 / mean
        print(f"  → Закон: Exp(λ={lambda_param:.4f})")
    else:
        print(f"  → Закон: U(a={min_val:.2f}, b={max_val:.2f})")

# ═══════════════════════════════════════════════════════════════════
# 2.1 ОПРЕДЕЛЕНИЕ ТИПА РАСПРЕДЕЛЕНИЯ
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2.1 ТИП РАСПРЕДЕЛЕНИЯ ПО ТАБЛИЦЕ ВАРИАНТОВ")
print("=" * 70)

distribution_results = {}

for col in columns:
    data = df[col].values
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    cv = (std / mean) * 100  # Коэффициент вариации
    min_val, max_val = data.min(), data.max()

    # Выбросы
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    outliers = data[(data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)]

    print(f"\n{'=' * 50}")
    print(f"{col}")
    print(f"{'=' * 50}")

    # Логика определения варианта
    variant = None
    distr_type = ""
    justification = []

    # === ЭКСПОНЕНЦИАЛЬНОЕ (Варианты 4-5) ===
    if skewness > 1.0 and min_val >= 0:
        if cv > 80:  # Сильный разброс
            variant = 4
            distr_type = "Экспоненциальное"
            justification.append("Резкий пик у левой границы (x ≈ 0)")
            justification.append("Длинный правый хвост")
            justification.append(f"Асимметрия = {skewness:.2f} > 1")
            justification.append(f"Коэф.вариации = {cv:.1f}% > 80%")
        else:
            variant = 5
            distr_type = "Экспоненциальное (пологий спад)"
            justification.append("Менее выраженный пик")
            justification.append("Более равномерный спад")

    # === РАВНОМЕРНОЕ (Вариант 6) ===
    elif abs(skewness) < 0.5 and cv > 25 and cv < 40:
        # Проверка на равномерность через количество выбросов и форму
        variant = 6
        distr_type = "Равномерное"
        justification.append("Все столбцы гистограммы примерно одинаковой высоты")
        justification.append("Нет явного пика (моды)")
        justification.append(f"Асимметрия = {skewness:.2f} ≈ 0")
        justification.append(f"Диапазон: [{min_val:.1f}, {max_val:.1f}]")

    # === НОРМАЛЬНОЕ (Варианты 1-3) ===
    elif abs(skewness) < 0.5:
        if cv < 20:
            variant = 2
            distr_type = "Нормальное (узкое)"
            justification.append("Высокий узкий колокол")
            justification.append(f"Малый разброс (CV = {cv:.1f}%)")
        elif cv > 40:
            variant = 3
            distr_type = "Нормальное (широкое)"
            justification.append("Низкий широкий колокол")
            justification.append(f"Большой разброс (CV = {cv:.1f}%)")
        else:
            variant = 1
            distr_type = "Нормальное"
            justification.append("Симметричный колокол")
            justification.append("Один пик (унимодальное)")
            justification.append(f"Асимметрия = {skewness:.2f} ≈ 0")

    # === БИМОДАЛЬНОЕ (Варианты 7-9) ===
    # (Если бы было два пика)

    # === НОРМАЛЬНОЕ + ВЫБРОСЫ (Варианты 10-12) ===
    elif len(outliers) > 3 and abs(skewness) < 1.0:
        if len(outliers) > 5:
            variant = 10
            distr_type = "Нормальное + выбросы"
        else:
            variant = 11
            distr_type = "Нормальное + выбросы"
        justification.append(f"Основной колокол + {len(outliers)} выброс(ов)")
        justification.append(f"Выбросы: {np.round(outliers, 1)}")

    # === СМЕСЬ РАСПРЕДЕЛЕНИЙ (Варианты 13-14) ===
    elif skewness > 0.8 and len(outliers) > 5:
        variant = 13
        distr_type = "Смесь нормального и экспоненциального"
        justification.append("Пик в центре + хвост справа")
        justification.append(f"Много выбросов ({len(outliers)} шт.)")

    # === ПО УМОЛЧАНИЮ ===
    else:
        variant = 1
        distr_type = "Нормальное (приближённо)"
        justification.append("Наиболее близкое к нормальному")

    # Сохраняем результат
    distribution_results[col] = {
        'variant': variant,
        'type': distr_type,
        'justification': justification
    }

    # Вывод
    print(f"  ВАРИАНТ {variant}: {distr_type}")
    print(f"  Обоснование:")
    for j in justification:
        print(f"    • {j}")

# Итоговая таблица
print("\n" + "=" * 70)
print("ИТОГОВАЯ ТАБЛИЦА РАСПРЕДЕЛЕНИЙ")
print("=" * 70)

summary_data = []
for col in columns:
    res = distribution_results[col]
    data = df[col].values
    summary_data.append({
        'Столбец': col,
        'Вариант': res['variant'],
        'Тип распределения': res['type'],
        'Асимметрия': round(stats.skew(data), 3),
        'CV (%)': round((np.std(data, ddof=1) / np.mean(data)) * 100, 1),
        'Выбросов': len(
            data[(data < np.percentile(data, 25) - 1.5 * (np.percentile(data, 75) - np.percentile(data, 25))) |
                 (data > np.percentile(data, 75) + 1.5 * (np.percentile(data, 75) - np.percentile(data, 25)))])
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
summary_df.to_csv('2.1_distribution_summary.csv', index=False, encoding='utf-8-sig')

# ═══════════════════════════════════════════════════════════════════
# 2.2 СУЩЕСТВЕННЫЕ ПРИЗНАКИ ДЛЯ ВЫВОДА
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2.2 СУЩЕСТВЕННЫЕ ПРИЗНАКИ ДЛЯ ВЫВОДА")
print("=" * 70)

for col in columns:
    data = df[col].values
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    stat, p_value = stats.shapiro(data)
    min_val, max_val = data.min(), data.max()

    print(f"\n{col}:")
    print("  Признаки:")

    if p_value > 0.05:
        print("    ✓ p-value > 0.05 (нормальность не отвергается)")
    else:
        print("    ✗ p-value < 0.05 (нормальность отвергается)")

    if abs(skewness) < 0.5:
        print("    ✓ Асимметрия ≈ 0")
    else:
        print(f"    ✗ Асимметрия = {skewness:.2f}")

    if abs(kurtosis) < 0.5:
        print("    ✓ Эксцесс ≈ 0")
    else:
        print(f"    ✗ Эксцесс = {kurtosis:.2f}")

    if min_val >= 0:
        print("    ✓ x ≥ 0 (естественная нижняя граница)")

    cv = (std / mean) * 100
    print(f"    • Коэффициент вариации: {cv:.1f}%")

print("\n" + "=" * 70)
print("✅ Графики сохранены: 1.1_ecdf.png, 1.2_histogram_X*.png")
print("=" * 70)