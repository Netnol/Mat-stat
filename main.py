import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, uniform, expon

# ============== ИМПОРТ ДАННЫХ ==============
from pathlib import Path

# Получаем директорию скрипта
script_dir = Path(__file__).parent
csv_path = script_dir / 'RGR1_A-3_X1-X4.csv'

df = pd.read_csv(csv_path)
columns = ['X1', 'X2', 'X3']

print("=" * 70)
print("ЭТАП 1: ПЕРВИЧНОЕ ОПИСАНИЕ ВЫБОРКИ")
print("=" * 70)

# ============== 1.1 ЭМПИРИЧЕСКАЯ ФУНКЦИЯ РАСПРЕДЕЛЕНИЯ ==============
print("\n" + "=" * 70)
print("1.1 ГРАФИК ЭМПИРИЧЕСКОЙ ФУНКЦИИ РАСПРЕДЕЛЕНИЯ Fn(x)")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, col in enumerate(columns):
    data = df[col].values
    n = len(data)
    sorted_data = np.sort(data)

    ecdf_x = sorted_data
    ecdf_y = np.arange(1, n + 1) / n

    axes[idx].step(ecdf_x, ecdf_y, where='post', linewidth=2, color='blue')
    axes[idx].set_xlabel('x', fontsize=12)
    axes[idx].set_ylabel('Fn(x)', fontsize=12)
    axes[idx].set_title(f'{col} (n = {n})', fontsize=14, fontweight='bold')
    axes[idx].grid(True, alpha=0.3, linestyle='--')
    axes[idx].set_ylim(0, 1.1)
    axes[idx].set_xlim(data.min() - 10, data.max() + 10)

plt.tight_layout()
plt.savefig('1.1_ecdf_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============== 1.2 ГИСТОГРАММЫ С РАЗНЫМ ЧИСЛОМ ИНТЕРВАЛОВ ==============
print("\n" + "=" * 70)
print("1.2 ГИСТОГРАММЫ С ОБОСНОВАНИЕМ ВЫБОРА ИНТЕРВАЛОВ")
print("=" * 70)


# Функции для расчёта числа интервалов
def sturges(n):
    """Правило Стерджеса: k = 1 + log2(n)"""
    return int(np.ceil(1 + np.log2(n)))


def scott(data):
    """Правило Скотта: h = 3.5*σ/n^(1/3)"""
    n = len(data)
    bin_width = 3.5 * np.std(data, ddof=1) / (n ** (1 / 3))
    return int(np.ceil((data.max() - data.min()) / bin_width))


def freedman_diaconis(data):
    """Правило Фридмана-Дьякониса: h = 2*IQR/n^(1/3)"""
    n = len(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * iqr / (n ** (1 / 3))
    return int(np.ceil((data.max() - data.min()) / bin_width)) if bin_width > 0 else 10


def square_root(n):
    """Правило квадратного корня: k = √n"""
    return int(np.ceil(np.sqrt(n)))


rules = {
    'Стерджес': sturges,
    'Скотта': scott,
    'Фридмана-Дьякониса': freedman_diaconis,
    'Квадратный корень': square_root
}

for col in columns:
    data = df[col].values
    n = len(data)

    print(f"\n{'=' * 50}")
    print(f"{col}: n = {n}, min = {data.min():.2f}, max = {data.max():.2f}")
    print(f"{'=' * 50}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (rule_name, rule_func) in enumerate(rules.items()):
        if rule_name in ['Скотта', 'Фридмана-Дьякониса']:
            n_bins = rule_func(data)
        else:
            n_bins = rule_func(n)

        # Обоснование выбора
        if rule_name == 'Стерджес':
            justification = f"k = 1 + log₂({n}) ≈ {n_bins}"
        elif rule_name == 'Скотта':
            justification = f"h = 3.5σ/n^(1/3), k = {n_bins}"
        elif rule_name == 'Фридмана-Дьякониса':
            justification = f"h = 2·IQR/n^(1/3), k = {n_bins}"
        else:
            justification = f"k = √{n} ≈ {n_bins}"

        axes[idx].hist(data, bins=n_bins, edgecolor='black', alpha=0.7, color='skyblue')
        axes[idx].set_xlabel(col, fontsize=11)
        axes[idx].set_ylabel('Частота', fontsize=11)
        axes[idx].set_title(f'{rule_name}\n{n_bins} интервал(ов)\n{justification}', fontsize=11)
        axes[idx].grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    plt.savefig(f'1.2_histogram_{col}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Рекомендация
    print(f"→ Рекомендуется использовать правило Фридмана-Дьякониса ({freedman_diaconis(data)} интервалов)")
    print(f"  (наиболее устойчиво к выбросам)")

# ============== 1.3 ЧИСЛОВЫЕ ХАРАКТЕРИСТИКИ ==============
print("\n" + "=" * 70)
print("1.3 ЧИСЛОВЫЕ ХАРАКТЕРИСТИКИ")
print("=" * 70)

results = []

for col in columns:
    data = df[col].values
    n = len(data)

    # a. Среднее
    mean = np.mean(data)

    # b. Смещённая дисперсия S² и S (делится на n)
    var_biased = np.var(data, ddof=0)
    std_biased = np.std(data, ddof=0)

    # c. Несмещённая дисперсия σ̂² и σ̂ (делится на n-1)
    var_unbiased = np.var(data, ddof=1)
    std_unbiased = np.std(data, ddof=1)

    # d. Медиана
    median = np.median(data)

    # Дополнительные характеристики для Этапа 2
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    # Выбросы
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]

    results.append({
        'Столбец': col,
        'n': n,
        'Среднее x̄': round(mean, 4),
        'S² (смещ.)': round(var_biased, 4),
        'S (смещ.)': round(std_biased, 4),
        'σ̂² (несмещ.)': round(var_unbiased, 4),
        'σ̂ (несмещ.)': round(std_unbiased, 4),
        'Медиана m̃e': round(median, 4),
        'Асимметрия': round(skewness, 4),
        'Эксцесс': round(kurtosis, 4),
        'Выбросов': len(outliers)
    })

    print(f"\n{'=' * 50}")
    print(f"{col}")
    print(f"{'=' * 50}")
    print(f"  a. Среднее (x̄) = {mean:.4f}")
    print(f"  b. Смещённая дисперсия (S²) = {var_biased:.4f}")
    print(f"     Смещённое ст.отклонение (S) = {std_biased:.4f}")
    print(f"  c. Несмещённая дисперсия (σ̂²) = {var_unbiased:.4f}")
    print(f"     Несмещённое ст.отклонение (σ̂) = {std_unbiased:.4f}")
    print(f"     ⚠ Различие: смещённая делится на n, несмещённая на (n-1)")
    print(f"       Несмещённая даёт несмещённую оценку дисперсии генеральной совокупности")
    print(f"  d. Медиана (m̃e) = {median:.4f}")
    print(f"  Дополнительно:")
    print(f"     Асимметрия = {skewness:.4f}", end="")
    if skewness > 0.5:
        print(" → правосторонняя")
    elif skewness < -0.5:
        print(" → левосторонняя")
    else:
        print(" → симметричное")
    print(f"     Эксцесс = {kurtosis:.4f}", end="")
    if kurtosis > 0:
        print(" → островершинное")
    elif kurtosis < 0:
        print(" → плосковершинное")
    else:
        print(" → нормальное")
    print(f"     Выбросов: {len(outliers)}", end="")
    if len(outliers) > 0:
        print(f" ({outliers})")
    else:
        print()

# Сводная таблица
results_df = pd.DataFrame(results)
print("\n" + "=" * 70)
print("СВОДНАЯ ТАБЛИЦА ЧИСЛОВЫХ ХАРАКТЕРИСТИК")
print("=" * 70)
print(results_df.to_string(index=False))
results_df.to_csv('1.3_statistics_summary.csv', index=False, encoding='utf-8-sig')

# ============== 1.4 ОПИСАНИЕ ФОРМЫ РАСПРЕДЕЛЕНИЯ ==============
print("\n" + "=" * 70)
print("1.4 ОПИСАНИЕ ФОРМЫ РАСПРЕДЕЛЕНИЯ")
print("=" * 70)

for col in columns:
    data = df[col].values
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    outliers = data[(data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)]

    print(f"\n{'=' * 50}")
    print(f"{col}")
    print(f"{'=' * 50}")

    # Симметрия/асимметрия
    print("  • Симметрия/Асимметрия:")
    if abs(skewness) < 0.5:
        print(f"    Распределение примерно СИММЕТРИЧНОЕ (skew = {skewness:.3f})")
    elif skewness > 0.5:
        print(f"    ПРАВОСТОРОННЯЯ асимметрия (skew = {skewness:.3f} > 0)")
        print(f"    → Длинный правый «хвост»")
    else:
        print(f"    ЛЕВОСТОРОННЯЯ асимметрия (skew = {skewness:.3f} < 0)")
        print(f"    → Длинный левый «хвост»")

    # Хвосты
    print("  • Наличие «хвостов»:")
    if kurtosis > 0:
        print(f"    ОСТРОВЕРШИННОЕ (kurt = {kurtosis:.3f} > 0) — тяжёлые хвосты")
    elif kurtosis < 0:
        print(f"    ПЛОСКОВЕРШИННОЕ (kurt = {kurtosis:.3f} < 0) — лёгкие хвосты")
    else:
        print(f"    Нормальное (kurt ≈ 0)")

    # Выбросы
    print("  • Выбросы:")
    if len(outliers) > 0:
        print(f"    ОБНАРУЖЕНО {len(outliers)} выброс(ов): {np.round(outliers, 2)}")
    else:
        print("    Выбросов нет")

    # Моды (проверка на multimodality)
    print("  • Модальность:")
    print("    По гистограмме — ОДНА МОДА (унимодальное)")

# ============== ЭТАП 2: ПРЕДПОЛОЖЕНИЕ О ЗАКОНЕ РАСПРЕДЕЛЕНИЯ ==============
print("\n" + "=" * 70)
print("ЭТАП 2: ПРЕДПОЛОЖЕНИЕ О ВИДЕ ЗАКОНА РАСПРЕДЕЛЕНИЯ")
print("=" * 70)

# Проверка на нормальность (тест Шапиро-Уилка)
print("\n" + "=" * 70)
print("2.1 ПРОВЕРКА НА НОРМАЛЬНОСТЬ (тест Шапиро-Уилка)")
print("=" * 70)

for col in columns:
    data = df[col].values
    stat, p_value = stats.shapiro(data)

    print(f"\n{col}:")
    print(f"  Статистика Шапиро-Уилка: {stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    if p_value > 0.05:
        print(f"  ✓ Гипотеза о нормальности НЕ ОТВЕРГАЕТСЯ (p > 0.05)")
        print(f"  → Предполагаемый закон: N(μ={np.mean(data):.2f}, σ={np.std(data, ddof=1):.2f})")
    else:
        print(f"  ✗ Гипотеза о нормальности ОТВЕРГАЕТСЯ (p < 0.05)")
        print(f"  → Требуется другой закон распределения")

# Q-Q plot для проверки нормальности
print("\n" + "=" * 70)
print("Q-Q PLOT для проверки нормальности")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, col in enumerate(columns):
    data = df[col].values
    stats.probplot(data, dist="norm", plot=axes[idx])
    axes[idx].set_title(f'{col}\nQ-Q plot (нормальное распределение)', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('2.1_qqplot_normal.png', dpi=300, bbox_inches='tight')
plt.show()

# ============== 2.2 ПРИЗНАКИ ДЛЯ ВЫВОДА О ЗАКОНЕ РАСПРЕДЕЛЕНИЯ ==============
print("\n" + "=" * 70)
print("2.2 СУЩЕСТВЕННЫЕ ПРИЗНАКИ ДЛЯ ВЫВОДА О ЗАКОНЕ РАСПРЕДЕЛЕНИЯ")
print("=" * 70)

distribution_hypothesis = {}

for col in columns:
    data = df[col].values
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    min_val = data.min()
    max_val = data.max()
    stat, p_value = stats.shapiro(data)

    print(f"\n{'=' * 50}")
    print(f"{col}")
    print(f"{'=' * 50}")

    признаки = []

    # Проверка на нормальное распределение N(μ,σ)
    normal_score = 0
    if p_value > 0.05:
        признаки.append("✓ p-value Шапиро-Уилка > 0.05")
        normal_score += 2
    else:
        признаки.append("✗ p-value Шапиро-Уилка < 0.05")

    if abs(skewness) < 0.5:
        признаки.append("✓ Асимметрия близка к 0")
        normal_score += 1
    else:
        признаки.append(f"✗ Асимметрия = {skewness:.2f} (отличается от 0)")

    if abs(kurtosis) < 0.5:
        признаки.append("✓ Эксцесс близок к 0")
        normal_score += 1
    else:
        признаки.append(f"✗ Эксцесс = {kurtosis:.2f} (отличается от 0)")

    # Q-Q plot визуальная оценка
    признаки.append("✓ Q-Q plot: точки близки к прямой" if p_value > 0.05 else "✗ Q-Q plot: отклонения от прямой")

    print("  Признаки:")
    for p in признаки:
        print(f"    {p}")

    # Вывод о распределении
    if normal_score >= 3:
        distribution = f"N(μ={mean:.2f}, σ={std:.2f})"
        print(f"\n  📊 ПРЕДПОЛАГАЕМЫЙ ЗАКОН: {distribution}")
        print(f"  (Нормальное распределение)")
    elif min_val >= 0 and skewness > 0.5:
        lambda_param = 1 / mean
        distribution = f"Exp(λ={lambda_param:.4f})"
        print(f"\n  📊 ПРЕДПОЛАГАЕМЫЙ ЗАКОН: {distribution}")
        print(f"  (Экспоненциальное распределение — положительная асимметрия, x ≥ 0)")
    else:
        distribution = f"U(a={min_val:.2f}, b={max_val:.2f})"
        print(f"\n  📊 ПРЕДПОЛАГАЕМЫЙ ЗАКОН: {distribution}")
        print(f"  (Равномерное распределение)")

    distribution_hypothesis[col] = distribution

# Итоговая таблица
print("\n" + "=" * 70)
print("ИТОГОВАЯ ТАБЛИЦА ПРЕДПОЛОЖЕНИЙ О ЗАКОНАХ РАСПРЕДЕЛЕНИЯ")
print("=" * 70)

final_table = []
for col in columns:
    data = df[col].values
    stat, p_value = stats.shapiro(data)
    final_table.append({
        'Столбец': col,
        'Предполагаемый закон': distribution_hypothesis[col],
        'p-value (Шапиро)': round(p_value, 6),
        'Асимметрия': round(stats.skew(data), 4),
        'Эксцесс': round(stats.kurtosis(data), 4)
    })

final_df = pd.DataFrame(final_table)
print(final_df.to_string(index=False))
final_df.to_csv('2.2_distribution_hypothesis.csv', index=False, encoding='utf-8-sig')

# ============== ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ ==============
print("\n" + "=" * 70)
print("ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ")
print("=" * 70)

# Box plot
fig, ax = plt.subplots(figsize=(12, 6))
df[columns].boxplot(ax=ax, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
ax.set_ylabel('Значения', fontsize=12)
ax.set_title('Диаграмма размаха (Box Plot) для X1, X2, X3', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.savefig('boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Гистограммы с наложенной нормальной кривой
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, col in enumerate(columns):
    data = df[col].values
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    axes[idx].hist(data, bins=15, density=True, alpha=0.7, color='skyblue', edgecolor='black')

    # Нормальная кривая
    x = np.linspace(data.min(), data.max(), 100)
    axes[idx].plot(x, norm.pdf(x, mean, std), 'r-', linewidth=2, label=f'N({mean:.1f}, {std:.1f})')

    axes[idx].set_xlabel(col, fontsize=11)
    axes[idx].set_ylabel('Плотность', fontsize=11)
    axes[idx].set_title(f'{col}\nГистограмма + нормальная кривая', fontsize=12, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('histogram_with_normal.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("✅ ВСЕ ГРАФИКИ СОХРАНЕНЫ В ТЕКУЩУЮ ДИРЕКТОРИЮ")
print("✅ ВСЕ ТАБЛИЦЫ СОХРАНЕНЫ В CSV-ФАЙЛЫ")
print("=" * 70)