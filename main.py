import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Импорт данных
df = pd.read_csv('RGR1_A-3_X1-X4.csv')

columns = ['X1', 'X2', 'X3']

print("1. ВАРИАЦИОННЫЙ РЯД")

for col in columns:
    sorted_data = np.sort(df[col].values)
    print(f"\n{col}:")
    print(sorted_data)

print("2. ЭМПИРИЧЕСКАЯ ФУНКЦИЯ РАСПРЕДЕЛЕНИЯ Fn(x)")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, col in enumerate(columns):
    data = df[col].values
    n = len(data)
    sorted_data = np.sort(data)

    # Эмпирическая функция распределения
    ecdf_x = sorted_data
    ecdf_y = np.arange(1, n + 1) / n

    axes[idx].step(ecdf_x, ecdf_y, where='post', linewidth=2)
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('Fn(x)')
    axes[idx].set_title(f'{col} - Эмпирическая функция распределения')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('ecdf_plot.png', dpi=300)
plt.show()

# print(df.head())
# print(df.info())