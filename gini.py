import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'Inflation_Adjusted_Data.csv'
df = pd.read_csv(file_path)
income_column = 'INCTOT_adjusted'
generation_column = 'GENERATION'

# cumulative shares and Gini index
def lorenz_curve(income):
    income_sorted = np.sort(income)
    population_share = np.linspace(0, 1, len(income_sorted) + 1)
    income_cumsum = np.cumsum(income_sorted)
    income_share = np.insert(income_cumsum / income_cumsum[-1], 0, 0)
    A = np.trapz(y=income_share, x=population_share)
    Gini_index = 1 - 2 * A
    return population_share, income_share, Gini_index

income_sets = {}
for generation, group in df.groupby(generation_column):
    income = group[income_column]
    income = income[income > 0]
    if len(income) > 0:
        income_sets[generation] = income.values

plt.figure(figsize=(10, 8))

# Lorenz curves for each income distribution
for label, income in income_sets.items():
    population_share, income_share, gini_index = lorenz_curve(income)
    plt.plot(population_share, income_share, label=f"{label} (Gini={gini_index:.2f})", lw=2)

# line of equality
plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Line of Equality")

plt.title("Income Inequality Across Generations: Lorenz Curves and Gini Indices", fontsize=16)
plt.xlabel("Cumulative Share of Population", fontsize=14)
plt.ylabel("Cumulative Share of Income", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()
