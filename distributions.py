import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

file_path = 'Inflation_Adjusted_Data.csv'
df = pd.read_csv(file_path)

income_column = 'INCTOT_adjusted'
generation_column = 'GENERATION'

df = df[[income_column, generation_column]].dropna()

def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

df = remove_outliers(df, income_column)
generations = df[generation_column].unique()
plt.figure(figsize=(10, 6))

for gen in generations:
    gen_data = df[df[generation_column] == gen][income_column]
    
    kde = gaussian_kde(gen_data)
    x_vals = np.linspace(gen_data.min(), gen_data.max(), 1000)
    y_vals = kde(x_vals)
    
    line, = plt.plot(x_vals, y_vals, label=f"{gen} (Adjusted)", linewidth=2)
    mean_value = gen_data.mean()
    plt.axvline(mean_value, color=line.get_color(), linestyle='--', alpha=0.7, label=f"{gen} Mean: {mean_value:.2f}")

plt.title("Income Distribution Adjusted for Inflation (2023 $)", fontsize=16)
plt.xlabel("Income (Adjusted for Inflation in 2023 $)", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
