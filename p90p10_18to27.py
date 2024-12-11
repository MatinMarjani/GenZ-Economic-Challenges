import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

file_path = 'Inflation_Adjusted_Data.csv'
df = pd.read_csv(file_path)
income_column = 'INCTOT_adjusted'
generation_column = 'GENERATION'
age_column = 'AGE'
df = df[(df[income_column] > 0) & (df[age_column] >= 18) & (df[age_column] <= 27)]

# P90/P10 ratio and bootstrap confidence intervals
def calculate_p90_p10_with_ci(group):
    # Remove outliers using IQR
    q1 = group[income_column].quantile(0.25)
    q3 = group[income_column].quantile(0.75)
    iqr = q3 - q1
    filtered_group = group[(group[income_column] >= q1 - 1.5 * iqr) & (group[income_column] <= q3 + 1.5 * iqr)]
    
    # Calculate P90 and P10
    p90 = filtered_group[income_column].quantile(0.9)
    p10 = filtered_group[income_column].quantile(0.1)
    
    # Avoid division by very small P10 values
    if p10 > 0:
        ratio = p90 / p10
    else:
        return np.nan, np.nan
    
    # Bootstrap confidence intervals for the ratio
    def bootstrap_ratio(data):
        p90 = np.percentile(data, 90)
        p10 = np.percentile(data, 10)
        return p90 / p10 if p10 > 0 else np.nan

    res = bootstrap((filtered_group[income_column].values,), bootstrap_ratio, confidence_level=0.95, n_resamples=1000, method="percentile")
    ci_error = res.confidence_interval.high - ratio

    return ratio, ci_error

results = df.groupby(generation_column).apply(calculate_p90_p10_with_ci)
generation_order = ['Baby Boomers', 'Generation X', 'Millennials', 'Generation Z']
results = results.reindex(generation_order).dropna()

categories = results.index.tolist()
p90_p10_values = [result[0] for result in results]
error_bars = [result[1] for result in results]

x = np.arange(len(categories))
bar_width = 0.4

plt.figure(figsize=(10, 6))
plt.bar(x, p90_p10_values, bar_width, color="purple", alpha=0.8, yerr=error_bars, capsize=5, 
        error_kw=dict(ecolor="black", elinewidth=1.5))
for i in range(len(x)):
    plt.text(x[i], p90_p10_values[i] + error_bars[i] + 0.2, f"{p90_p10_values[i]:.1f}", ha='center', fontsize=10)
plt.title("P90/P10 Ratio by Generation (Ages 18-27)", fontsize=16)
plt.xlabel("Generation", fontsize=14)
plt.ylabel("P90/P10 Ratio", fontsize=14)
plt.xticks(x, categories, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
