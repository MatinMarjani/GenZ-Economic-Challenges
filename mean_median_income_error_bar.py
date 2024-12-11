import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

file_path = 'Inflation_Adjusted_Data.csv'
df = pd.read_csv(file_path)

# between the ages of 18 and 27
filtered_df = df[(df['AGE'] >= 18) & (df['AGE'] <= 27)]

# mean and median income by generation
income_stats = filtered_df.groupby('GENERATION')['INCTOT_adjusted'].agg(['mean', 'median']).reset_index()

# standard error of the mean (SEM)
income_stats['mean_se'] = filtered_df.groupby('GENERATION')['INCTOT_adjusted'].sem().reset_index(drop=True)

# bootstrap confidence interval for the median
def calculate_median_ci(data, n_resamples=1000):
    res = bootstrap((data,), np.median, confidence_level=0.95, n_resamples=n_resamples, method='percentile')
    return res.confidence_interval.high - np.median(data)  # Error bar = CI high - median

# error bars for the median using bootstrap
income_stats['median_ci'] = filtered_df.groupby('GENERATION')['INCTOT_adjusted'].apply(
    lambda x: calculate_median_ci(x.values)
).reset_index(drop=True)

# total mean and median for income (not by generation)
total_mean_income = filtered_df['INCTOT_adjusted'].mean()
total_median_income = filtered_df['INCTOT_adjusted'].median()

print(f"Total Mean Income (Ages 18-27): ${total_mean_income:,.2f}")
print(f"Total Median Income (Ages 18-27): ${total_median_income:,.2f}")

generation_order = ['Baby Boomers', 'Generation X', 'Millennials', 'Generation Z']
income_stats['GENERATION'] = pd.Categorical(income_stats['GENERATION'], categories=generation_order, ordered=True)
income_stats = income_stats.sort_values('GENERATION')
categories = income_stats['GENERATION'].tolist()
mean_gross_income = income_stats['mean'].tolist()
median_gross_income = income_stats['median'].tolist()
mean_errors = income_stats['mean_se'].tolist()
median_errors = income_stats['median_ci'].tolist()

x = np.arange(len(categories))
bar_width = 0.4
plt.figure(figsize=(12, 6))
plt.bar(x - bar_width / 2, mean_gross_income, bar_width, yerr=mean_errors, capsize=5,
        label='Mean Gross Income (2023 $)', color='blue', alpha=0.7, error_kw=dict(ecolor='black'))
plt.bar(x + bar_width / 2, median_gross_income, bar_width, yerr=median_errors, capsize=5,
        label='Median Gross Income (2023 $)', color='orange', alpha=0.7, error_kw=dict(ecolor='black'))
for i, (mean, mean_err) in enumerate(zip(mean_gross_income, mean_errors)):
    plt.text(i - bar_width / 2, mean + mean_err + 200, f"${mean:,.0f}", ha='center', va='bottom', fontsize=9) 
for i, (median, median_err) in enumerate(zip(median_gross_income, median_errors)):
    plt.text(i + bar_width / 2, median + median_err + 200, f"${median:,.0f}", ha='center', va='bottom', fontsize=9)

plt.title("Mean and Median Gross Income by Generation (Ages 18–27, Adjusted to 2023 $)", fontsize=16)
plt.xlabel("Generation", fontsize=14)
plt.ylabel("Amount ($)", fontsize=14)
plt.xticks(x, categories, fontsize=12)
plt.legend(fontsize=10, loc='lower left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
