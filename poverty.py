import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = 'Inflation_Adjusted_Data.csv'
df = pd.read_csv(file_path)
year_column = 'YEAR'
poverty_column = 'POVERTY'
generation_column = 'GENERATION'
age_column = 'AGE'
filtered_df = df[(df[poverty_column] > 0) & (df[poverty_column] < 501)]
filtered_df['Below_Poverty'] = filtered_df[poverty_column] < 100
age_filtered_df = filtered_df[(filtered_df[age_column] >= 18) & (filtered_df[age_column] <= 27)]

# 1.poverty rates by year
poverty_rate_by_year = filtered_df.groupby(year_column)['Below_Poverty'].mean() * 100
poverty_rate_by_year_sem = filtered_df.groupby(year_column)['Below_Poverty'].apply(
    lambda x: np.sqrt(x.mean() * (1 - x.mean()) / len(x))
) * 100

# 2.poverty rates by generation
poverty_rate_by_generation = filtered_df.groupby(generation_column)['Below_Poverty'].mean() * 100

# 3.poverty rates by generation for people aged 18–27
poverty_rate_by_generation_young = age_filtered_df.groupby(generation_column)['Below_Poverty'].mean() * 100

# Plot 1: Poverty rate by year
plt.figure(figsize=(14, 6))
plt.errorbar(
    poverty_rate_by_year.index,
    poverty_rate_by_year.values,
    yerr=poverty_rate_by_year_sem.values,
    fmt='-o',
    color='blue',
    capsize=5,
    label='Poverty Rate by Year',
)
plt.title("Poverty Rate Over Years", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Poverty Rate (%)", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Plot 2: Poverty rate by generation
plt.figure(figsize=(14, 6))
x = np.arange(len(poverty_rate_by_generation))
plt.bar(x, poverty_rate_by_generation.values, color='orange', alpha=0.7, label='Poverty Rate by Generation')
plt.xticks(x, poverty_rate_by_generation.index, rotation=45, fontsize=12)
plt.title("Poverty Rate by Generation", fontsize=16)
plt.xlabel("Generation", fontsize=14)
plt.ylabel("Poverty Rate (%)", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Plot 3: Poverty rate by generation for people aged 18–27
plt.figure(figsize=(14, 6))
x = np.arange(len(poverty_rate_by_generation_young))
plt.bar(x, poverty_rate_by_generation_young.values, color='green', alpha=0.7, label='Poverty Rate (18–27) by Generation')
plt.xticks(x, poverty_rate_by_generation_young.index, rotation=45, fontsize=12)
plt.title("Poverty Rate (Ages 18–27) by Generation", fontsize=16)
plt.xlabel("Generation", fontsize=14)
plt.ylabel("Poverty Rate (%)", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
