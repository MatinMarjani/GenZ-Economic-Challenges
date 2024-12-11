import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = 'Inflation_Adjusted_Data.csv'
df = pd.read_csv(file_path)

generation_column = 'GENERATION'
empstat_2_column = 'EMPSTAT_2'
empstat_3_column = 'EMPSTAT_3'
age_column = 'AGE'

df['Employed'] = np.where((df[empstat_2_column] == 0) & (df[empstat_3_column] == 0), 1, 0)
df['Unemployed'] = np.where(df[empstat_2_column] == 1, 1, 0)
df['Not_in_Labor_Force'] = np.where(df[empstat_3_column] == 1, 1, 0)

generation_order = ['Baby Boomers', 'Generation X', 'Millennials', 'Generation Z']
df[generation_column] = pd.Categorical(df[generation_column], categories=generation_order, ordered=True)

# Group by generation for all ages
generation_data = df.groupby(generation_column).agg(
    Employed=('Employed', 'mean'),
    Unemployed=('Unemployed', 'mean'),
    Not_in_Labor_Force=('Not_in_Labor_Force', 'mean')
).reset_index()

# Group by generation for ages 18-27
filtered_df = df[(df[age_column] >= 18) & (df[age_column] <= 27)]
generation_data_young = filtered_df.groupby(generation_column).agg(
    Employed=('Employed', 'mean'),
    Unemployed=('Unemployed', 'mean'),
    Not_in_Labor_Force=('Not_in_Labor_Force', 'mean')
).reset_index()

for data in [generation_data, generation_data_young]:
    data['Employed'] *= 100
    data['Unemployed'] *= 100
    data['Not_in_Labor_Force'] *= 100

categories_all = generation_data[generation_column].tolist()
employment_rate_all = generation_data['Employed'].tolist()
unemployment_rate_all = generation_data['Unemployed'].tolist()
not_in_labor_force_all = generation_data['Not_in_Labor_Force'].tolist()

categories_young = generation_data_young[generation_column].tolist()
employment_rate_young = generation_data_young['Employed'].tolist()
unemployment_rate_young = generation_data_young['Unemployed'].tolist()
not_in_labor_force_young = generation_data_young['Not_in_Labor_Force'].tolist()

x_all = np.arange(len(categories_all))
width = 0.4

plt.figure(figsize=(10, 6))
bars_employed_all = plt.bar(x_all, employment_rate_all, width, label="Employed (%)", color="green")
bars_unemployed_all = plt.bar(x_all, unemployment_rate_all, width, bottom=employment_rate_all, label="Unemployed (%)", color="orange")
bars_not_in_labor_force_all = plt.bar(
    x_all, not_in_labor_force_all, width, bottom=np.add(employment_rate_all, unemployment_rate_all), label="Not in Labor Force (%)", color="red"
)

for bar, value in zip(bars_employed_all, employment_rate_all):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"{value:.1f}%", ha='center', va='center', color='white', fontsize=10)
for bar, value, bottom in zip(bars_unemployed_all, unemployment_rate_all, employment_rate_all):
    plt.text(bar.get_x() + bar.get_width() / 2, bottom + value / 2, f"{value:.1f}%", ha='center', va='center', color='black', fontsize=10)
for bar, value, bottom in zip(bars_not_in_labor_force_all, not_in_labor_force_all, np.add(employment_rate_all, unemployment_rate_all)):
    plt.text(bar.get_x() + bar.get_width() / 2, bottom + value / 2, f"{value:.1f}%", ha='center', va='center', color='white', fontsize=10)

plt.title("Labor Market Composition by Generation (All Ages)", fontsize=16)
plt.xlabel("Generation", fontsize=14)
plt.ylabel("Percentage (%)", fontsize=14)
plt.xticks(x_all, categories_all, fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot for ages 18-27
x_young = np.arange(len(categories_young))

plt.figure(figsize=(10, 6))
bars_employed_young = plt.bar(x_young, employment_rate_young, width, label="Employed (%)", color="green")
bars_unemployed_young = plt.bar(x_young, unemployment_rate_young, width, bottom=employment_rate_young, label="Unemployed (%)", color="orange")
bars_not_in_labor_force_young = plt.bar(
    x_young, not_in_labor_force_young, width, bottom=np.add(employment_rate_young, unemployment_rate_young), label="Not in Labor Force (%)", color="red"
)

for bar, value in zip(bars_employed_young, employment_rate_young):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"{value:.1f}%", ha='center', va='center', color='white', fontsize=10)
for bar, value, bottom in zip(bars_unemployed_young, unemployment_rate_young, employment_rate_young):
    plt.text(bar.get_x() + bar.get_width() / 2, bottom + value / 2, f"{value:.1f}%", ha='center', va='center', color='black', fontsize=10)
for bar, value, bottom in zip(bars_not_in_labor_force_young, not_in_labor_force_young, np.add(employment_rate_young, unemployment_rate_young)):
    plt.text(bar.get_x() + bar.get_width() / 2, bottom + value / 2, f"{value:.1f}%", ha='center', va='center', color='white', fontsize=10)

plt.title("Labor Market Composition by Generation (Ages 18-27)", fontsize=16)
plt.xlabel("Generation", fontsize=14)
plt.ylabel("Percentage (%)", fontsize=14)
plt.xticks(x_young, categories_young, fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
