import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'Inflation_Adjusted_Data.csv'
df = pd.read_csv(file_path)
age_min, age_max = 18, 27
df = df[(df['AGE'] >= age_min) & (df['AGE'] <= age_max)]
income_column = 'INCTOT_adjusted'
generation_column = 'GENERATION'

income_shares = {}
for generation, group in df.groupby(generation_column):
    sorted_incomes = group[income_column].sort_values(ascending=False)
    total_income = sorted_incomes.sum()
    # income shares
    top_1_percent_share = sorted_incomes[:int(len(sorted_incomes) * 0.01)].sum() / total_income * 100
    top_10_percent_share = sorted_incomes[:int(len(sorted_incomes) * 0.10)].sum() / total_income * 100
    bottom_50_percent_share = sorted_incomes[int(len(sorted_incomes) * 0.50):].sum() / total_income * 100
    # Top 10% to exclude Top 1%
    top_10_percent_excluding_top_1 = top_10_percent_share - top_1_percent_share
    remaining_share = 100 - (top_1_percent_share + top_10_percent_excluding_top_1 + bottom_50_percent_share)
    bottom_50_percent_share += remaining_share
    income_shares[generation] = {
        "Top 1%": top_1_percent_share,
        "Top 10% (Excluding Top 1%)": top_10_percent_excluding_top_1,
        "Bottom 50%": bottom_50_percent_share
    }

generation_order = ['Baby Boomers', 'Generation X', 'Millennials', 'Generation Z']
income_shares = {gen: income_shares[gen] for gen in generation_order if gen in income_shares}
categories = list(income_shares.keys())
top_1_percent = [income_shares[gen]["Top 1%"] for gen in categories]
top_10_percent = [income_shares[gen]["Top 10% (Excluding Top 1%)"] for gen in categories]
bottom_50_percent = [income_shares[gen]["Bottom 50%"] for gen in categories]

x = np.arange(len(categories))
bar_width = 0.6

plt.figure(figsize=(10, 6))
bars_bottom = plt.bar(x, bottom_50_percent, bar_width, label="Bottom 50%", color="green", alpha=0.8)
bars_top10 = plt.bar(x, top_10_percent, bar_width, bottom=bottom_50_percent, label="Top 10% (Excluding Top 1%)", color="orange", alpha=0.8)
bars_top1 = plt.bar(x, top_1_percent, bar_width, bottom=np.add(bottom_50_percent, top_10_percent), label="Top 1%", color="red", alpha=0.8)

for bar, value in zip(bars_bottom, bottom_50_percent):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"{value:.1f}%", ha='center', va='center', color='white', fontsize=9)

for bar, value, bottom in zip(bars_top10, top_10_percent, bottom_50_percent):
    plt.text(bar.get_x() + bar.get_width() / 2, bottom + value / 2, f"{value:.1f}%", ha='center', va='center', color='black', fontsize=9)

for bar, value, bottom in zip(bars_top1, top_1_percent, np.add(bottom_50_percent, top_10_percent)):
    plt.text(bar.get_x() + bar.get_width() / 2, bottom + value / 2, f"{value:.1f}%", ha='center', va='center', color='white', fontsize=9)

plt.title("Top Income Shares by Generation (Age 18-27)", fontsize=16)
plt.xlabel("Generation", fontsize=14)
plt.ylabel("Percentage of Total Income (%)", fontsize=14)
plt.xticks(x, categories, fontsize=12)
plt.legend(fontsize=12, loc="lower left")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
