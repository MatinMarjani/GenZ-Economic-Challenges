import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = 'Inflation_Adjusted_Data.csv'
df = pd.read_csv(file_path)
year_column = 'YEAR'
income_column = 'INCTOT_adjusted'  # (annual income)
ownership_column = 'OWNERSHP_2'  # (0 = own, 1 = rent)
housing_value_column = 'HOUSING_VALUE_adjusted'  # (rent or house value)

renters = df[df[ownership_column] == 1]
renters['Annual_Rent'] = renters[housing_value_column] * 12
income_rent_by_year = renters.groupby(year_column).agg(
    Average_Income=(income_column, 'mean'),
    Income_SEM=(income_column, lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
    Average_Annual_Rent=('Annual_Rent', 'mean'),
    Rent_SEM=('Annual_Rent', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
).reset_index()

# rent-to-income ratio and propagate error using SEM
income_rent_by_year['Rent_to_Income_Ratio'] = (
    income_rent_by_year['Average_Annual_Rent'] / income_rent_by_year['Average_Income']
)
income_rent_by_year['Rent_to_Income_Ratio_Error'] = income_rent_by_year['Rent_to_Income_Ratio'] * np.sqrt(
    (income_rent_by_year['Rent_SEM'] / income_rent_by_year['Average_Annual_Rent']) ** 2 +
    (income_rent_by_year['Income_SEM'] / income_rent_by_year['Average_Income']) ** 2
)

plt.figure(figsize=(14, 6))
plt.errorbar(
    income_rent_by_year[year_column],
    income_rent_by_year['Rent_to_Income_Ratio'],
    yerr=income_rent_by_year['Rent_to_Income_Ratio_Error'],
    fmt='-o', color='purple', linewidth=2, capsize=5, label='Rent-to-Income Ratio'
)

for i, row in income_rent_by_year.iterrows():
    year = row[year_column]
    ratio = row['Rent_to_Income_Ratio']
    if i % 5 == 0 or i == len(income_rent_by_year) - 1:
        plt.text(year, ratio + 0.02, f"{ratio:.2f}", ha='center', fontsize=9)

plt.title("Rent-to-Income Ratio Over the Years with Error Bars", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Rent-to-Income Ratio", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
