import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = 'Inflation_Adjusted_Data.csv'
df = pd.read_csv(file_path)
year_column = 'YEAR'
ownership_column = 'OWNERSHP_2'
housing_value_column = 'HOUSING_VALUE_adjusted'
income_column = 'INCTOT_adjusted'
homeowners = df[df[ownership_column] == 0]

# average house value per year
average_house_values = homeowners.groupby(year_column)[housing_value_column].mean()
# error bars for house values using standard error of the mean (SEM)
grouped_house = homeowners.groupby(year_column)[housing_value_column]
house_errors = grouped_house.std() / np.sqrt(grouped_house.count())  # SEM as error bars

# average income value per year
average_income_values = df.groupby(year_column)[income_column].mean()
# error bars for income using standard error of the mean (SEM)
grouped_income = df.groupby(year_column)[income_column]
income_errors = grouped_income.std() / np.sqrt(grouped_income.count())  # SEM as error bars

years_house = average_house_values.index
house_values_2023 = average_house_values.values
house_errors = house_errors.values
years_income = average_income_values.index
income_values_2023 = average_income_values.values
income_errors = income_errors.values
years_house = years_house.astype(int)
years_income = years_income.astype(int)

plt.figure(figsize=(14, 8))
plt.errorbar(years_house, house_values_2023, yerr=house_errors, fmt='-o', color='blue', linewidth=2, 
             label='Average House Value (2023 $)', capsize=5)
plt.errorbar(years_income, income_values_2023, yerr=income_errors, fmt='-o', color='green', linewidth=2, 
             label='Average Income (2023 $)', capsize=5)
manual_offsets_house = {
    1970: 10000,
    1980: 10000,
    1990: 10000,
    2000: 32000,
    2010: 5000,
    2020: 5000,
    2023: 23000
}
for year, offset in manual_offsets_house.items():
    if year in average_house_values.index:
        value = average_house_values.loc[year]
        error = house_errors[np.where(years_house == year)[0][0]]
        plt.text(year, value + error + offset, f"${value:,.0f}", ha='center', fontsize=9, color='blue')
manual_offsets_income = {
    1970: -15000,
    1980: -15000,
    1990: -15000,
    2000: -15000,
    2010: -18000,
    2020: -18000,
    2023: 6000
}
for year, offset in manual_offsets_income.items():
    if year in average_income_values.index:
        value = average_income_values.loc[year]
        error = income_errors[np.where(years_income == year)[0][0]]
        plt.text(year, value + error + offset, f"${value:,.0f}", ha='center', fontsize=9, color='green')

plt.title("Inflation-Adjusted Average House Value and Income (2023 Dollars)", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Value (2023 $)", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
