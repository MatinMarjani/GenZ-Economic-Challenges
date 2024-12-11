from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set up FRED API
fred_api_key = '5ed2c780159d3f69873d3733844898c0'
fred = Fred(api_key=fred_api_key)

# Fetch CPI data (CPIAUCSL: Consumer Price Index for All Urban Consumers)
cpi_data = fred.get_series('CPIAUCSL')

# Convert to DataFrame and set starting year to 1970
cpi_df = cpi_data.to_frame(name='CPI')
cpi_df['Year'] = cpi_df.index.year
cpi_df = cpi_df[cpi_df['Year'] >= 1970] 
cpi_df = cpi_df.groupby('Year').mean().reset_index()

# Calculate cumulative inflation
cpi_df['CPI_Start'] = cpi_df['CPI'].iloc[0]  # Starting CPI value in 1970
cpi_df['Real_Cumulative_Inflation'] = ((cpi_df['CPI'] / cpi_df['CPI_Start']) - 1) * 100  # Cumulative percentage change

file_path = 'Inflation_Adjusted_Data.csv'
df = pd.read_csv(file_path)

year_column = 'YEAR'
income_column = 'INCTOT_adjusted'
data_by_year = df.groupby(year_column).agg(
    mean_income=(income_column, 'mean'),
    sem_income=(income_column, lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))  # SEM
).reset_index()

# Merge real cumulative inflation data from FRED
data_by_year = pd.merge(data_by_year, cpi_df[['Year', 'Real_Cumulative_Inflation']], left_on=year_column, right_on='Year', how='left')

# relative growth in income over the years
data_by_year['income_growth'] = (data_by_year['mean_income'] / data_by_year['mean_income'].iloc[0] - 1) * 100

# Use SEM of income to calculate real error margins for income growth
data_by_year['income_growth_error'] = data_by_year['sem_income'] / data_by_year['mean_income'].iloc[0] * 100

years = data_by_year[year_column].tolist()
income_growth = data_by_year['income_growth'].tolist()
real_cumulative_inflation = data_by_year['Real_Cumulative_Inflation'].tolist()
income_growth_error = data_by_year['income_growth_error'].tolist()


plt.figure(figsize=(10, 6))
plt.errorbar(years, income_growth, yerr=income_growth_error, label="Income Growth Rate (%)",
             marker='o', color='blue', capsize=5, linestyle='-', capthick=1)
plt.plot(years, real_cumulative_inflation, label="Cumulative Inflation Rate (%)", 
         marker='o', color='red', linestyle='-', linewidth=2)
plt.title("Income Growth Rate vs Cumulative Inflation Rate Over Time", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Cumulative Change (%)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
