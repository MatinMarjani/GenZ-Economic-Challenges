from feature_selection import filter_top_features
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'Inflation_Adjusted_Data_Normalized.csv'
df = pd.read_csv(file_path)

birth_year_column = 'BIRTHYR'
target = df[birth_year_column]

columns_to_drop = ['GENERATION', 'AGE', 'YEAR', 'BIRTHYR']
features = df.drop(columns=columns_to_drop)

features = features.apply(pd.to_numeric, errors='coerce').fillna(0)

selected_features = filter_top_features(features, target, n_features=20, add_fisher_features= False)

selected_data = features[selected_features].copy()
selected_data[birth_year_column] = target

correlations = {}
p_values = {}
for feature in selected_features:
    corr, p_value = pearsonr(selected_data[feature], selected_data[birth_year_column])
    correlations[feature] = corr
    p_values[feature] = p_value

formatted_p_values = {
    feature: "< 1E-5" if p_value < 1E-5 else p_value
    for feature, p_value in p_values.items()
}

correlation_df = pd.DataFrame({
    'Feature': correlations.keys(),
    'Correlation': correlations.values(),
    'P-Value': formatted_p_values.values()
}).sort_values(by='Correlation', ascending=False)

print(correlation_df)

heatmap_data = selected_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title('Correlation Heatmap of Selected Features', fontsize=16)
plt.tight_layout()
plt.show()
