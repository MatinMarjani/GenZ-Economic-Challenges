import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from feature_selection import filter_top_features

data = pd.read_csv('Inflation_Adjusted_Data_Normalized.csv')
data = data.dropna()
X = data.drop(['BIRTHYR', 'AGE', 'YEAR', 'GENERATION'], axis=1)  # Features
y = data['BIRTHYR']  # Target
n_features_to_select =  20 # top features
selected_features = filter_top_features(X, y, n_features=n_features_to_select, add_fisher_features= False)
X = X[selected_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
coef = regressor.coef_
intercept = regressor.intercept_

# predictions
y_pred = regressor.predict(X_test)
# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# correlation and p-value for correlation
correlation, corr_p_value = pearsonr(y_test, y_pred)
print(f"\nCorrelation: {correlation:.4f}")
print(f"P-Value for Correlation: {corr_p_value:.4e}")

# p-values for regression coefficients
X_train_const = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_const).fit()

formatted_summary = ols_model.summary2().tables[1].copy()
formatted_summary['P>|t|'] = formatted_summary['P>|t|'].apply(lambda p: f"{p:.4e}" if p < 0.001 else f"{p:.4f}")

print("\nRegression Coefficients and Formatted P-Values:")
print(formatted_summary)

# Absolute Error and Relative Error
absolute_error = np.abs(y_pred - y_test)
relative_error = (absolute_error / y_test) * 100

# average errors
avg_absolute_error = np.mean(absolute_error)
avg_relative_error = np.mean(relative_error)

print(f"\nAverage Absolute Error: {avg_absolute_error:.4f}")
print(f"Average Relative Error (%): {avg_relative_error:.2f}%")

errors_df = pd.DataFrame({
    'Actual Year': y_test.values,
    'Predicted Year': y_pred,
    'Absolute Error': absolute_error,
    'Relative Error (%)': relative_error
})

print("\nError Analysis (First 10 Samples):")
print(errors_df.head(10))


plt.figure(figsize=(10, 8))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='blue', label='Data point')
sns.regplot(x=y_test, y=y_pred, scatter=False, color='red', label='Linear Fit', ci=95)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Perfect Prediction')
plt.xlabel('Actual Birth Year', fontsize=14)
plt.ylabel('Predicted Birth Year', fontsize=14)
plt.title(
    f'Regression: Predicted vs Actual Birth Year',
    fontsize=16
)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
