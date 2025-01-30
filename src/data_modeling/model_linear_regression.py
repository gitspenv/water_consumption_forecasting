### Model for testing purposes only###

import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Add the parent directory to system path for module imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Import data loading and exploration functions
from data_exploration.data_loader import load_and_clean_data
from data_exploration.data_exploration import perform_data_exploration

# Define the base path
base_path = os.getcwd()

# Specify the file path
input_file_path = os.path.join(base_path, "data", "processed", "water", "water_consumption_2015_2023_normalized.csv")

# Load and clean the data using the data loader
df_cleaned = load_and_clean_data(input_file_path)

# Perform data exploration
perform_data_exploration(df_cleaned)

# Set the index to a period
df_cleaned.index = pd.DatetimeIndex(df_cleaned.index).to_period('D')

# Combine Saturday and Sunday into one 'is_weekend' variable
df_cleaned['is_weekend'] = df_cleaned['is_saturday'] | df_cleaned['is_sunday']  # Use bitwise OR
df_cleaned = df_cleaned.drop(columns=['is_saturday', 'is_sunday'])  # Drop individual weekend columns

# Define target and features
endog = df_cleaned['Wasserverbrauch']  # Target variable (water consumption)
exog = df_cleaned.drop(columns=['Wasserverbrauch', 'rolling_mean', 'RainDur_min', 'Geburte', 'StrGlo_W/m2'])  # Features (other variables)

# Add new features: squared terms and interaction terms
exog['Temp^2'] = df_cleaned['T_C'] ** 2
exog['StrGlo^2'] = df_cleaned['StrGlo_W/m2'] ** 2
exog['RainDur_min^2'] = df_cleaned['RainDur_min'] ** 2
exog['RainDur_min*weekend'] = df_cleaned['RainDur_min'] * df_cleaned['is_weekend']  # Interaction term

# Clean the exogenous variables 
exog_clean = exog.replace([np.inf, -np.inf], np.nan)
exog_clean = exog_clean.dropna()

# Ensure the target variable has no missing values
endog_clean = endog[exog_clean.index]

# Add constant to exogenous variables for the intercept
exog_clean = sm.add_constant(exog_clean)

# Split into train and test sets
train_size = len(exog_clean) - 360
train_exog = exog_clean.iloc[:train_size]
test_exog = exog_clean.iloc[train_size:]
train_endog = endog_clean.iloc[:train_size]
test_endog = endog_clean.iloc[train_size:]

# Fit the OLS (Ordinary Least Squares) model on the train set
model = sm.OLS(train_endog, train_exog)
results = model.fit()

# Predict on the test set
test_predictions = results.predict(test_exog)

# Print model results
print(results.summary())

# Plot actual vs predicted values for the test set
df_cleaned.index = df_cleaned.index.to_timestamp()

observed_test_values = test_endog
predicted_test_values = test_predictions

# Convert index for plotting
observed_test_values.index = observed_test_values.index.to_timestamp()
predicted_test_values.index = predicted_test_values.index.to_timestamp()

# Plot observed vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(observed_test_values.index, observed_test_values, label='Observed', color='blue')
plt.plot(predicted_test_values.index, predicted_test_values, label='Predicted', color='red', linestyle='--')

# Add labels and title
plt.title('Actual vs Predicted Water Consumption (Test Set)')
plt.xlabel('Date')
plt.ylabel('Water Consumption')

# Show the legend
plt.legend()

# Show the plot
plt.show()
