# imports
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys

# preprocessing function
def preprocess_data():
    # add parent directory to path
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)

    # import data loader
    from data_exploration.data_loader import load_and_clean_data

    # set base path
    base_path = os.getcwd()

    # set input file path
    input_file_path = os.path.join(base_path, "water_data", "output", "water_consumption_2015_2023_normalized.csv")

    # load and clean data
    df_cleaned = load_and_clean_data(input_file_path, with_lag_features=True, lag_days=7)

    # define function to calculate days since rain
    def calculate_days_since_rain(df):
        days_since_rain = []
        count = 0
        for rain in df['RainDur_min']:
            if rain > 0:
                count = 0
            else:
                count += 1
            days_since_rain.append(count)
        df['days_since_rain'] = days_since_rain
        return df

    # calculate days since rain
    df_cleaned = calculate_days_since_rain(df_cleaned)

    # create lag features for weather
    weather_columns = ['RainDur_min', 'StrGlo_W/m2', 'T_C', 'T_max_h1_C', 'p_hPa']
    for col in weather_columns:
        for lag in range(1, 8):
            df_cleaned[f'{col}_lag_{lag}'] = df_cleaned[col].shift(lag)

    # drop NaNs from lagging
    df_cleaned.dropna(inplace=True)

    # set datetime index
    df_cleaned.index = pd.to_datetime(df_cleaned.index)

    return df_cleaned

# preprocess data
df_cleaned = preprocess_data()

# split data
train_data = df_cleaned[df_cleaned.index.year <= 2022]
test_data = df_cleaned[df_cleaned.index.year == 2023]

# define feature creation function
def create_lag_and_rolling_features(df):
    # create lag features for weather
    weather_columns = ['RainDur_min', 'StrGlo_W/m2', 'T_C', 'T_max_h1_C', 'p_hPa']
    for col in weather_columns:
        for lag in range(1, 8):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # create rolling features
    df['rolling_mean_3'] = df['Wasserverbrauch'].rolling(window=3).mean()
    df['rolling_mean_7'] = df['Wasserverbrauch'].rolling(window=7).mean()
    df['rolling_std_3'] = df['Wasserverbrauch'].rolling(window=3).std()

    return df

# create features for training and testing
train_data = create_lag_and_rolling_features(train_data)
test_data = create_lag_and_rolling_features(test_data)

# drop NaNs from rolling
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# define feature columns
feature_columns = ['is_saturday', 'is_sunday', 'month', 'weekday', 'rolling_mean_3', 'rolling_mean_7', 
                   'rolling_std_3', 'days_since_rain'] + \
                  [f'{col}_lag_{i}' for col in ['RainDur_min', 'StrGlo_W/m2', 'T_C', 'T_max_h1_C', 'p_hPa'] for i in range(1, 8)]

# split into features and target
X_train = train_data[feature_columns]
y_train = train_data['Wasserverbrauch']
X_test = test_data[feature_columns]
y_test = test_data['Wasserverbrauch']

# define base models
base_models = [
    ('lightgbm', LGBMRegressor(random_state=42)),
    ('random_forest', RandomForestRegressor(random_state=42)),
    ('linear_regression', LinearRegression())
]

# define stacking regressor
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=LinearRegression(),
    cv=5  # cross-validation
)

# train ensemble model
stacking_model.fit(X_train, y_train)

# save ensemble model
ensemble_model_path = "models/ensemble_model.pkl"
joblib.dump(stacking_model, ensemble_model_path)

# make predictions
y_pred = stacking_model.predict(X_test)

# evaluate model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print metrics
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'R-squared (R²): {r2:.4f}')

# show actual vs predicted
print("\nActual vs Predicted Values:")
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())

# plot observed vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolors='k', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # identity line
plt.title('Predicted vs Actual Water Consumption')
plt.xlabel('Actual Water Consumption')
plt.ylabel('Predicted Water Consumption')
plt.grid(True)
plt.show()
