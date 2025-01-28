# imports
import sys
import os
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import joblib
import datetime

import yaml

# frequency config
from data_modeling.helpers.frequency_config import AGGREGATION_CONFIG

# paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from data_exploration.data_loader import load_and_clean_data

base_path = os.getcwd()
model_name = "random_forest"
model_version = "v1.0"

model_output_dir = os.path.join(base_path, "data", "output", "model", model_name)
predictions_dir = os.path.join(model_output_dir, "predictions")
metrics_dir = os.path.join(model_output_dir, "metrics")
shap_dir = os.path.join(model_output_dir, "shap_values")
viz_dir = os.path.join(model_output_dir, "visualizations")
dashboard_dir = os.path.join(model_output_dir, "dashboard")

# target column
target_col = 'Wasserverbrauch'

# hyperparameters
tscv_default_splits = 5
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# function to display available features and get user selection
def get_user_selected_features(available_features):
    print("\nAvailable Features:")
    for idx, feature in enumerate(available_features, start=1):
        print(f"{idx}. {feature}")
    
    print("\nFeature Selection Options:")
    print("1. Select features to INCLUDE")
    print("2. Select features to EXCLUDE")
    print("3. Keep feature selection AS IS")
    selection_option = input("Choose an option (1, 2 or 3): ").strip()
    
    if selection_option == '1':
        selected = input("Enter the feature names to INCLUDE, separated by commas: ").strip()
        selected_features = [feat.strip() for feat in selected.split(',')]
        valid_features = [feat for feat in selected_features if feat in available_features]
        invalid_features = set(selected_features) - set(valid_features)
        if invalid_features:
            print(f"Warning: The following features are invalid and will be ignored: {invalid_features}")
        return valid_features
    elif selection_option == '2':
        excluded = input("Enter the feature names to EXCLUDE, separated by commas: ").strip()
        excluded_features = [feat.strip() for feat in excluded.split(',')]
        valid_exclusions = set(excluded_features) & set(available_features)
        remaining_features = [feat for feat in available_features if feat not in valid_exclusions]
        invalid_exclusions = set(excluded_features) - set(valid_exclusions)
        if invalid_exclusions:
            print(f"Warning: The following features are invalid and will be ignored: {invalid_exclusions}")
        return remaining_features
    elif selection_option == '3':
        print("Option 3 selected. Proceeding with all available features.")
        return list(available_features)
    else:
        print("Invalid option selected. Proceeding with all available features.")
        return list(available_features)

# function to train/test on each frequency
def train_and_evaluate_for_frequency(freq):
    freq_config = AGGREGATION_CONFIG[freq]
    
    freq_dir = os.path.join(model_output_dir, freq)
    freq_predictions_dir = os.path.join(freq_dir, "predictions")
    freq_metrics_dir = os.path.join(freq_dir, "metrics")
    freq_shap_dir = os.path.join(freq_dir, "shap_values")
    freq_viz_dir = os.path.join(freq_dir, "visualizations")
    freq_dashboard_dir = os.path.join(freq_dir, "dashboard")
    for d in [freq_dir, freq_predictions_dir, freq_metrics_dir, freq_shap_dir, freq_viz_dir, freq_dashboard_dir]:
        os.makedirs(d, exist_ok=True)

    # input file path
    input_file = freq_config["input_file"]
    frequency = freq_config["frequency"]
    with_lag_features = freq_config.get("with_lag_features", True)
    lag_periods = freq_config.get("lag_periods", 7)

    # if yearly, reduce splits (example)
    if freq == "yearly":
        config_n_splits = 2
    else:
        config_n_splits = tscv_default_splits

    # load and clean data
    df_freq = load_and_clean_data(
        file_path=input_file, 
        frequency=frequency,
        with_lag_features=with_lag_features,
        lag_periods=lag_periods,
        days_since_rain=True
    )
    df_freq.index = pd.to_datetime(df_freq.index)

    # ensure target column exists
    if target_col not in df_freq.columns:
        print(f"Error: Target column '{target_col}' not found in the data.")
        return

    # define available features (exclude target)
    available_features = [col for col in df_freq.columns if col != target_col]

    # user feature selection
    selected_features = get_user_selected_features(available_features)
    if not selected_features:
        print("No features selected. Exiting the training for this frequency.")
        return

    # relevant dataframe
    df_freq_relevant = df_freq[selected_features + [target_col]]

    # save dashboard data
    dash_path = os.path.join(freq_dashboard_dir, f"{model_name}_dashboard_table_{model_version}_{freq}.csv")
    df_freq_relevant.to_csv(dash_path)

    # split data
    train_mask = freq_config["train_split"](df_freq_relevant)
    test_mask  = freq_config["test_split"](df_freq_relevant)

    df_freq_train = df_freq_relevant[train_mask]
    df_freq_test  = df_freq_relevant[test_mask]

    Xf_train = df_freq_train.drop(columns=[target_col])
    yf_train = df_freq_train[target_col]
    Xf_test  = df_freq_test.drop(columns=[target_col])
    yf_test  = df_freq_test[target_col]

    # time series split
    tscv = TimeSeriesSplit(n_splits=config_n_splits)

    # initialize random forest model
    rf_model = RandomForestRegressor(random_state=42)

    # grid search with cross-validation
    grid_search_freq = GridSearchCV(
        estimator=rf_model, 
        param_grid=param_grid, 
        cv=tscv, 
        n_jobs=-1, 
        verbose=2
    )

    try:
        grid_search_freq.fit(Xf_train, yf_train)
    except ValueError as e:
        print(f"Error during GridSearchCV for frequency '{freq}': {e}")
        return

    # best model
    best_rf_model = grid_search_freq.best_estimator_

    # predictions
    yf_pred = best_rf_model.predict(Xf_test)

    # metrics
    mse_f = mean_squared_error(yf_test, yf_pred)
    mae_f = mean_absolute_error(yf_test, yf_pred)
    rmse_f = np.sqrt(mse_f)
    r2_f = r2_score(yf_test, yf_pred)

    print(f"\nTest MSE:    {mse_f:.2f}")
    print(f"Test MAE:    {mae_f:.2f}")
    print(f"Test RMSE:   {rmse_f:.2f}")
    print(f"Test R^2:    {r2_f:.2f}")

    # comparison dataframe
    comparison_df_f = pd.DataFrame({
        'Date': yf_test.index, 
        'Actual': yf_test.values, 
        'Predicted': yf_pred
    })
    comparison_df_f.set_index('Date', inplace=True)

    # save predictions
    freq_predictions_path = os.path.join(freq_predictions_dir, f"{model_name}_predictions_{model_version}_{freq}.csv")
    comparison_df_f.to_csv(freq_predictions_path, sep=";")
    print(f"Predictions saved to: {freq_predictions_path}")

    # save metrics
    freq_metrics_path = os.path.join(freq_metrics_dir, f"{model_name}_metrics_{model_version}_{freq}.csv")
    metrics_df_f = pd.DataFrame([{
        'MSE':  mse_f, 
        'MAE':  mae_f, 
        'RMSE': rmse_f, 
        'R2':   r2_f
    }])
    metrics_df_f.to_csv(freq_metrics_path, index=False, sep=";")
    print(f"Metrics saved to: {freq_metrics_path}")

    # shap
    explainer_f = shap.TreeExplainer(best_rf_model)
    shap_values_f = explainer_f.shap_values(Xf_test)
    shap_df_f = pd.DataFrame(shap_values_f, columns=Xf_test.columns)
    freq_shap_path = os.path.join(freq_shap_dir, f"{model_name}_shap_values_{model_version}_{freq}.csv")
    shap_df_f.to_csv(freq_shap_path, index=False, sep=";")
    print(f"SHAP values saved to: {freq_shap_path}")

    # plot SHAP summary
    shap.summary_plot(shap_values_f, Xf_test, show=False)
    freq_shap_plot = os.path.join(freq_viz_dir, f"{model_name}_shap_summary_{model_version}_{freq}.png")
    plt.savefig(freq_shap_plot, dpi=300, bbox_inches='tight')
    plt.close()

    # plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.plot(yf_test.index, yf_test, label='Observed')
    plt.plot(yf_test.index, yf_pred, label='Predicted', color='red')
    plt.legend()
    plt.title(f'Random Forest Predictions vs Observed Values ({freq} Test Set)')

    freq_pred_plot = os.path.join(freq_viz_dir, f"{model_name}_predictions_plot_{model_version}_{freq}.png")
    plt.savefig(freq_pred_plot, dpi=300, bbox_inches='tight')
    plt.close()

    # plot residuals
    residuals_f = yf_test - yf_pred
    plt.figure(figsize=(10, 6))
    plt.plot(yf_test.index, residuals_f, label='Residuals', color='purple')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.legend()
    plt.title(f'Residuals (Observed - Predicted) ({freq})')

    freq_resid_plot = os.path.join(freq_viz_dir, f"{model_name}_residuals_plot_{model_version}_{freq}.png")
    plt.savefig(freq_resid_plot, dpi=300, bbox_inches='tight')
    plt.close()

    # feature importances
    importances_f = best_rf_model.feature_importances_
    indices_f = np.argsort(importances_f)[::-1]
    features_f = Xf_train.columns

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.barh(range(len(indices_f)), importances_f[indices_f], align="center")
    plt.yticks(range(len(indices_f)), features_f[indices_f])
    plt.xlabel("Relative Importance")
    plt.gca().invert_yaxis()
    freq_fi_plot = os.path.join(freq_viz_dir, f"{model_name}_feature_importances_{model_version}_{freq}.png")
    plt.savefig(freq_fi_plot, dpi=300, bbox_inches='tight')
    plt.close()

    # save the model
    model_filename = f"{model_name}_model_{model_version}_{freq}.pkl"
    model_save_path = os.path.join(base_path, "models", f"{model_name}", model_filename)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Ensure the models directory exists
    joblib.dump(best_rf_model, model_save_path)
    print(f"Model saved to {model_save_path}")
    print("-----")

    yaml_metadata = {
    "model_options": [
        {
            "name": f"{model_name} {model_version}",
            "model_path": os.path.relpath(model_save_path, base_path),
            "data_path": os.path.relpath(dash_path, base_path),
            "expected_features": list(features_f)
        }
        ]
    }

    yaml_file_name = f"{model_name}_config_{model_version}_{freq}.yaml"
    yaml_file_path = os.path.join(freq_dashboard_dir, yaml_file_name)

    with open(yaml_file_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_metadata, f, allow_unicode=True, sort_keys=False)

    print(f"Generated YAML config at: {yaml_file_path}")
    print("-----")

# run for frequency
selected_freq = 'daily'
if selected_freq in AGGREGATION_CONFIG:
    train_and_evaluate_for_frequency(selected_freq)
