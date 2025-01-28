# imports
import sys
import os
import shutil
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import joblib
import shap

from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yaml

from helpers.frequency_config import AGGREGATION_CONFIG

# paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from data_exploration.data_loader import load_and_clean_data
from data_exploration.plot_functions import plot_correlation_matrix, plot_feature_correlation_with_target

base_path = os.getcwd()
model_name = "lightgbm" 
model_version = "v1.1"  

model_output_dir = os.path.join(base_path, "data", "output", "model", model_name)
predictions_dir = os.path.join(model_output_dir, "predictions")
metrics_dir = os.path.join(model_output_dir, "metrics")
shap_dir = os.path.join(model_output_dir, "shap_values")
viz_dir = os.path.join(model_output_dir, "visualizations")
dashboard_dir = os.path.join(model_output_dir, "dashboard")

# target column
target_col = 'Wasserverbrauch'

# hyperparams
param_grid = {
    'n_estimators':      [100, 200, 300, 400],
    'learning_rate':     [0.01, 0.05, 0.1, 0.2],
    'num_leaves':        [20, 31, 40, 50],
    'max_depth':         [-1, 10, 20, 30],
    'min_child_samples': [5, 10, 20]
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
        # Validate selected features
        valid_features = [feat for feat in selected_features if feat in available_features]
        invalid_features = set(selected_features) - set(valid_features)
        if invalid_features:
            print(f"Warning: The following features are invalid and will be ignored: {invalid_features}")
        return valid_features
    elif selection_option == '2':
        excluded = input("Enter the feature names to EXCLUDE, separated by commas: ").strip()
        excluded_features = [feat.strip() for feat in excluded.split(',')]
        # Validate excluded features
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

    # input file path and frequency
    input_file_path = freq_config["input_file"]
    frequency = freq_config["frequency"]  
    with_lag_features = freq_config["with_lag_features"]
    if freq == "yearly":
        config_n_splits = 2
        lag_periods= 0
    else:
        lag_periods = freq_config.get("lag_periods", 7)
        config_n_splits = 5

    # load and clean data
    df_cleaned = load_and_clean_data(
        file_path=input_file_path,
        frequency=frequency, 
        with_lag_features=with_lag_features,
        lag_periods=lag_periods,
        add_time_features=True,
        add_weather_features=True,
        add_vacation_days=True
    )
    
    print(f"\nTotal samples after cleaning: {df_cleaned.shape[0]}")

    df_cleaned.index = pd.to_datetime(df_cleaned.index)
    
    # ensure target column exists
    if target_col not in df_cleaned.columns:
        print(f"Error: Target column '{target_col}' not found in the data.")
        return
    
    # define available features
    available_features = [col for col in df_cleaned.columns if col != target_col]

    # plot correlation matrices

    plot_correlation_matrix(df_cleaned)
    plot_feature_correlation_with_target(df_cleaned)
    
    # feature selection
    selected_features = get_user_selected_features(available_features)
    
    if not selected_features:
        print("No features selected. Exiting the training for this frequency.")
        return
    
    # f_relevant with selected features and target
    df_relevant = df_cleaned[selected_features + [target_col]]
    print(f"Total relevant samples: {df_relevant.shape[0]}")

    # dashboard
    dashboard_file_path = os.path.join(freq_dashboard_dir, f"{model_name}_dashboard_table_{model_version}_{freq}.csv")
    df_relevant.to_csv(dashboard_file_path)

    print(f"df_relevant.head():\n{df_relevant.head()}")
    print(f"df_relevant.isna().sum():\n{df_relevant.isna().sum()}")
    
    # split data
    train_mask = freq_config["train_split"](df_relevant)
    test_mask  = freq_config["test_split"](df_relevant)

    train_data = df_relevant[train_mask]
    test_data  = df_relevant[test_mask]

    # check training and testing samples
    print(f"Training samples for '{freq}': {train_data.shape[0]}")
    print(f"Testing samples for '{freq}': {test_data.shape[0]}")

    required_splits = config_n_splits

    # initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=required_splits)

    # extract target and features
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_test  = test_data.drop(columns=[target_col])
    y_test  = test_data[target_col]

    # define feature_cols based on available columns
    feature_cols = list(X_train.columns)
    print(f"Using features: {feature_cols}")

    # model
    lgbm_model = LGBMRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=lgbm_model, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=2)
    
    try:
        grid_search.fit(X_train, y_train)
    except ValueError as e:
        print(f"Error during GridSearchCV for frequency '{freq}': {e}")
        return

    best_lgbm_model = grid_search.best_estimator_

    # predictions
    y_pred = best_lgbm_model.predict(X_test)

    # calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # print metrics
    print(f"\nTest MSE:    {mse:.2f}")
    print(f"Test MAE:    {mae:.2f}")
    print(f"Test RMSE:   {rmse:.2f}")
    print(f"Test R^2:    {r2:.2f}")

    # save predictions
    comparison_df = pd.DataFrame({'Date': y_test.index, 'Actual': y_test.values, 'Predicted': y_pred})
    comparison_df.set_index('Date', inplace=True)

    predictions_path = os.path.join(freq_predictions_dir, f"{model_name}_predictions_{model_version}_{freq}.csv")
    comparison_df.to_csv(predictions_path, sep=";")
    print(f"Predictions saved to: {predictions_path}")

    # save metrics
    metrics_path = os.path.join(freq_metrics_dir, f"{model_name}_metrics_{model_version}_{freq}.csv")
    metrics_df = pd.DataFrame([{'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}])
    metrics_df.to_csv(metrics_path, index=False, sep=";")
    print(f"Metrics saved to: {metrics_path}")

    # calculate SHAP
    explainer = shap.TreeExplainer(best_lgbm_model)
    shap_values = explainer.shap_values(X_test)
    shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
    shap_values_path = os.path.join(freq_shap_dir, f"{model_name}_shap_values_{model_version}_{freq}.csv")
    shap_df.to_csv(shap_values_path, index=False, sep=";")
    print(f"SHAP values saved to: {shap_values_path}")

    # plot SHAP summary
    shap.summary_plot(shap_values, X_test, show=False)
    shap_plot_path = os.path.join(freq_viz_dir, f"{model_name}_shap_summary_{model_version}_{freq}.png")
    plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to prevent display

    # plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label='Observed')
    plt.plot(y_test.index, y_pred, label='Predicted', color='red')
    plt.legend()
    plt.title(f'LightGBM Predictions vs Observed Values ({freq} Test Set)')

    pred_plot_path = os.path.join(freq_viz_dir, f"{model_name}_predictions_plot_{model_version}_{freq}.png")
    plt.savefig(pred_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, residuals, label='Residuals', color='purple')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.legend()
    plt.title(f'Residuals (Observed - Predicted) ({freq})')

    resid_plot_path = os.path.join(freq_viz_dir, f"{model_name}_residuals_plot_{model_version}_{freq}.png")
    plt.savefig(resid_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # feature importances
    importances = best_lgbm_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_names = [feature_cols[i] for i in indices]

    # plot feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), feat_names)
    plt.xlabel("Relative Importance")
    plt.gca().invert_yaxis()
    fi_plot_path = os.path.join(freq_viz_dir, f"{model_name}_feature_importances_{model_version}_{freq}.png")
    plt.savefig(fi_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # save the model
    model_filename = f"{model_name}_model_{model_version}_{freq}.pkl"
    model_save_path = os.path.join(base_path, "models", f"{model_name}", model_filename)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(best_lgbm_model, model_save_path)
    print(f"Model saved to {model_save_path}")

    yaml_metadata = {
    "model_options": [
        {
            "name": f"{model_name} {model_version}",
            "model_path": os.path.relpath(model_save_path, base_path),
            "data_path": os.path.relpath(dashboard_file_path, base_path),
            "expected_features": feature_cols 
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
