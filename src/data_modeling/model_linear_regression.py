# imports
import sys
import os
import yaml
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import joblib  # We'll use joblib or pickle to save the OLS results
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

from helpers.frequency_config import AGGREGATION_CONFIG

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from data_exploration.data_loader import load_and_clean_data

base_path = os.getcwd()
model_name = "linear_regression"
model_version = "v1.0"

model_output_dir = os.path.join(base_path, "data", "output", "model", model_name)
predictions_dir = os.path.join(model_output_dir, "predictions")
metrics_dir = os.path.join(model_output_dir, "metrics")
viz_dir = os.path.join(model_output_dir, "visualizations")
dashboard_dir = os.path.join(model_output_dir, "dashboard")


# target column
target_col = 'Wasserverbrauch'

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

def train_and_evaluate_for_frequency(freq):
    # folder structure
    freq_dir = os.path.join(model_output_dir, freq)
    freq_predictions_dir = os.path.join(freq_dir, "predictions")
    freq_metrics_dir = os.path.join(freq_dir, "metrics")
    freq_viz_dir = os.path.join(freq_dir, "visualizations")
    freq_dashboard_dir = os.path.join(freq_dir, "dashboard")

    for fdir in [freq_dir, freq_predictions_dir, freq_metrics_dir, freq_viz_dir, freq_dashboard_dir]:
        os.makedirs(fdir, exist_ok=True)

    freq_config = AGGREGATION_CONFIG[freq]
    input_file = freq_config["input_file"]
    frequency = freq_config["frequency"]
    with_lag_features = freq_config.get("with_lag_features", True)
    lag_periods = freq_config.get("lag_periods", 7)

    # load data
    df_cleaned = load_and_clean_data(
        file_path=input_file, 
        frequency=frequency,
        with_lag_features=with_lag_features,
        lag_periods=lag_periods,
    )
    df_cleaned.index = pd.to_datetime(df_cleaned.index)

    if target_col not in df_cleaned.columns:
        print(f"Error: '{target_col}' not in df.")
        return

    # user picks features
    available_features = [c for c in df_cleaned.columns if c != target_col]
    selected_features = get_user_selected_features(available_features)
    if not selected_features:
        print("No features selected. Exiting.")
        return

    # relevant df
    df_relevant = df_cleaned[selected_features + [target_col]]

    # save a "dashboard" CSV
    dash_path = os.path.join(freq_dashboard_dir, f"{model_name}_dashboard_table_{model_version}_{freq}.csv")
    df_relevant.to_csv(dash_path)

    # split data using freq_config
    train_mask = freq_config["train_split"](df_relevant)
    test_mask  = freq_config["test_split"](df_relevant)

    df_train = df_relevant[train_mask]
    df_test  = df_relevant[test_mask]

    if len(df_train) < 10 or len(df_test) < 1:
        print(f"Not enough data in train or test for freq '{freq}'.")
        return

    # define X, y
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_test  = df_test.drop(columns=[target_col])
    y_test  = df_test[target_col]

    # add intercept
    X_train = sm.add_constant(X_train, has_constant='add')
    X_test  = sm.add_constant(X_test, has_constant='add')

    # fit OLS
    model_ols = sm.OLS(y_train, X_train)
    results_ols = model_ols.fit()

    # predict
    y_pred = results_ols.predict(X_test)

    # metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\nTest MSE:  {mse:.2f}")
    print(f"Test MAE:  {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R^2:  {r2:.2f}")

    # save predictions
    comp_df = pd.DataFrame({"Date": y_test.index, "Actual": y_test, "Predicted": y_pred})
    comp_df.set_index("Date", inplace=True)
    pred_path = os.path.join(freq_predictions_dir, f"{model_name}_predictions_{model_version}_{freq}.csv")
    comp_df.to_csv(pred_path, sep=";")
    print(f"Predictions saved to: {pred_path}")

    # save metrics
    metrics_path = os.path.join(freq_metrics_dir, f"{model_name}_metrics_{model_version}_{freq}.csv")
    mdf = pd.DataFrame([{"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2}])
    mdf.to_csv(metrics_path, sep=";", index=False)
    print(f"Metrics saved to: {metrics_path}")

    # plot predictions vs. actual
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label="Observed", color="blue")
    plt.plot(y_test.index, y_pred, label="Predicted", color="red", linestyle="--")
    plt.title(f"OLS Observed vs Predicted ({freq} Test)")
    plt.legend()
    pred_plot_path = os.path.join(freq_viz_dir, f"{model_name}_predictions_plot_{model_version}_{freq}.png")
    plt.savefig(pred_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.plot(residuals.index, residuals, label="Residuals", color="purple")
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title(f"Residuals ({freq})")
    plt.legend()
    resid_plot_path = os.path.join(freq_viz_dir, f"{model_name}_residuals_plot_{model_version}_{freq}.png")
    plt.savefig(resid_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    summary_str = results_ols.summary().as_text()
    summary_path = os.path.join(freq_viz_dir, f"{model_name}_summary_{model_version}_{freq}.txt")
    with open(summary_path, "w") as f:
        f.write(summary_str)

    # save the model
    model_filename = f"{model_name}_model_{model_version}_{freq}.pkl"
    model_save_path = os.path.join(base_path, "models", model_name, model_filename)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(results_ols, model_save_path)
    print(f"Model saved to {model_save_path}")
    print("-----")

    # generate YAML with final features
    yaml_metadata = {
        "model_options": [
            {
                "name": f"{model_name} {model_version}",
                "model_path": os.path.relpath(model_save_path, base_path),
                "data_path": os.path.relpath(dash_path, base_path),
                "expected_features": list(X_train.columns)  # includes 'const' unless you remove it
            }
        ]
    }

    yaml_file_name = f"{model_name}_config_{model_version}_{freq}.yaml"
    yaml_file_path = os.path.join(freq_dashboard_dir, yaml_file_name)
    with open(yaml_file_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_metadata, f, allow_unicode=True, sort_keys=False)
    print(f"Generated YAML config at: {yaml_file_path}")
    print("-----")

# run
selected_freq = "daily"
if selected_freq in AGGREGATION_CONFIG:
    train_and_evaluate_for_frequency(selected_freq)
