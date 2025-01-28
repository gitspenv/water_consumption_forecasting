# imports
import sys
import os
import yaml
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from helpers.frequency_config import AGGREGATION_CONFIG

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from data_exploration.data_loader import load_and_clean_data
from data_exploration.plot_functions import plot_forecast

# define base paths
base_path = os.getcwd()
model_name = "sarima"
model_version = "v1.1"

model_output_dir = os.path.join(base_path, "data", "output", "model", model_name)
predictions_dir = os.path.join(model_output_dir, "predictions")
metrics_dir = os.path.join(model_output_dir, "metrics")
viz_dir = os.path.join(model_output_dir, "visualizations")
dashboard_dir = os.path.join(model_output_dir, "dashboard")

target_col = 'Wasserverbrauch'

# function to train/test for a given frequency
def train_and_evaluate_for_frequency(freq):
    freq_config = AGGREGATION_CONFIG[freq]

    # set up local directories
    freq_dir = os.path.join(model_output_dir, freq)
    freq_predictions_dir = os.path.join(freq_dir, "predictions")
    freq_metrics_dir = os.path.join(freq_dir, "metrics")
    freq_viz_dir = os.path.join(freq_dir, "visualizations")
    freq_dashboard_dir = os.path.join(freq_dir, "dashboard")
    for d in [freq_dir, freq_predictions_dir, freq_metrics_dir, freq_viz_dir, freq_dashboard_dir]:
        os.makedirs(d, exist_ok=True)

    # input file path and frequency
    freq_mode = freq_config["frequency"]
    input_file_path = freq_config["input_file"]

    if freq_mode == "daily":
        loader_freq = 'D'  
    elif freq_mode == "monthly":
        loader_freq = 'ME' 
    else:
        loader_freq = 'D' 

    # load data
    df_cleaned = load_and_clean_data(
        file_path=input_file_path,
        frequency=loader_freq,  
        with_lag_features=False,
        lag_periods=0
    )
    print(f"\nLoaded data with shape: {df_cleaned.shape}")

    if freq_mode == "daily":
        df_cleaned.index = pd.DatetimeIndex(df_cleaned.index)  
    elif freq_mode == "monthly":
        df_cleaned.index = pd.DatetimeIndex(df_cleaned.index).to_period('M')

    # pick feature cols
    exog_cols = ['T_C', 'Zuzüge']
    # drop nan
    df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col] + exog_cols)
    print("Data shape after cleaning:", df_cleaned.shape)

    # stationarity check
    adf_result = adfuller(df_cleaned[target_col])
    print(f"ADF statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")

    # plot and save ACF/PACF
    acf_fig, acf_axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_acf(df_cleaned[target_col], lags=24, ax=acf_axes[0])
    plot_pacf(df_cleaned[target_col], lags=24, ax=acf_axes[1])
    acf_fig.suptitle("ACF and PACF")
    acf_plot_path = os.path.join(freq_viz_dir, f"{model_name}_acf_pacf_{model_version}_{freq}.png")
    plt.savefig(acf_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # train/test split
    train_mask = freq_config["train_split"](df_cleaned)
    test_mask  = freq_config["test_split"](df_cleaned)
    train_data = df_cleaned[train_mask]
    test_data  = df_cleaned[test_mask]

    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    endog_train = train_data[target_col]
    endog_test  = test_data[target_col]
    exog_train  = train_data[exog_cols]
    exog_test   = test_data[exog_cols]

    # final params
    order = (1, 1, 0)
    seasonal_order = (0, 1, 1, 7)

    print(f"Using SARIMA{order}x{seasonal_order} with exogs {exog_cols}")
    model = sm.tsa.SARIMAX(
        endog_train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    print(results.summary())

    steps_ahead = len(endog_test)
    forecast_res = results.get_forecast(steps=steps_ahead, exog=exog_test)
    forecast_mean = forecast_res.predicted_mean
    forecast_ci   = forecast_res.conf_int()

    if freq_mode == "monthly":
        forecast_mean.index = forecast_mean.index.to_timestamp()
        forecast_ci.index   = forecast_ci.index.to_timestamp()
        train_data_plot     = train_data.copy()
        train_data_plot.index = train_data_plot.index.to_timestamp()
    else:
        train_data_plot = train_data.copy()

    # plot forecast
    plot_forecast(train_data_plot, forecast_mean, forecast_ci, observed_column=target_col)

    # evaluate
    mae_val = mean_absolute_error(endog_test, forecast_mean)
    mse_val = mean_squared_error(endog_test, forecast_mean)
    rmse_val = np.sqrt(mse_val)
    print(f"\nMAE={mae_val:.2f}, MSE={mse_val:.2f}, RMSE={rmse_val:.2f}")

    # save predictions
    # convert test index to timestamps if monthly
    if freq_mode == "monthly":
        test_index = endog_test.index.to_timestamp()
    else:
        test_index = endog_test.index

    pred_df = pd.DataFrame({
        'Date': test_index,
        'Actual': endog_test.values,
        'Predicted': forecast_mean.values
    })
    pred_df.set_index('Date', inplace=True)
    pred_path = os.path.join(freq_predictions_dir, f"{model_name}_predictions_{model_version}_{freq}.csv")
    pred_df.to_csv(pred_path, sep=";")
    print(f"Predictions saved to: {pred_path}")

    # save metrics
    metrics_path = os.path.join(freq_metrics_dir, f"{model_name}_metrics_{model_version}_{freq}.csv")
    mdf = pd.DataFrame([{
        'MAE': mae_val,
        'MSE': mse_val,
        'RMSE': rmse_val
    }])
    mdf.to_csv(metrics_path, index=False, sep=";")
    print(f"Metrics saved to: {metrics_path}")

    # plot residuals
    residuals = endog_test - forecast_mean
    if freq_mode == "monthly":
        res_index = residuals.index.to_timestamp()
    else:
        res_index = residuals.index

    plt.figure(figsize=(10, 6))
    plt.plot(res_index, residuals, label='Residuals', color='purple')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.legend()
    plt.title(f"Residuals ({freq})")
    resid_plot_path = os.path.join(freq_viz_dir, f"{model_name}_residuals_{model_version}_{freq}.png")
    plt.savefig(resid_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # residual diagnostics
    diag_fig = results.plot_diagnostics(figsize=(10,6))
    diag_path = os.path.join(freq_viz_dir, f"{model_name}_diagnostics_{model_version}_{freq}.png")
    diag_fig.savefig(diag_path, dpi=300, bbox_inches='tight')
    plt.close()

    # save model
    model_filename = f"{model_name}_model_{model_version}_{freq}.pkl"
    model_save_path = os.path.join(base_path, "models", model_name, model_filename)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    results.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # create YAML config
    yaml_metadata = {
        "model_options": [
            {
                "name": f"{model_name} {model_version}",
                "model_path": os.path.relpath(model_save_path, base_path),
                "data_path": os.path.relpath(input_file_path, base_path),
                "expected_features": exog_cols,
                "order": str(order),
                "seasonal_order": str(seasonal_order),
                "freq": freq_mode
            }
        ]
    }

    yaml_file_name = f"{model_name}_config_{model_version}_{freq}.yaml"
    yaml_file_path = os.path.join(freq_dashboard_dir, yaml_file_name)

    with open(yaml_file_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_metadata, f, allow_unicode=True, sort_keys=False)

    print(f"YAML config saved to {yaml_file_path}")
    print("-----")

# run
selected_freq = "daily" 
if selected_freq in AGGREGATION_CONFIG:
    train_and_evaluate_for_frequency(selected_freq)
