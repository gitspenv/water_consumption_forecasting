import pandas as pd
from data_exploration.plot_functions import (
    plot_aggregated_data,
    plot_correlation_matrix,
    plot_moving_average,
    plot_acf_pacf,
    plot_feature_correlation_with_target)

def perform_data_exploration(df):
    # Aggregated Data
    plot_aggregated_data(df, days=7, column='Wasserverbrauch')
    
    # Moving Average
    plot_moving_average(df, column='Wasserverbrauch', window=30)
    
    # Correlation Matrix
    plot_correlation_matrix(df, annot=True, cmap='coolwarm', fmt='.2f')
    
    # ACF/PACF Plot
    plot_acf_pacf(df, column='Wasserverbrauch', lags=50)

    # ACF/PACF Plot
    plot_feature_correlation_with_target(df, target_col='Wasserverbrauch', top_n=20)
