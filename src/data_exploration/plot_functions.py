import matplotlib.pyplot as plt
import seaborn as sns

def plot_aggregated_data(df, days=7, column='Wasserverbrauch'):

    df_resampled = df[column].resample(f'{days}D').sum()
    plt.figure(figsize=(10, 6))
    plt.plot(df_resampled.index, df_resampled, label=f'{days}-Day Aggregated Data', color='blue')
    plt.title(f'{column} Aggregated Over {days}-Day Periods')
    plt.xlabel('Date')
    plt.ylabel(f'{column}')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_correlation_matrix(df, annot=True, cmap='coolwarm', fmt='.2f'):

    corr_matrix = df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, fmt=fmt, linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

def plot_moving_average(df, column='Wasserverbrauch', window=30):

    df['rolling_mean'] = df[column].rolling(window=window).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[column], label='Original')
    plt.plot(df.index, df['rolling_mean'], label=f'{window}-Day Rolling Mean', color='red')
    plt.legend()
    plt.title(f'{column} with {window}-Day Rolling Mean')
    plt.show()

def plot_acf_pacf(df, column='Wasserverbrauch', lags=100):

    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    plot_acf(df[column].dropna(), lags=lags)
    plot_pacf(df[column].dropna(), lags=lags)
    plt.show()

def plot_forecast(df, forecast_mean, forecast_ci, observed_column='Wasserverbrauch'):

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[observed_column], label='Observed')  # Observed data
    plt.plot(forecast_mean.index, forecast_mean, label='Forecast', color='red')  # Forecasted data
    plt.fill_between(forecast_mean.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)  # Confidence interval
    plt.legend()
    plt.title(f'{observed_column} with Forecast and Confidence Interval')
    plt.show()

def plot_feature_correlation_with_target(df, target_col='Wasserverbrauch', top_n=20):
    corr_series = df.corr()[target_col].dropna().drop(target_col, errors='ignore')
    # sort by absolute correlation
    corr_sorted = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)
    # If top_n is specified, take that slice
    if len(corr_sorted) > top_n:
        corr_sorted = corr_sorted.iloc[:top_n]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=corr_sorted.values, y=corr_sorted.index, palette='viridis')
    plt.title(f'Top Feature Correlations with {target_col}')
    plt.xlabel('Correlation')
    plt.ylabel('Feature')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.tight_layout()
    plt.show()