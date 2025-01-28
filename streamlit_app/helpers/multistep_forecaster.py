# imports
import pandas as pd
import numpy as np
from datetime import timedelta

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# forecast multi-step with LightGBM
def forecast_multi_step_lightgbm(model, df, expected_features, forecast_days=7, target_col="Wasserverbrauch"):
    df_forecast = df.copy()
    # get last date
    last_date = df_forecast.index.max()

    forecast_results = []
    for step in range(1, forecast_days+1):
        # new forecast date
        new_date = last_date + pd.Timedelta(days=step)

        # get past 7 days
        past_7 = df_forecast[target_col].iloc[-7:]

        future_row = {}
        # create lag features
        for i in range(1, 8):
            future_row[f"lag_{i}"] = past_7.iloc[-i] if len(past_7) >= i else df_forecast[target_col].mean()
        
        # add rolling stats
        future_row["rolling_mean_7"] = past_7.mean()
        future_row["rolling_std_3"]  = past_7.tail(3).std() if len(past_7)>=3 else 0.0

        # add time features
        future_row["month"]     = new_date.month
        future_row["weekday"]   = new_date.dayofweek
        future_row["year"]      = new_date.year
        day_of_year            = new_date.dayofyear
        future_row["day_of_year"] = day_of_year
        future_row["sin_day_of_year"] = np.sin(2*np.pi*day_of_year/365)
        future_row["cos_day_of_year"] = np.cos(2*np.pi*day_of_year/365)
        future_df = pd.DataFrame([future_row], index=[new_date], columns=expected_features)

        # predict
        y_pred = model.predict(future_df)[0]
        forecast_results.append((new_date, y_pred))

        # add prediction to dataframe
        new_row = pd.DataFrame({target_col: y_pred}, index=[new_date])
        df_forecast = pd.concat([df_forecast, new_row], axis=0)

    # create forecast dataframe
    forecast_df = pd.DataFrame(forecast_results, columns=["Date","Forecast"]).set_index("Date")
    return forecast_df


# forecast multi-step with SARIMA
def forecast_multi_step_sarima(model, df, forecast_days=7, exog=None):

    # make forecast
    forecast_result = model.forecast(steps=forecast_days, exog=exog)

    # build forecast dataframe
    if isinstance(forecast_result, pd.Series):
        forecast_df = forecast_result.to_frame(name="Forecast")
    else:
        date_start = df.index.max() + pd.Timedelta(days=1)
        date_index = pd.date_range(date_start, periods=forecast_days, freq='D')
        forecast_df = pd.DataFrame({
            "Forecast": forecast_result
        }, index=date_index)

    return forecast_df
