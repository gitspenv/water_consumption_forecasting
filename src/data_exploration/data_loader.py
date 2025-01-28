# imports
import pandas as pd
import numpy as np

def load_and_clean_data(file_path, 
                        frequency='D', 
                        with_lag_features=False, 
                        lag_periods=7,
                        add_time_features=True,
                        add_weather_features=True,
                        days_since_rain=False,
                        add_vacation_days=False
                       ):
    # load data
    df = pd.read_csv(file_path, delimiter=';', parse_dates=['Datum'], dayfirst=False)

    # filter outliers
    Q1 = df['Wasserverbrauch'].quantile(0.25)
    Q3 = df['Wasserverbrauch'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df['Wasserverbrauch'] < lower_bound) | (df['Wasserverbrauch'] > upper_bound)
    df['Wasserverbrauch'] = df['Wasserverbrauch'].where(~outliers).interpolate(method='linear')

    # set datetime index
    df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
    df.set_index('Datum', inplace=True)

    # ensure numeric columns
    df = df.apply(pd.to_numeric, errors='coerce')

    # sort and fill dates
    df = df.sort_index()

    # convert to specified frequency
    df = df.asfreq(frequency)

    # interpolate missing wasserverbrauch
    if df['Wasserverbrauch'].isna().any():
        df['Wasserverbrauch'] = df['Wasserverbrauch'].interpolate(method='linear')

    # weekend dummies (only applicable for daily agg.)
    if frequency == 'D':
        df['is_saturday'] = (df.index.dayofweek == 5).astype(int)
        df['is_sunday']   = (df.index.dayofweek == 6).astype(int)

    # add time features
    if add_time_features:
        df['month'] = df.index.month
        df['year']  = df.index.year

        if frequency in ['D', 'W']:
            df['weekday']      = df.index.weekday
            df['day_of_year']  = df.index.dayofyear
            df['week_of_year'] = df.index.isocalendar().week.astype(int)
            df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.0)
            df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.0)
        else:
            # monthly/yearly
            df['weekday'] = 0
            df['day_of_year'] = 0
            df['week_of_year'] = 0
            df['sin_day_of_year'] = 0
            df['cos_day_of_year'] = 0

    # add weather features
    if add_weather_features:
        if 'T_C' in df.columns:
            df['T_C_rolling3'] = df['T_C'].rolling(window=3).mean()
        if 'RainDur_min' in df.columns:
            df['rained_today'] = (df['RainDur_min'] > 0).astype(int)

    # holiday and vacation flags (only applicable for daily agg.)
    if frequency == 'D' and add_vacation_days:
        df['is_neujahr']       = 0
        df['is_berchtoldstag'] = 0
        df['is_tag_der_arbeit'] = 0
        df['is_weihnachtstag'] = 0
        df['is_stephanstag']   = 0
        df['is_sportferien']   = 0
        df['is_fruehlingsferien'] = 0
        df['is_sommerferien']  = 0
        df['is_herbstferien']  = 0

        # single-day holidays
        df.loc[(df.index.month == 1) & (df.index.day == 1), 'is_neujahr'] = 1
        df.loc[(df.index.month == 1) & (df.index.day == 2), 'is_berchtoldstag'] = 1
        df.loc[(df.index.month == 5) & (df.index.day == 1), 'is_tag_der_arbeit'] = 1
        df.loc[(df.index.month == 12) & (df.index.day == 25), 'is_weihnachtstag'] = 1
        df.loc[(df.index.month == 12) & (df.index.day == 26), 'is_stephanstag'] = 1

        # vacation weeks
        df.loc[df['week_of_year'].isin([7, 8]), 'is_sportferien']       = 1
        df.loc[df['week_of_year'] == 17, 'is_fruehlingsferien']         = 1
        df.loc[df['week_of_year'].between(29, 33), 'is_sommerferien']   = 1
        df.loc[df['week_of_year'].isin([41, 42]), 'is_herbstferien']    = 1

    # add lag features
    if with_lag_features:
        for i in range(1, lag_periods + 1):
            df[f'lag_{i}'] = df['Wasserverbrauch'].shift(i)
        
        # rolling features
        if frequency == 'D':
            df['rolling_mean_3'] = df['Wasserverbrauch'].rolling(window=3).mean()
            df['rolling_mean_7'] = df['Wasserverbrauch'].rolling(window=7).mean()
            df['rolling_std_3']  = df['Wasserverbrauch'].rolling(window=3).std()
        elif frequency == 'W':
            df['rolling_mean_3'] = df['Wasserverbrauch'].rolling(window=3).mean()
            df['rolling_mean_7'] = df['Wasserverbrauch'].rolling(window=7).mean()
            df['rolling_std_3']  = df['Wasserverbrauch'].rolling(window=3).std()
        elif frequency == 'ME':
            df['rolling_mean_3'] = df['Wasserverbrauch'].rolling(window=3).mean()
            df['rolling_mean_6'] = df['Wasserverbrauch'].rolling(window=7).mean()
            df['rolling_std_3']  = df['Wasserverbrauch'].rolling(window=3).std()

        # drop NaNs from lag and rolling
        df = df.dropna()

    # days since rain
    if days_since_rain:
        days_since_rain_list = []
        count = 0
        for rain in df['RainDur_min']:
            if rain > 0:
                count = 0
            else:
                count += 1
            days_since_rain_list.append(count)
        df['days_since_rain'] = days_since_rain_list
        return df

    return df
