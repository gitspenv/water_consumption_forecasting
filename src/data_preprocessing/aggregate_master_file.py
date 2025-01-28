# imports
import pandas as pd
import os

def aggregate_data(input_path, output_path, frequency='ME'):
    # read data
    df = pd.read_csv(input_path, delimiter=";")

    # convert to datetime
    df['Datum'] = pd.to_datetime(df['Datum'], format='%Y-%m-%d')

    # set index
    df.set_index('Datum', inplace=True)

    # make numeric
    df = df.apply(pd.to_numeric, errors='coerce', axis=1)

    # resample and aggregate
    df_resampled = df.resample(frequency).agg({
        'Wasserverbrauch': 'sum',
        'Wegzüge': 'sum',
        'Zuzüge': 'sum',
        'Geburte': 'sum',
        'Todesfälle': 'sum',
        'RainDur_min': 'sum',
        'StrGlo_W/m2': 'sum',
        'T_C': 'mean',
        'T_max_h1_C': 'mean',
        'p_hPa': 'mean'
    })

    # round values
    df_resampled[['T_C', 'T_max_h1_C', 'p_hPa']] = df_resampled[['T_C', 'T_max_h1_C', 'p_hPa']].round(2)

    # save to file
    df_resampled.to_csv(output_path, sep=";", index=True)

# get current directory
base_path = os.getcwd()

input_path = os.path.join(base_path, "data", "processed" , "water", "water_consumption_2015_2023_normalized.csv")
output_dir = os.path.join(base_path, "data", "processed", "water")

# define output files
output_path_yearly = os.path.join(output_dir, "water_consumption_2015_2023_yearly_normalized.csv")
output_path_monthly = os.path.join(output_dir, "water_consumption_2015_2023_monthly_normalized.csv")
output_path_weekly = os.path.join(output_dir, "water_consumption_2015_2023_weekly_normalized.csv")
output_path_daily = os.path.join(output_dir, "water_consumption_2015_2023_daily_normalized.csv")

# run aggregations
aggregate_data(input_path, output_path_yearly, frequency='A')  # yearly
aggregate_data(input_path, output_path_monthly, frequency='ME') # monthly
aggregate_data(input_path, output_path_weekly, frequency='W')  # weekly
aggregate_data(input_path, output_path_daily, frequency='D')   # daily
