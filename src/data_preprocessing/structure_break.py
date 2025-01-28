# imports
import sys
import os
import pandas as pd

# add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# import custom functions
from data_exploration.data_loader import load_and_clean_data
from data_exploration.plot_functions import plot_aggregated_data

# paths
base_path = os.getcwd()
input_file_path = os.path.join(base_path, "water_data", "input", "water_consumption_2015_2023.csv")

# load and clean data
df_cleaned = load_and_clean_data(input_file_path)

# define break period
break_start = "2020-01-01"
break_end = "2020-12-31"

# find break period rows
mask_break_period = (df_cleaned.index >= break_start) & (df_cleaned.index <= break_end)

# calculate means before and after break
mean_before_break = df_cleaned.loc[df_cleaned.index < break_start, 'Wasserverbrauch'].mean()
mean_after_break = df_cleaned.loc[df_cleaned.index > break_end, 'Wasserverbrauch'].mean()
mean_adjustment_factor = (mean_before_break + mean_after_break) / 2

# adjust wasserverbrauch during break
df_cleaned.loc[mask_break_period, 'Wasserverbrauch'] = (
    df_cleaned.loc[mask_break_period, 'Wasserverbrauch']
    / df_cleaned.loc[mask_break_period, 'Wasserverbrauch'].mean()
    * mean_adjustment_factor
)

# update the dataframe
df_cleaned.update(df_cleaned.loc[mask_break_period])

# save adjusted data
output_path = os.path.join(base_path, 'water_data', 'output', 'water_consumption_2015_2023_normalized.csv')
df_cleaned.to_csv(output_path, index=True, sep=";")

# print adjusted data
print("Adjusted Data for Structural Break Period:")
print(df_cleaned.loc[mask_break_period])

# plot the data
plot_aggregated_data(df_cleaned)
