# imports
import pandas as pd
import glob
import os

# paths
base_path = os.getcwd()
path = os.path.join(base_path, "data", "raw", "meteo", "*.csv")
all_files = glob.glob(path)

# read csv files
df_list = [pd.read_csv(file) for file in all_files]
df = pd.concat(df_list, ignore_index=True)

# pivot data
df_pivoted = df.pivot_table(index=['Datum', 'Standort', 'Intervall'], 
                            columns=['Parameter', 'Einheit'], 
                            values='Wert').reset_index()

# flatten columns
df_pivoted.columns = [f"{i}_{j}" if j else i for i, j in df_pivoted.columns]

# convert dates
df_pivoted['Datum'] = pd.to_datetime(df_pivoted['Datum']).dt.strftime('%d.%m.%Y')

# filter locations
excluded_locations = ['Zch_Rosengartenstrasse', 'Zch_Schimmelstrasse']
df_filtered = df_pivoted[~df_pivoted['Standort'].isin(excluded_locations)]

# drop columns
columns_to_drop = ['Intervall', 'Standort']
df_filtered = df_filtered.drop(columns=columns_to_drop, errors='ignore')

# save data
output_path = os.path.join(base_path, 'data', 'processed', 'meteo', 'filtered_meteo_data.csv')
df_filtered.to_csv(output_path, index=False)

print("Filtered meteo data saved to:", output_path)
