# imports
import pandas as pd
import os

def merge_files(file1_path, file2_path, output_path):
    # read files
    df1 = pd.read_csv(file1_path, delimiter=";")
    df2 = pd.read_csv(file2_path, delimiter=";")
    
    # convert dates
    df1['Datum'] = pd.to_datetime(df1['Datum'], format='%Y-%m-%d')
    df2['Datum'] = pd.to_datetime(df2['Datum'], format='%Y-%m-%d')

    # merge dataframes
    merged_df = pd.merge(df1, df2, on='Datum', how='outer')

    # create output filename
    file1_name = os.path.basename(file1_path).split('.')[0]
    file2_name = os.path.basename(file2_path).split('.')[0]
    output_filename = f"{file1_name}_join_{file2_name}.csv"

    # set output path
    output_path = os.path.join(output_folder, output_filename)

    # save merged data
    merged_df.to_csv(output_path, sep=";", index=False)

    # print message
    print(f"Merged file saved to: {output_path}")

# paths
base_path = os.getcwd()

file1_path = os.path.join(base_path, 'data', 'processed', 'water', 'water_consumption_2015_2023_monthly_normalized.csv')
file2_path = os.path.join(base_path, 'data', 'processed', 'population', 'population_monthly.csv')

# output folder
output_folder = os.path.join(base_path, 'data', 'output', 'water')

# run merge
merge_files(file1_path, file2_path, output_folder)
