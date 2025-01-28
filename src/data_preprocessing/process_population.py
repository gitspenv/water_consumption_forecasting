# imports
import pandas as pd
import os

# paths
base_path = os.getcwd()

input_path = os.path.join(base_path, 'data', 'raw', 'population', 'bevölkerung_monatlich.csv')
output_path = os.path.join(base_path, 'data', 'processed', 'population', 'population_monthly.csv')

# read data
df = pd.read_csv(input_path, delimiter=";")

# convert dates
df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y')

# extract month
df['Monat'] = df['Datum'].dt.to_period('M')

# pivot data
pivoted_data = (
    df.pivot_table(
        index='Monat', 
        columns='Herkunft', 
        values='Bevölkerung', 
        aggfunc='sum'
    )
    .reset_index()
)

# rename columns
pivoted_data.columns = ['Monat', 'Bevölkerung_Swiss', 'Bevölkerung_Foreign']

# calculate total population
pivoted_data['Bevölkerung_total'] = pivoted_data['Bevölkerung_Swiss'] + pivoted_data['Bevölkerung_Foreign']

# set date to last day of month
pivoted_data['Datum'] = pivoted_data['Monat'].dt.to_timestamp('M')

# drop Monat column
pivoted_data.drop(columns=['Monat'], inplace=True)

# reorder columns
pivoted_data = pivoted_data[['Datum', 'Bevölkerung_Swiss', 'Bevölkerung_Foreign', 'Bevölkerung_total']]

# sort data
pivoted_data.sort_values(by='Datum', ascending=True, inplace=True)

# save data
pivoted_data.to_csv(output_path, index=False, sep=";")

# done
print("Aggregated data saved to:", output_path)
