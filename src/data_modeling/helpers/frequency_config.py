import os

base_path = os.getcwd()
data_processed_dir = os.path.join(base_path, "data", "processed", "water")

AGGREGATION_CONFIG = {
    "daily": {
        "input_file": os.path.join(data_processed_dir, "water_consumption_2015_2023_daily_normalized.csv"),
        "frequency": "D",
        "with_lag_features": True,
        "lag_periods": 7, # 1 week
        "train_split": lambda df: df.index.year <= 2022,
        "test_split":  lambda df: df.index.year == 2023
    },
    "weekly": {
        "input_file": os.path.join(data_processed_dir, "water_consumption_2015_2023_weekly_normalized.csv"),
        "frequency": "W",
        "with_lag_features": True,
        "lag_periods": 4,  # 1 month
        "train_split": lambda df: df.index.year <= 2022,
        "test_split":  lambda df: df.index.year == 2023
    },
    "monthly": {
        "input_file": os.path.join(data_processed_dir, "water_consumption_2015_2023_monthly_normalized_join_population_monthly.csv"),
        "frequency": "M",
        "with_lag_features": True,
        "lag_periods": 12,  # 1 year
        "train_split": lambda df: df.index.year <= 2020,
        "test_split":  lambda df: df.index.year == 2023
    },
    "yearly": {
        "input_file": os.path.join(data_processed_dir, "water_consumption_2015_2023_yearly_normalized.csv"),
        "frequency": "YE",
        "with_lag_features": False,
        "train_split": lambda df: df.index.year <= 2021,
        "test_split":  lambda df: df.index.year >= 2022
    }
}
