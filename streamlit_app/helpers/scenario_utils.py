import pandas as pd
import numpy as np

def apply_scenario_modifications(
    test_df: pd.DataFrame,
    scenario_params: dict,
    population_scenarios: dict,
    weather_cols: list = ["T_C", "RainDur_min", "StrGlo_W/m2"],
    population_cols: list = ["Geburte", "Todesfälle", "Zuzüge", "Wegzüge"]
) -> pd.DataFrame:

    # copy data
    scenario_df = test_df.copy()

    # SCENARIO: Weather
    temp_offset = scenario_params.get("temp_offset", 0.0)
    rain_factor = scenario_params.get("rain_factor", 1.0)
    rad_factor  = scenario_params.get("rad_factor", 1.0)
    temp_noise  = scenario_params.get("temp_noise", 0.0)
    rain_noise  = scenario_params.get("rain_noise", 0.0)
    rad_noise   = scenario_params.get("rad_noise", 0.0)

    if "T_C" in weather_cols and "T_C" in scenario_df.columns:
        scenario_df["T_C"] = scenario_df["T_C"].apply(
            lambda x: x + temp_offset + np.random.normal(0, temp_noise)
        )
    if "RainDur_min" in weather_cols and "RainDur_min" in scenario_df.columns:
        scenario_df["RainDur_min"] = scenario_df["RainDur_min"].apply(
            lambda x: max(0, x * rain_factor + np.random.normal(0, rain_noise))
        )
    if "StrGlo_W/m2" in weather_cols and "StrGlo_W/m2" in scenario_df.columns:
        scenario_df["StrGlo_W/m2"] = scenario_df["StrGlo_W/m2"].apply(
            lambda x: x * rad_factor + np.random.normal(0, rad_noise)
        )

    # SCENARIO: Rolling Mean & Lags
    rolling_mean_factor = scenario_params.get("rolling_mean_factor", 1.0)
    lag_offset          = scenario_params.get("lag_offset", 0.0)

    for col in ["rolling_mean_3", "rolling_mean_7"]:
        if col in scenario_df.columns:
            scenario_df[col] = scenario_df[col] * rolling_mean_factor


    for i in range(1, 8):
        lag_col = f"lag_{i}"
        if lag_col in scenario_df.columns:
            scenario_df[lag_col] = scenario_df[lag_col] + lag_offset

    # SCENARIO: Population
    wegzuege_offset    = population_scenarios.get("wegzuege_offset", 0.0)
    zuzuege_offset     = population_scenarios.get("zuzuege_offset", 0.0)
    geburten_offset    = population_scenarios.get("geburten_offset", 0.0)
    todesfaelle_offset = population_scenarios.get("todesfaelle_offset", 0.0)

    wegzuege_noise     = population_scenarios.get("wegzuege_noise", 0.0)
    zuzuege_noise      = population_scenarios.get("zuzuege_noise", 0.0)
    geburten_noise     = population_scenarios.get("geburten_noise", 0.0)
    todesfaelle_noise  = population_scenarios.get("todesfaelle_noise", 0.0)

    if "Geburte" in population_cols and "Geburte" in scenario_df.columns:
        scenario_df["Geburte"] = scenario_df["Geburte"].apply(
            lambda x: x + geburten_offset + np.random.normal(0, geburten_noise)
        )
    if "Todesfälle" in population_cols and "Todesfälle" in scenario_df.columns:
        scenario_df["Todesfälle"] = scenario_df["Todesfälle"].apply(
            lambda x: x + todesfaelle_offset + np.random.normal(0, todesfaelle_noise)
        )
    if "Zuzüge" in population_cols and "Zuzüge" in scenario_df.columns:
        scenario_df["Zuzüge"] = scenario_df["Zuzüge"].apply(
            lambda x: x + zuzuege_offset + np.random.normal(0, zuzuege_noise)
        )
    if "Wegzüge" in population_cols and "Wegzüge" in scenario_df.columns:
        scenario_df["Wegzüge"] = scenario_df["Wegzüge"].apply(
            lambda x: x + wegzuege_offset + np.random.normal(0, wegzuege_noise)
        )

    return scenario_df
