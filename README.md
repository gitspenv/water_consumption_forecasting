
# Water Consumption Forecasting in Zurich

This project aims to help Zurich’s utilities and planners better understand, predict, and optimize water usage. Below is a quick tour of what we did, how everything is structured, and how you can explore our results.

---

## Project Overview

- **Goal**: Predict water consumption in Zurich at various time scales (daily, weekly, monthly, yearly) using models like **Linear Regression**, **Random Forest**, **LightGBM**, and **SARIMA**.
- **Data Sources**:
  1. **Water consumption** (2015–2023)
  2. **Meteorological data** (temperature, rain, radiation, barometric pressure, humidity)
  3. **Population data** (births, deaths, migrations, vacation)
- **Approach**:
  1. **Data Preprocessing** – Cleaning, missing-value handling, and feature engineering (e.g., lags, rolling means).  
  2. **Modeling** – Training multiple models to predict future consumption.  
  3. **Scenario Analysis** – Scenario Analysis – Simulating changes in temperature, rainfall, population, and time-based patterns (rolling means, lags) to see how demand shifts.
  4. **Dashboards** – Interactive visualizations using **Streamlit**.
 
## Folder Structure

```text
├─ data
│  ├─ raw          <- Original CSV files (water, meteo, population)
│  ├─ processed    <- Cleaned & aggregated data files
│  └─ output       <- Model outputs (predictions, metrics, SHAP values, etc.)
│     └─ model
│        ├─ lightgbm
│        ├─ linear_regression
│        ├─ random_forest
│        └─ sarima
├─ models          <- Pickled models
├─ src
│  ├─ data_exploration    <- Scripts for loading & exploring data
│  ├─ data_modeling       <- Model training code & helpers
│  └─ data_preprocessing  <- Scripts for merging, cleaning, & structuring data
├─ streamlit_app
│  ├─ config              <- Config files for the dashboard
│  ├─ app.py              <- Main Streamlit dashboard
│  └─ helpers             <- Utility scripts for the dashboard
├─ requirements.txt       <- Python dependencies
```

## How To Run

**Python Version**: **3.12.1** or higher

1. **Clone the repository**

```bash
 git clone git clone https://github.com/gitspenv/water_consumption_forecasting.git
 cd water_consumption_forecasting
```
2. **Create a Virtual Environment (Recommended)**
```bash
python3 -m venv venv
```
3. **Check Python Version**
```bash
python3 --version
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Run the Streamlit App**
```bash
streamlit run streamlit_app/app.py
```

