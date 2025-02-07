# Machine Learning for Trees, by Trees: Soil Moisture Forecasting with Random Forests

## Project Description

This repository showcases a practical application of **Random Forests** for soil moisture forecasting in **urban trees**. By anticipating soil moisture fluctuations, city managers and arborists can schedule irrigation more efficiently—an increasingly vital task given the **warming climate** and **limited staff availability**. The project includes end-to-end data extraction from sensor APIs, data preprocessing, model training, and result visualization.

Three core objectives guided this work:
1. **Predict short- and medium-term soil moisture levels** (1-, 3-, 5-, 7-day horizons).
2. **Evaluate different target types** (daily minimum, maximum, or combined “min-max”) to determine which best prevents critically low moisture conditions.
3. **Evaluate Impact of Irrigation Events** on performance

Below is an example GIF demonstrating how closely (or not) the model’s predictions match actual soil moisture over time:

<div align="center">
    <img src="/media/readme/sm_forecast_minmax_3days.gif" width="800" alt="Description of GIF">
</div>
*(GIF showing 3-day predictions of daily minimum and maximum VWC vs. actual values.)*


---

## Table of Contents

- [Project Description](#project-description)
- [Table of Contents](#table-of-contents)
- [Abstract](#abstract)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Acquisition](#data-acquisition)
- [Data Processing (ETL)](#data-processing-etl)
- [Experiments](#experiments)
- [Usage](#usage)
- [Future Work](#future-work)
- [Additional Resources](#additional-resources)
- [Acknowledgements](#acknowledgements)

---
## Abstract

> Urban trees help mitigate rising temperatures and prolonged droughts in cities, yet their survival hinges on maintaining adequate soil moisture. This study evaluates the feasibility of **Random Forest models** for soil moisture forecasting in an urban tree setting in Erlangen, Germany, using a 1.5-year dataset (488 samples). The models target **1-, 3-, 5-, and 7-day** prediction horizons, incorporating both **historical weather data** and **irrigation events** identified via empirical rules. Various forecast targets—daily minimum, maximum, and a min-max combination—were examined to identify the most effective approach.

> **Results** show that **short-term (1- and 3-day) forecasts** reliably capture irrigation-induced spikes and outperform the **5- and 7-day** models, which struggle with accuracy. **Daily minimum** values consistently yield lower errors than daily maximum or “close” values, making them more suitable for preventing critically low moisture levels. However, **spurious irrigation-like spikes** occur at low volumetric water content (<12%) in **all horizon models**, suggesting an overly deterministic link between low moisture and subsequent irrigation events. Additionally, **seasonal biases** are evident; models optimized primarily on summer data falter when predicting early autumn conditions, underscoring the limitations of a **small, single-tree dataset**.

> To address these challenges, **future work** should include collecting larger, multi-year datasets spanning diverse seasons and multiple trees, incorporating **realistic weather forecasts**, and refining strategies to handle rare, low-VWC scenarios. Such efforts would enhance model robustness and support the development of reliable, data-driven irrigation management systems for urban forestry.

---

## Key Features
- **Random Forest Models** that predict soil moisture at 1-, 3-, 5-, and 7-day horizons.  
- **Flexible Target Types** (daily min, max, or both) to tailor predictions to practical irrigation needs.  
- **Irrigation Detection**: Empirical rules to flag irrigation events in the data.  
- **Data Enrichment**: Merge of sensor-based and Open-Meteo weather data.  
- **Visual Demonstrations**: GIFs and plots comparing predicted vs. actual moisture levels.

---



## Prerequisites

- Python 3.8+
- pip
- Git
- Required Python libraries (listed in `requirements.txt`)

## Installation

1. **Clone the repository:**

2. **Create and activate a virtual environment**

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up Credentials** (if you plan to fetch data via APIs):
   - In `src/etl/extract/`, create `credentials.json` with:
     ```json
     {
       "username": "your_username",
       "password": "your_password"
     }
     ```
   - Modify this file or the scripts as needed for your own authentication flow.

5. **Adjust Sensor Settings (Optional):**

    - Modify `data/climavi_sensor_table.csv` to control which tree data and weather data are fetched and how they are merged.


---


## Project Structure

```
soil-moisture-forecasting-project/
├── data/
│   ├── external/
│   ├── merged/
│   │   ├── climavi_tree_climavi_weather/
│   │   └── climavi_tree_openmeteo_weather/
│   ├── raw/
│   │   ├── climavi/
│   │   │   ├── forecasts/
│   │   │   └── tree_sensors/
│   │   └── openmeteo/
│   │       ├── daily.csv
│   │       └── hourly.csv
│   ├── climavi_sensor_table.csv
├── docs/
│   ├── sensor_selection.md
│   └── sensor_variables.md
├── media/
│   ├── climavi_weather/
│   ├── merged_climavi/
│   ├── merged_climavi_openmeteo/
│   ├── openmeteo_weather/
│   └── trees/
├── src/
│   ├── experiments/
│   │   ├── get_engineered_features.py
│   │   └── run_experiments.py
│   ├── etl/
│   │   ├── extract/
│   │   │   ├── auth.py
│   │   │   ├── credentials.json
│   │   │   ├── extract_climavi_forecast_data.py
│   │   │   ├── extract_climavi_sensor_info.py
│   │   │   ├── extract_climavi_tree_data.py
│   │   │   ├── extract_climavi_weather_data.py
│   │   │   └── extract_openmeteo_weather_data.py
│   │   ├── transform/
│   │   │   ├── create_data_reports.py
│   │   │   ├── create_plots.py
│   │   │   ├── detect_irrigations.py
│   │   │   ├── merge_climavi_sensors_hourly_data.py
│   │   │   ├── merge_tree_openmeteo_weather_hourly_data.py
│   │   │   └── notebook_detect_irrigations.ipynb
│   │   └── main.py
│   ├── analyze_results/
│   │   └── (files to analyze results and create LaTeX tables)
│   └── visualizations/
│       ├── visualize_locations.ipynb
├── results/
│   └── runs/
├── requirements.txt
├── README.md
└── Paper.pdf
```

### Key Directories and Files

- **data/**
    - **external/**: Contains external data like api documentation and documented irrigation events.
    - **merged/**: Merged data where tree sensor data and weather data are combined.
        - **climavi_tree_climavi_weather/**: Both tree and weather data from Climavi sensors.
        - **climavi_tree_openmeteo_weather/**: Tree data from Climavi and weather data from Open-Meteo API.
    - **raw/**: Raw data from data sources.
        - **climavi/**: Contains forecasts and tree sensor data.
            - **forecasts/**: Forecast data from Climavi API (not utilized in this project).
            - **tree_sensors/**: Saved CSV files for each tree sensor.
        - **openmeteo/**: Contains daily and hourly weather data (daily data used after aggregation).
    - **climavi_sensor_table.csv**: Controls which sensors are used and how data is merged.

- **docs/**
    - **sensor_selection.md**: Notes on why certain sensors were selected or excluded.
    - **sensor_variables.md**: Documentation of available variables and their relevance.

- **media/**: Contains plots and images for various data visualizations.
    - Subfolders include:
        - **climavi_weather/**
        - **merged_climavi/**
        - **merged_climavi_openmeteo/**
        - **openmeteo_weather/**
        - **trees/**

- **src/**
    - **experiments/**: Scripts to prepare data and run experiments.
        - **get_engineered_features.py**: Generates engineered features for Random Forest modeling.
        - **run_experiments.py**: Runs experiments with varying parameters.
    - **etl/**: Extract, Transform, Load pipeline scripts.
        - **extract/**: Scripts for data extraction from APIs.
            - **auth.py**: Handles API authentication.
            - **credentials.json**: User-provided credentials for API access.
            - **extract_climavi_forecast_data.py**: Extracts forecast data (not utilized).
            - **extract_climavi_sensor_info.py**: Retrieves information on all sensors.
            - **extract_climavi_tree_data.py**: Downloads and updates tree sensor data.
            - **extract_climavi_weather_data.py**: Downloads and updates Climavi weather data.
            - **extract_openmeteo_weather_data.py**: Fetches weather data from Open-Meteo API.
        - **transform/**: Data transformation scripts.
            - **create_data_reports.py**: Generates reports on processed data.
            - **create_plots.py**: Generates and saves plots in the media folder.
            - **detect_irrigations.py**: Algorithm to detect irrigation events.
            - **merge_climavi_sensors_hourly_data.py**: Merges Climavi sensor data.
            - **merge_tree_openmeteo_weather_hourly_data.py**: Merges tree data with Open-Meteo weather data.
            - **notebook_detect_irrigations.ipynb**: Experimental notebook for irrigation detection (non-productive).
        - **main.py**: Runs the entire ETL pipeline.
    - **analyze_results/**: Scripts and files to analyze experiment results, create LaTeX tables and gif visualizations.
    - **visualizations/**: Notebooks for data visualization and exploration.
        - **visualize_locations.ipynb**: Displays sensor locations and types on a map.

- **results/runs/**: Contains results from different experiment runs, including plots, dataset splits, feature importances, predictions, and more.
    - Each run folder corresponds to an experiment with specific parameters (e.g., `run_1days_d_FalseIrrigations_close_2024-11-06_22-08`).

- **requirements.txt**: Lists all Python dependencies.

- **Paper.pdf**: Full research documentation.


---

## Data Acquisition

Data is collected from soil moisture sensors installed in urban trees in Erlangen, Germany, provided by Agvolution GmbH's Climavi platform. Weather data is obtained from both Climavi sensors and the Open-Meteo API.

- **Tree Sensor Data:**
    - Extracted using `src/etl/extract/extract_climavi_tree_data.py`
- **Weather Data:**
    - **Climavi Weather Sensors:**
        - Extracted using `src/etl/extract/extract_climavi_weather_data.py`
    - **Open-Meteo API:**
        - Extracted using `src/etl/extract/extract_openmeteo_weather_data.py`

- **Sensor Configuration:**
    - Adjust the `data/climavi_sensor_table.csv` file to control which tree and weather data are fetched and how they are merged.

## Data Processing (ETL)

The ETL pipeline extracts raw data, transforms it, and loads processed data for analysis and modeling.

- **Run the ETL Pipeline:**

    ```bash
    python src/etl/main.py
    ```

- **Transformations Include:**
    - **Merging Data:**
        - `merge_climavi_sensors_hourly_data.py`: Merges Climavi tree and weather data.
        - `merge_tree_openmeteo_weather_hourly_data.py`: Merges tree data with Open-Meteo weather data.
    - **Detecting Irrigation Events:**
        - `detect_irrigations.py`: Implements an irrigation detection algorithm.
    - **Data Reporting and Visualization:**
        - `create_data_reports.py`: Generates reports on processed data.
        - `create_plots.py`: Generates and saves plots in the media folder.

## Experiments

Experiments involve training Random Forest models with different configurations.

- **Prepare Engineered Features:**

    ```bash
    python src/experiments/get_engineered_features.py
    ```

- **Run Experiments:**

    ```bash
    python src/experiments/run_experiments.py
    ```

    - Adjust parameters within `run_experiments.py` to experiment with different settings:
        - Prediction horizons (e.g., 1-day, 3-day)
        - Inclusion or exclusion of irrigation data
        - Feature granularity (daily or hourly)
        - Target definitions (e.g., minimum, maximum, close values)


## Usage
1. **Adjust Sensor Settings (Optional):**

    - Edit `data/climavi_sensor_table.csv` to select which sensors to use and how to merge data.

2. **Run the ETL Pipeline to Fetch the Most Recent Data:**

    ```bash
    python src/etl/main.py
    ```

3. **Run Experiments:**

    - Navigate to `src/experiments/run_experiments.py` to configure and run experiments:

        ```bash
        python src/experiments/run_experiments.py
        ```

4. **Analyze Results:**

    - Use scripts in `src/analyze_results/` or check the `results/runs/` directories for outputs.


---

## Future Work

**Short-term forecasts** (1- and 3-day) performed well, but they struggled with seasonal shifts. Also, **spurious irrigation spikes** at low VWC highlight the need for more balanced data that covers non-irrigation scenarios thoroughly. Future research avenues include:
- **Collecting multi-year, multi-site data** to reduce seasonal biases and validate model generality.  
- **Incorporating actual weather forecasts** rather than historical “perfect” data to reflect real-world uncertainty.  
- **Addressing imbalanced conditions** with techniques like oversampling or synthetic data for low-VWC scenarios.  
- **Evaluating other machine learning approaches** (e.g., LSTM or transformer-based models) that may capture longer temporal dependencies.  
- **Real-time integration** into irrigation management systems, providing automated alerts when critical moisture thresholds approach.

---

## Additional Resources
For more context on urban forestry, tree health, and ongoing projects in Erlangen, consider exploring the following:

- [Bäume für die Stadt: Natur unter Druck in Erlangen (Video)](https://www.youtube.com/watch?v=POaVaifuQpU)
- [Intelligentes Baumgieß-System für Kommunen (Video)](https://www.youtube.com/watch?v=SqopzjHBSyU)
- [Passion4Tech Blog - Interview with Andreas Bechmann](https://www.passion4tech.de/blog/preventio-andreas-bechmann-zehnk%C3%A4mpfer-mit-starkem-gr%C3%BCnder-gen-1)
- [Climavi Platform](https://www.climavi.eu/)
- [Monitoring of Urban Water Demand for City Trees (FAU)](https://www.mad.tf.fau.de/research/projects/monitoring-of-urban-water-demand-for-city-trees/)

---

## Acknowledgements

- **City of Erlangen** for facilitating sensor data.  
- **Agvolution GmbH** for the Climavi platform and technical support.  
- **Thomas Maier** for guidance and collaboration.

---
