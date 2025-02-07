# Detects irrigation events for both merged folders
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import warnings


def get_irrigation_predictions(df):
    '''
    Rules:
    1. 'SOIL_VWC_10_DELTA' > 0.7
        -> Steigt VWC 10 innerhalb einer Stunde um 70% 
        - von SOIL_VWC_10 wird in den letzten 6h der mean genommen um Tageszeitbedingte Schwankungen zu minimieren
    2. 'RAIN_DELTA_ACC' < 6
        - Regenfall der letzten 24h ist unter 6
        - Regen der letzten 24h wird pro Zeile aufsummiert
    3. Datum ist April - Ende Oktober
    4. Wenn >1 Prognosen im Zeitraum 72h vor und nach einer Prognose wird die erste genommen und die restlichen verworfen
    
    required columns (col name can be different): 'timestamp', 'datetime', 'RAIN_DELTA', 'SOIL_VWC_10', 'DOC_IRRIGATION'
    '''
    # Create a new DataFrame with daily values
    df['datetime'] = pd.to_datetime(df['datetime'])
    # filter columns
    new_column_names = ['timestamp', 'datetime', 'RAIN_DELTA', 'SOIL_VWC_10', 'DOC_IRRIGATION']

    # rename columns by refering column list from the function
    df.columns = new_column_names


    new_df = pd.DataFrame(columns=['timestamp', 'datetime', 'RAIN_DELTA', 'RAIN_DELTA_ACC', 'SOIL_VWC_10', 'DOC_IRRIGATION'])
    for index, row in df.iterrows():
        # get the the sum of the rain delta over the last 24 hours
        row['RAIN_DELTA_ACC'] = df.loc[(df['datetime'] >= row['datetime'] - pd.Timedelta(hours=24)) & (df['datetime'] <= row['datetime'])]['RAIN_DELTA'].sum()

        # get the mean of the soil vwc over the last 6 hours
        row['SOIL_VWC_10'] = df.loc[(df['datetime'] >= row['datetime'] - pd.Timedelta(hours=6)) & (df['datetime'] <= row['datetime'])]['SOIL_VWC_10'].mean()

        # aggregate the last 12 hours
        new_row = {
            'timestamp': row['timestamp'],
            'datetime': row['datetime'],
            'RAIN_DELTA': row['RAIN_DELTA'],
            'RAIN_DELTA_ACC': row['RAIN_DELTA_ACC'],
            'SOIL_VWC_10': row['SOIL_VWC_10'],
            'DOC_IRRIGATION': row['DOC_IRRIGATION']
        }
        # dont use append in next line
        # join the new row to the new_df
        new_df.loc[len(new_df)] = new_row

    
    # add column VWC Delta
    new_df['SOIL_VWC_10_DELTA'] = new_df['SOIL_VWC_10'].diff()
    # catch warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_df['SOIL_VWC_10_DELTA'].iloc[0] = 0
    # add a column VWC Percentual Delta
    new_df['SOIL_VWC_10_DELTA_PERCENTUAL'] = new_df['SOIL_VWC_10_DELTA'] / new_df['SOIL_VWC_10']

    new_df['DOC_IRRIGATION_PREDICTION'] = np.where(
        (new_df['SOIL_VWC_10_DELTA'] > 0.7) &
        (new_df['RAIN_DELTA_ACC'] < 6) &
        ((new_df['datetime'].dt.month >= 4) & (new_df['datetime'].dt.month <= 10)),
        1, 0
    )
    
    # Post-processing step to filter irrigation predictions
    # For loop über folgende Events: es eine dokumentierte Bewässerung oder eine prediction
    # für beide Fälle: schau die 72h davor und danach an
    # wenn es in den 72h davor oder danach mehrere Events gibt (Prediction oder Dokumentierte), dann nimm jeweils das erste und setze die anderen auf 0 

    prediction_indices = new_df[new_df['DOC_IRRIGATION_PREDICTION'] == 1].index
    documented_indices = new_df[new_df['DOC_IRRIGATION'] == 1].index
    # add them to one list
    all_irrigation_indices = np.concatenate((prediction_indices, documented_indices))
    for i in range(len(all_irrigation_indices)):
        current_index = all_irrigation_indices[i]
        current_time = new_df.loc[current_index, 'datetime']
        if new_df.loc[current_index, 'DOC_IRRIGATION_PREDICTION'] == 1 or new_df.loc[current_index, 'DOC_IRRIGATION'] == 1:
            window_start = current_time - pd.Timedelta(hours=72)
            window_end = current_time + pd.Timedelta(hours=72)
            overlapping_predictions = new_df[(new_df['datetime'] > window_start) & 
                                             (new_df['datetime'] < window_end) & 
                                             ((new_df['DOC_IRRIGATION_PREDICTION'] == 1) | (new_df['DOC_IRRIGATION'] == 1))].index
            if len(overlapping_predictions) > 1:
                # Set all overlapping predictions except the first one to 0
                new_df.loc[overlapping_predictions[1:], 'DOC_IRRIGATION_PREDICTION'] = 0
    return new_df['DOC_IRRIGATION_PREDICTION']



if __name__ == "__main__":    
    # get current dir
    cur_dir = os.path.dirname(__file__)
    merged_path = os.path.join(cur_dir, "..", "..", "..", "data", "merged")
    merged_climavis_path = os.path.join(merged_path, "climavi_tree_climavi_weather", "final")
    merged_openmeteo_path = os.path.join(merged_path, "climavi_tree_openmeteo_weather", "final")

    # process both folders
    folders = [merged_openmeteo_path, merged_climavis_path]
    for folder_path in folders:
        # go through all files in the merged folder that end with _merged.csv, load them as df and store them in a dict
        dfs = {}
        for file in os.listdir(folder_path):
            if file.endswith("_merged.csv"):
                key = file.split('_merged.csv')[0]
                dfs[key] = pd.read_csv(os.path.join(folder_path, file))
                # convert timestamp to datetime
                if "timestamp_tree" in dfs[key]:
                    dfs[key]['timestamp'] = dfs[key]['timestamp_tree']
                # convert datetime to datetime
                dfs[key]['datetime'] = pd.to_datetime(dfs[key]['datetime'])
                # sort by datetime
                dfs[key] = dfs[key].sort_values(by='datetime')
                # reset index
                dfs[key] = dfs[key].reset_index(drop=True)
        # print keys
        print(dfs.keys())

        # get irrigation predictions for all dataframes
        for key in dfs:
            if folder_path == merged_climavis_path:
                climavi_subset_cols = ['timestamp', 'datetime', 'EXT|ENV__ATMO__RAIN__DELTA', '-10|ENV__SOIL__VWC', 'DOC|ENV__SOIL__IRRIGATION']
                dfs[key]['DOC_IRRIGATION_PREDICTION'] = get_irrigation_predictions(dfs[key][climavi_subset_cols])
            elif folder_path == merged_openmeteo_path:
                openmeteo_subset_cols = ['timestamp', 'datetime', 'precipitation', '-10|ENV__SOIL__VWC', 'DOC|ENV__SOIL__IRRIGATION']
                dfs[key]['DOC_IRRIGATION_PREDICTION'] = get_irrigation_predictions(dfs[key][openmeteo_subset_cols])
            # save the new dataframe
            dfs[key].to_csv(os.path.join(folder_path, key + '_merged_processed.csv'), index=False)