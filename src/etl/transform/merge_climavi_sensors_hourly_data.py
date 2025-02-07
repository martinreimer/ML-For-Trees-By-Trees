# Merges climavi tree sensor data with climavi weather sensor data based on specifications in climavi_sensor_table.csv

import pandas as pd
# surpress anyoying warnings
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import warnings
import datetime

'''
Configuration
'''
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CUR_DIR, "..", "..", "..", "data")
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
MERGED_DATA_PATH = os.path.join(DATA_PATH, "merged", "climavi_tree_climavi_weather")
MERGED_INTERIM_DATA_PATH = os.path.join(MERGED_DATA_PATH, "interim")
MERGED_FINAL_DATA_PATH = os.path.join(MERGED_DATA_PATH, "final")
MERGED_INTERPOL_DATA_PATH = os.path.join(MERGED_INTERIM_DATA_PATH, "interpolated")
MERGED_SEP_DATA_PATH = os.path.join(MERGED_INTERIM_DATA_PATH, "seperate")
TREE_DATA_PATH = os.path.join(RAW_DATA_PATH, "climavi", "tree_sensors")
WEATHER_DATA_PATH = os.path.join(RAW_DATA_PATH, "climavi", "weather_sensors")
SENSOR_TABLE_PATH = os.path.join(DATA_PATH, "climavi_sensor_table.csv")

# create directories if not exist
os.makedirs(MERGED_DATA_PATH, exist_ok=True)
os.makedirs(MERGED_INTERIM_DATA_PATH, exist_ok=True)
os.makedirs(MERGED_FINAL_DATA_PATH, exist_ok=True)
os.makedirs(MERGED_INTERPOL_DATA_PATH, exist_ok=True)
os.makedirs(MERGED_SEP_DATA_PATH, exist_ok=True)

'''
Get Data for each sensor
'''
def get_data_for_sensor_type(df_sensors, sensor_type_path):
    '''
    Get data for each sensor type (tree or weather)
    - df_sensors: df with sensor data for weather or tree
    - sensor_type_path: path to weather or tree data
    '''
    data = {}
    for _, row in df_sensors.iterrows():
        sensor_name = row['label']
        file_path = os.path.join(sensor_type_path, f"{sensor_name}.csv")
        df = pd.read_csv(file_path, sep=',', encoding='latin1')
        data[sensor_name] = df
    return data


'''
Preprocessing Tree / Weather Data

- assumed cols before processing:
    Tree:
    ['timestamp', 'datetime', '-10|ENV__SOIL__VWC', '-30|ENV__SOIL__VWC',
        '-45|ENV__SOIL__VWC', '-10|ENV__SOIL__T', '-30|ENV__SOIL__T',
        '-45|ENV__SOIL__T', 'TOP|ENV__ATMO__T', 'DOC|ENV__SOIL__IRRIGATION'
    ]
    Weather:
    ['timestamp', 'datetime', 'TOP|ENV__ATMO__T', 'TOP|ENV__ATMO__RH',        
        'TOP|ENV__ATMO__IRRADIATION', 'EXT|ENV__ATMO__RAIN__DELTA',
        'MTB|ENV__ATMO__T', 'MTB|ENV__ATMO__RAIN__DELTA', 'MTB|ENV__ATMO__RH',
        'DOC|ENV__SOIL__IRRIGATION'
    ]
- cols after drop:
    Tree:
    ['timestamp', 'datetime', '-10|ENV__SOIL__VWC', '-30|ENV__SOIL__VWC',
        '-45|ENV__SOIL__VWC', '-10|ENV__SOIL__T', '-30|ENV__SOIL__T',
        '-45|ENV__SOIL__T', 'TOP|ENV__ATMO__T', 'DOC|ENV__SOIL__IRRIGATION'
    ]
    Weather:
    ['timestamp', 'datetime', 'TOP|ENV__ATMO__T', 'TOP|ENV__ATMO__RH',        
        'TOP|ENV__ATMO__IRRADIATION', 'EXT|ENV__ATMO__RAIN__DELTA',
        'DOC|ENV__SOIL__IRRIGATION'
    ]
'''

def drop_cols(df, cols_to_drop):
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df



def drop_duplicates(df, time_min_threshold=30):
    '''Drop duplicate rows based on a time threshold: if the time difference between two consecutive rows is less than the threshold, they are considered duplicates
        - create group of rows where the time difference is less than threshold to the next row
        - duplicates are identified based on all columns except timestamp, datetime, time_diff, and group
        - first occurrence of each duplicate is kept
    '''
    # Ensure DataFrame is sorted by datetime
    df = df.sort_values(by='datetime').reset_index(drop=True)
    df_len_before = len(df)
    # Calculate the time difference between consecutive rows
    df['time_diff'] = df['datetime'].diff().dt.total_seconds().div(60).fillna(0)
    # Assign a group number based on the time threshold
    df['group'] = (df['time_diff'] > time_min_threshold).cumsum()
    # Drop duplicates within each group, keeping the first occurrence
    relevant_cols = [col for col in df.columns if col not in ['timestamp', 'datetime', 'time_diff', 'group']]
    # Suppress DeprecationWarning from pandas
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        df = df.groupby('group', group_keys=False).apply(
            lambda group: group.drop_duplicates(subset=relevant_cols, keep='first')
        ).reset_index(drop=True)
        # Drop temporary columns
        df = df.drop(columns=['time_diff', 'group'])
    print(f"- {df_len_before - len(df)} removed duplicate rows")
    return df


def drop_almost_duplicates(df, time_min_threshold=5):
    '''Drop almost duplicate rows based on a time threshold: if the time difference between two consecutive rows is less than the threshold, they are considered almost duplicates
        Motivation: Sometimes if tree sensor observes rain, it will send two rows at almost the same time, where one row will contain a non NaN value for the column rain_delta and sometimes tree sensor has two rows at almost the same time, where the first one will have the correct values for -10,-30,-45 soil temperature, while the second will have the value for -10 for all the other soil temperatures
            -> not 100% sure if this handling is really correct 
    '''
    counter = 0
    keep_indices = []
    i = 0
    while i < len(df):
        current_row = df.iloc[i]
        keep = True
        skip_next = False
        # Check the next row
        if i + 1 < len(df):
            next_row = df.iloc[i + 1]
            time_diff_next = (next_row['datetime'] - current_row['datetime']).total_seconds() / 60
            # if the time difference is less than 5 minutes
            if time_diff_next < time_min_threshold:
                # check which which row has less nan values
                current_row_nan = current_row.isna().sum()
                next_row_nan = next_row.isna().sum()
                counter += 1
                # if current row has more nan values, keep next row
                if current_row_nan > next_row_nan:
                    # keep next row
                    keep = False
                # if next row has more nan values, keep current row
                elif current_row_nan < next_row_nan:
                    # keep current row
                    skip_next = True 
                else:
                    # pick the current row just bcs in this case the current row is correct
                    skip_next = True
        if keep:
            keep_indices.append(i)
        i += 1
        if skip_next and i < len(df):
            i += 1
    print(f"- {counter} almost duplicates removed")
    df = df.iloc[keep_indices].reset_index(drop=True)
    return df


def merge_irrigation_values_w_prev_row(df):
    '''Merge the 'DOC|ENV__SOIL__IRRIGATION' column with the previous row if the value is greater than 0
       Motivation: Rows with irragation values which are not 0, dont have values for other columns, so we merge them with the previous row if its at most 24 hours apart
    '''
    # check if column is in df
    if 'DOC|ENV__SOIL__IRRIGATION' not in df.columns:
        return df
    last_row = None
    for i, row in df.iterrows():
        # check if the current row has a 0 value for DOC|ENV__SOIL__IRRIGATION
        if row['DOC|ENV__SOIL__IRRIGATION'] > 0 and last_row is not None:
            time_diff = (last_row['datetime'] - row['datetime']).total_seconds() / 60
            if abs(time_diff) < 60*24:
                # use value  
                df.at[i - 1, 'DOC|ENV__SOIL__IRRIGATION'] = row['DOC|ENV__SOIL__IRRIGATION']
                df.at[i, 'DOC|ENV__SOIL__IRRIGATION'] = 0
        last_row = df.iloc[i]
    return df


def check_zeros_and_nans(row):
    '''Check if a row has only 0 and NaN values in the subset of columns'''
    return row.isin([0, np.nan]).all()

def drop_empty_rows(df, cols_to_ignore=['timestamp', 'datetime', 'DOC|ENV__SOIL__IRRIGATION']):
    '''Drop rows that have only NaN and 0 values in the subset of columns
    - cols_to_ignore: columns to ignore when checking for empty rows
    '''
    before_len = len(df)
    # check if row is only NaN and or 0, drop if so, keep if not
    # Apply the function to each row in the subset
    relevant_cols = [col for col in df.columns if col not in cols_to_ignore]
    rows_to_drop = df[relevant_cols].apply(check_zeros_and_nans, axis=1)
    # Drop the rows that meet the condition
    df = df[~rows_to_drop]
    after_len = len(df)
    print(f"- {str(after_len-before_len)} empty rows removed")
    return df

def drop_almost_empty_rows(df, cols_to_ignore=['timestamp', 'datetime', 'DOC|ENV__SOIL__IRRIGATION']):
    '''Drop rows that have at least one NaN value in the subset of columns
    - cols_to_ignore: columns to ignore when checking for empty rows
    '''
    before_len = len(df)
    # check if row has at least one Nan value, drop if so, keep if not
    # Apply the function to each row in the subset
    relevant_cols = [col for col in df.columns if col not in cols_to_ignore]
    rows_to_drop = df[relevant_cols].apply(lambda row: row.isna().any(), axis=1)
    # Drop the rows that meet the condition
    df = df[~rows_to_drop]
    after_len = len(df)
    print(f"- {str(after_len-before_len)} almost empty rows removed")
    return df


def fill_na_w_zero_df(df, cols_to_fill):
    '''Fill NaN values in the specified columns with 0'''
    for col in cols_to_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df


def add_missing_rows(df, time_min_threshold=90):
    '''Add missing rows to the DataFrame if the time difference to the next row is at least threshold (70 minutes)
       Motivation: We dont have rows for every hour, so we create new rows for every hour such that we can interpolate in the next step
    '''
    # Create a list to hold new rows
    new_rows = []
    # Create a column to keep track of added rows
    df['added_row'] = False
    # sort by timestmap
    df = df.sort_values(by='datetime').reset_index(drop=True)
    for i, row in df.iterrows():
        if i + 1 < len(df):
            next_row = df.iloc[i + 1]
            time_diff = (next_row['datetime'] - row['datetime']).total_seconds() / 60
            # while bcs we want to create rows until the time difference is less than the tresold
            while time_min_threshold < time_diff:
                # Create a new empty row with NaN values one hour after the current row
                new_row = pd.Series({col: np.nan for col in df.columns})
                new_row['timestamp'] = int(row['timestamp']) + 60 * 60 * 1000  # Assuming timestamp is in milliseconds
                # surpress warning 
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    new_row['datetime'] = pd.to_datetime(row['datetime'] + datetime.timedelta(minutes=60))
                # Mark the current row as having added a new row
                new_row['added_row'] = True
                # If "DOC|ENV__SOIL__IRRIGATION" in the row, set it to 0
                if 'DOC|ENV__SOIL__IRRIGATION' in new_row:
                    new_row['DOC|ENV__SOIL__IRRIGATION'] = 0
                # Append the new row to the list of new rows
                new_rows.append(new_row)
                # Update the row and time difference for the next iteration
                row = new_row
                time_diff = (next_row['datetime'] - row['datetime']).total_seconds() / 60
    # Create a DataFrame from the new rows
    new_rows_df = pd.DataFrame(new_rows)
    # Concatenate the original DataFrame with the new rows DataFrame
    df = pd.concat([df, new_rows_df], ignore_index=True)
    # Sort by datetime
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    # set timestamp as int
    df['timestamp'] = df['timestamp'].astype('int64')
    # print how many rows were added
    total_added_rows = new_rows_df.shape[0]
    print(f"- {total_added_rows} rows were added")
    return df

def interpolate_columns(df, hours_to_interpolate=4):
    '''Interpolate missing values in the DataFrame based on the time difference to the next valid value
       Motivation: We have missing values in the tree and weather data, so we interpolate them based on the time difference to the next valid value
       - for each col:
        - Copy the DataFrame to compare before and after interpolation.
        - Interpolate values for the next 4 hours/rows (linear interpolation)
        - Update the DataFrame with the interpolated values
        - Track which rows were updated
    '''
    df.set_index('datetime', inplace=True)
    # Create a boolean Series to track which rows have been interpolated.
    interpolated_rows = pd.Series(False, index=df.index) 

    for col in df.columns:
        # Skip timestamp, datetime, and interpolated columns
        if col in ['timestamp', 'datetime']:
            continue

        # Copy the original column for comparison after interpolation
        df_before = df.copy()
        '''
        # Identify gaps where the forward gap is between 70 minutes and 4 hours
        forward_valid = df[col].notna()
        forward_valid_shifted = forward_valid.shift(-1).fillna(False)
        time_diff_forward = df.index.to_series().diff().shift(-1)
        
        # Define the mask with the desired conditions
        min_time_diff = pd.Timedelta(minutes=70)
        max_time_diff = pd.Timedelta(hours=5)
        mask_forward = (time_diff_forward >= min_time_diff) & (time_diff_forward <= max_time_diff)
        
        # Mark where mask conditions and next value validity are true
        mask = mask_forward & forward_valid_shifted
        
        # Perform interpolation only on the masked areas
        df[f"{col}_masked_interp"] = df[col].interpolate(method='linear', limit_direction="forward")
        # Apply the mask to only accept interpolated values where mask is True
        df[col] = np.where(mask, df[f"{col}_masked_interp"], df[col])
        df.drop(columns=[f"{col}_masked_interp"], inplace=True)  # Cleanup intermediate columns
        # Create column to keep track of interpolated columns
        df[f"{col}_interpolated"] = ~((df_before[col].isna() & df[col].isna()) | (df_before[col] == df[col]))
        # Update the interpolated rows tracker
        interpolated_rows |= df[f"{col}_interpolated"]
        '''
        # Perform interpolation only on the masked areas
        df[col] = df[col].interpolate(method='linear', limit_direction="forward", limit=hours_to_interpolate, limit_area='inside')
        # Create column to keep track of interpolated columns
        df[f"{col}_interpolated"] = ~((df_before[col].isna() & df[col].isna()) | (df_before[col] == df[col]))
        # Update the interpolated rows tracker
        interpolated_rows |= df[f"{col}_interpolated"]
    
                
    # Count the total number of interpolated rows
    total_interpolated_rows = interpolated_rows.sum()
    print(f"- {total_interpolated_rows} values were interpolated in total")
    # Save processed data to CSV
    tmp_path = os.path.abspath(os.path.join(MERGED_INTERPOL_DATA_PATH, f"{name}_interpol.csv"))
    df.to_csv(tmp_path, index=True)
    
    # Drop the boolean columns after saving them
    bool_cols = [col for col in df.columns if col.endswith('_interpolated')]
    df.drop(columns=bool_cols, inplace=True)
    # drop added_row column
    df.drop(columns=['added_row'], inplace=True)
    return df


def round_columns(df, skip_cols=['timestamp', 'datetime'], digits=2):
    '''Round all columns in the DataFrame to a specified number of digits
    '''
    for col in df.columns:
        if col not in skip_cols:
            df[col] = df[col].round(digits)
    return df


def merge_tree_w_weather_dfs(tree_data, weather_data):
    '''Merge tree and weather DataFrames based on the timestamp
        - For each tree sensor DataFrame in `tree_data`, this function finds the matching weather DataFrame 
          and merges them using a datetime-based merge with a 5h tolerance. First it was 1h tolerance but there were few examples where there were no weather data for <4 hours and i decided to increase it to 5h such that we have less splits in the data
        - It aligns the data starting from the first valid weather observation.
    - tree_data: dictionary of tree DataFrames
    - weather_data: dictionary of weather DataFrames
    '''
    merged_data = {}
    for tree_sensor_name, tree_df in tree_data.items():
        # figure out the corresponding weather sensor
        weather_sensor_id = df_favorite_tree_sensors[df_favorite_tree_sensors['label'] == tree_sensor_name]['corresponding_weather_station_id'].values[0]
        weather_sensor_name = df_favorite_weather_sensors[df_favorite_weather_sensors['nameId'] == weather_sensor_id]['label'].values[0]
        weather_df = weather_data[weather_sensor_name]

        # set datetime column as index
        tree_df['datetime'] = pd.to_datetime(tree_df['timestamp'], unit='ms')
        weather_df['datetime'] = pd.to_datetime(weather_df['timestamp'], unit='ms')

        tree_df.set_index('datetime', inplace=True)
        weather_df.set_index('datetime', inplace=True)
        
        # get first valid weather row index (has value for irradatation)
        first_valid_index = weather_df[weather_df['TOP|ENV__ATMO__IRRADIATION'].notnull()].index[0]
        # Remove all rows before the first valid index
        tree_df = tree_df.loc[first_valid_index:]

        # merge        
        # specify tolerance for datetime based merge
        tolerance_in_minutes = pd.Timedelta(5, 'h') #4h
        df_left_merge = pd.merge_asof(tree_df, weather_df, left_index=True, right_index=True, 
                                      direction='nearest', tolerance=tolerance_in_minutes, suffixes=('_tree', '_weather'))
        #df_left_merge = df_left_merge.drop(columns=['timestamp_y', 'datetime_y'])
        # rename timestamp_x to timestamp and datetime_x to datetime
        #df_left_merge = df_left_merge.rename(columns={'timestamp_x': 'timestamp', 'datetime_x': 'datetime'})
        df_left_merge = drop_almost_empty_rows(df_left_merge)
        # make index a column
        df_left_merge.reset_index(inplace=True)
        # save to csv
        tmp_path = os.path.abspath(os.path.join(MERGED_FINAL_DATA_PATH, f"{tree_sensor_name}_+_{weather_sensor_name}_merged.csv"))
        df_left_merge.to_csv(tmp_path, index=False)
        print(f"Saved merged data to {tmp_path}")
        merged_data[f"{tree_sensor_name}_+_{weather_sensor_name}"] = df_left_merge
    return merged_data


if __name__ == "__main__":
    '''
    Get Data that we want to merge
    - marked as favorite in climavi_sensor_table.csv 
    '''
    # get sensor_table csv
    sensor_table_path = os.path.join(DATA_PATH, "climavi_sensor_table.csv")
    df_sensor_table = pd.read_csv(sensor_table_path, sep=',', encoding='latin1')
    # only rows where is_favorite is True
    df_favorite_sensors = df_sensor_table[df_sensor_table['is_favorite'] == True]
    df_favorite_tree_sensors = df_favorite_sensors[df_favorite_sensors['is_weather_station'] == False]
    df_favorite_weather_sensors = df_favorite_sensors[df_favorite_sensors['is_weather_station'] == True]
    print(f"To be merged Sensors:")
    print(f"- Tree:\n{df_favorite_tree_sensors['label']}")
    print(f"- Weather:\n{df_favorite_weather_sensors['label']}\n")

    # get data for favorite tree and weather sensors
    tree_data = get_data_for_sensor_type(df_favorite_tree_sensors, TREE_DATA_PATH)
    weather_data = get_data_for_sensor_type(df_favorite_weather_sensors, WEATHER_DATA_PATH)
    

    # Preprocessing
    # loop over tree and weather data
    # loop over all dfs
    sensors_data = [tree_data, weather_data]
    for sensor_data in sensors_data:
        for name, df in sensor_data.items():
            print(f"Preprocessing {name}")
            # create datetime column from time stamp
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            tmp_path = os.path.abspath(os.path.join(MERGED_SEP_DATA_PATH, f"{name}_raw.csv"))
            df.to_csv(tmp_path, index=False)
            # fill na DOC|ENV__SOIL__IRRIGATION column
            # print first timestamp
            df = fill_na_w_zero_df(df, ['DOC|ENV__SOIL__IRRIGATION'])

            # drop cols
            cols_to_drop = ['EXT1|ENV__ATMO__RAIN', 'TOP|ENV__ATMO__RH', 'MTB|ENV__ATMO__T', 'MTB|ENV__ATMO__RAIN__DELTA', 'MTB|ENV__ATMO__RH']
            df = drop_cols(df, cols_to_drop)
            # drop duplicates
            df = drop_duplicates(df)
            # drop almost duplicates
            df = drop_almost_duplicates(df)
            # merge irrigation values w/ prev row
            df = merge_irrigation_values_w_prev_row(df)
            # drop empty rows
            df = drop_empty_rows(df)
            # fill na w/ 0
            df = fill_na_w_zero_df(df, ['DOC|ENV__SOIL__IRRIGATION', 'EXT|ENV__ATMO__RAIN__DELTA'])
            # interpolate columns
            df = add_missing_rows(df)
            df = interpolate_columns(df)
            # round columns
            df = round_columns(df)
            # drop almost empty rows
            df = drop_almost_empty_rows(df)

            # create datetime column from time stamp
            df['timestamp'] = df['timestamp'].astype('int64')
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            # get first timestamp and datetime
            print(f"First timestamp: {df['timestamp'].iloc[0]} - {df['datetime'].iloc[0]}")
            # save processed data to csv
            tmp_path = os.path.abspath(os.path.join(MERGED_SEP_DATA_PATH, f"{name}_final.csv"))
            df.to_csv(tmp_path, index=False)
            print()
            # save to dict
            sensor_data[name] = df

    # Merge
    merge_tree_w_weather_dfs(sensors_data[0], sensors_data[1])
    print("Done")