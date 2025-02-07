import requests
import time
from auth import get_headers
import pandas as pd
from datetime import datetime
import os
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings("ignore")

URL_TEMPLATE = "https://iot.climavi.eu/api/plugins/telemetry/DEVICE/{entityId}/values/timeseries"

sensor_keys_list = [
    "-10|ENV__SOIL__VWC",
    "-30|ENV__SOIL__VWC",
    "-45|ENV__SOIL__VWC",
    "-10|ENV__SOIL__T",
    "-30|ENV__SOIL__T",
    "-45|ENV__SOIL__T",
    "TOP|ENV__ATMO__T",
    "DOC|ENV__SOIL__IRRIGATION"
]

SENSOR_KEYS = ",".join(sensor_keys_list)

def get_data_dump(headers, entity_id, from_date_str, keys=SENSOR_KEYS):
    url = URL_TEMPLATE.format(entityId=entity_id)
    
    # Parse the from_date string into a datetime object
    from_date = datetime.strptime(from_date_str, "%Y-%m-%d")

    # Convert from_date to timestamp in milliseconds
    start_ts = int(from_date.timestamp() * 1000)
    end_ts = int(time.time() * 1000)  # Current time

    params = {
        "startTs": start_ts,
        "endTs": end_ts,
        "keys": keys,
        "orderBy": "DESC",
        "useStrictDataTypes": True,
        "limit": 100000000
    }
    # Print parameters for debugging
    #print(params)

    # Make the API call to get time-series data
    response = requests.get(url=url, headers=headers, params=params)

    if response.status_code == 200:
        #print("Success")
        return response.json()
    else:
        print("Error: HTTP", response.status_code)
        response.raise_for_status()




def get_sensor_data(headers, entity_id, from_date_str):
    '''
    - Extract All Unique Timestamps: Collect all unique timestamps from the data to ensure that every timestamp is represented in the DataFrame.
    - Create Records: For each unique timestamp, create a record with the corresponding values for each sensor type. If a value for a specific sensor type does not exist at a given timestamp, it will be set to None.
    - Convert Timestamps to Datetime: Convert the timestamps to a human-readable datetime format.
    - Create DataFrame: Build the DataFrame from the list of records.
    '''
    data_dump = get_data_dump(headers, entity_id, from_date_str)
    
    # Extract all unique timestamps
    all_timestamps = set()
    for key in data_dump:
        for entry in data_dump[key]:
            all_timestamps.add(entry['ts'])

    # Create a list to store records for the DataFrame
    records = []

    # Iterate through all unique timestamps
    for ts in sorted(all_timestamps):
        record = {'timestamp': ts, 'datetime': datetime.utcfromtimestamp(ts / 1000)}

        for key in data_dump:
            # Find the value for the current timestamp if it exists
            value = next((entry['value'] for entry in data_dump[key] if entry['ts'] == ts), None)
            record[key] = value
        records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)
    
    return df

# Function to convert timestamp to datetime
def timestamp_to_datetime(timestamp):
    return datetime.utcfromtimestamp(timestamp/1000)



def merge_sensor_data_w_irrigations(df_sensor, sensor_id):
    '''
    merge data with external irrigation data (not needed anymore)
    '''
    # Load json
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(cur_dir, "..", "..", "data", "external", "Irrigations2023.json")
    with open(json_path, 'r') as file:
        data = json.load(file)
    json_df = pd.DataFrame(data['result'])
    # Filter the JSON data for the specific sensor_id
    filtered_json_df = json_df[json_df['sensor_id'] == sensor_id]
    
    print(f"Found {len(filtered_json_df)} irrigation entries for sensor {sensor_id}")
    # Iterate through the filtered JSON data and update the DataFrame
    for _, row in filtered_json_df.iterrows():
        timestamp = row['timestamp']
        giessmenge = row['giessmenge']
        dt = timestamp_to_datetime(timestamp) 

        # Check if the timestamp already exists in the DataFrame
        if timestamp in df_sensor['timestamp'].values:
            # Update the existing row
            df_sensor.loc[df_sensor['timestamp'] == timestamp, 'DOC|ENV__SOIL__IRRIGATION'] = df_sensor.loc[df_sensor['timestamp'] == timestamp, 'DOC|ENV__SOIL__IRRIGATION'].fillna(0) + giessmenge
        else:
            # Add a new row
            new_row = pd.DataFrame({'timestamp': [timestamp], 'DOC|ENV__SOIL__IRRIGATION': [giessmenge], 'datetime': [dt]})
            df_sensor = pd.concat([df_sensor, new_row], ignore_index=True)


    # Convert timestamp to datetime in the existing DataFrame
    df_sensor['datetime'] = df_sensor['timestamp'].apply(timestamp_to_datetime)
    # Sort the DataFrame by the 'timestamp' column
    df_sensor = df_sensor.sort_values(by='timestamp')
    return df_sensor

# This script iterates through a predefined list of sensors, checking for existing data files for each sensor.
# If a file exists, it fetches new sensor data starting from the last recorded date; otherwise, it fetches all available data, merging and de-duplicating entries before saving updated data back to CSV.
if __name__ == "__main__":
    # Welche Sensor Daten holen? OPTIONS = ['all', 'favorites', 'self_defined']
    ONLY_FAVORITES = False #True   
    cur_dir = os.path.dirname(os.path.abspath(__file__)) 
    abs_path = os.path.join(cur_dir, "..", "..", "..", "data", "climavi_sensor_table.csv")
    sensors_df = pd.read_csv(abs_path)
    # is_weather_station should be false
    sensors_df = sensors_df[sensors_df["is_weather_station"] == False]
    if ONLY_FAVORITES:
        sensors_df = sensors_df[sensors_df["is_favorite"] == True]


    # Create a dictionary with 3 values per key using a dictionary comprehension
    sensors = {
        row["label"]: (row["entityId"], row["nameId"])
        for _, row in sensors_df.iterrows()
    }

    default_from_date_str = "2023-01-01"

    headers = get_headers()

    # Base directory for data storage
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(cur_dir, "..", "..", "..", "data", "raw", "climavi", "tree_sensors")

    # Process each sensor using tqdm for the progress bar
    for sensor_name, (sensor_entity_id, sensor_name_id) in tqdm(sensors.items()):
        file_path = os.path.join(base_dir, f"{sensor_name}.csv")
        #print(f"Checking file: {file_path}")
        file_exists = False
        if os.path.exists(file_path):
            file_exists = True
            # Load existing data
            df_existing = pd.read_csv(file_path)
            # Convert 'datetime' column to datetime
            df_existing['datetime'] = pd.to_datetime(df_existing['datetime'])
            # Get the most recent date
            max_date = df_existing['datetime'].max().date()
            # Convert to string in format 'YYYY-MM-DD' for API call
            from_date_str = max_date.strftime("%Y-%m-%d")
            print(f"\nFetching new data from {from_date_str}...")
        else:
            from_date_str = default_from_date_str
            print(f"\nNo existing file found. Fetching all available data...")

        # Fetch new data
        df_new = get_sensor_data(headers, sensor_entity_id, from_date_str)

        if file_exists:
            # Combine existing and new data
            df_combined = pd.concat([df_existing, df_new]).drop_duplicates(subset='datetime', keep='last')
        else:
            df_combined = df_new
            # Combine data w/ irrigation data
            #df_combined = merge_sensor_data_w_irrigations(df_combined, sensor_name_id)

        # Save combined data to CSV
        try:
            df_combined.to_csv(file_path, index=False)
            print(f"Data saved for {sensor_name}.")
        except Exception as e:
            print(f"Error saving data for {sensor_name}: {e}")
            continue