import requests
import time
from auth import get_headers
import pandas as pd
from datetime import datetime
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

URL_TEMPLATE = "https://iot.climavi.eu/api/plugins/telemetry/DEVICE/{entityId}/values/timeseries"

weather_keys_list = [
    "TOP|ENV__ATMO__T",
    "TOP|ENV__ATMO__RH",
    "TOP|ENV__ATMO__IRRADIATION",
    "EXT|ENV__ATMO__RAIN__DELTA"
    ]

WEATHER_KEYS = ",".join(weather_keys_list)


def get_data_dump(headers, entity_id, from_date_str, keys=WEATHER_KEYS):
    url = URL_TEMPLATE.format(entityId=entity_id)
    
    # Parse the from_date string into a datetime object
    from_date = datetime.strptime(from_date_str, "%Y-%m-%d")

    # Convert from_date to timestamp in milliseconds
    start_ts = int(from_date.timestamp() * 1000)
    end_ts = int(time.time() * 1000)  # Current time

    params = {
        "keys": keys,
        "startTs": start_ts,
        "endTs": end_ts,
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
        record = {'timestamp': ts, 'datetime': datetime.utcfromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S')}

        for key in data_dump:
            # Find the value for the current timestamp if it exists
            value = next((entry['value'] for entry in data_dump[key] if entry['ts'] == ts), None)
            record[key] = value
        records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)
    
    return df


# This script iterates through a predefined list of sensors, checking for existing data files for each sensor.
# If a file exists, it fetches new sensor data starting from the last recorded date; otherwise, it fetches all available data, merging and de-duplicating entries before saving updated data back to CSV.
if __name__ == "__main__":
    # Welche Sensor Daten holen? OPTIONS = ['all', 'favorites', 'self_defined']
    option = 'all'#'favorites'    
    print(f"Fetching sensor data for option: {option}")
    if option is 'all' or option is 'favorites':
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.join(cur_dir, "..", "..", "..", "data", "climavi_sensor_table.csv")
        sensors_df = pd.read_csv(abs_path)
        # is_weather_station should be false
        sensors_df = sensors_df[sensors_df["is_weather_station"] == True]
        if option is 'favorites':
            sensors_df = sensors_df[sensors_df["is_favorite"] == True]
        sensors = dict(zip(sensors_df["label"], sensors_df["entityId"]))
    else:
        sensors = {
            "Wetter_Westbad": "3b636af0-bc54-11ee-92b2-4f7676009c58",
            "Wetter_Adalbert_Stifter": "f6d23140-bcef-11ee-92b2-4f7676009c58",
            "Wetter_Bruck": "5f4e7250-bc56-11ee-92b2-4f7676009c58"
        }
    default_from_date_str = "2023-01-01"
    headers = get_headers()

    # Base directory for data storage
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(cur_dir, "..", "..", "..", "data", "raw", "climavi", "weather_sensors")

    # Process each sensor
    for sensor_name, sensor_id in tqdm(sensors.items()):
        file_path = os.path.join(base_dir, f"{sensor_name}.csv")
        #print(f"Checking file: {file_path}")
        file_exists = False
        if os.path.exists(file_path):
            #print(f"File found for {sensor_name}.")
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
            print("\nNo existing file found. Fetching all available data...")

        # Fetch new data
        df_new = get_sensor_data(headers, sensor_id, from_date_str)
        print(f"df len: {len(df_new)}")
        if file_exists:
            # Combine existing and new data
            df_combined = pd.concat([df_existing, df_new]).drop_duplicates(subset='datetime', keep='last')
        else:
            df_combined = df_new

        # Save combined data to CSV
        df_combined.to_csv(file_path, index=False)
        print(f"\ndf len: {len(df_new)}")
        print(f"\nData saved for {sensor_name}.")
