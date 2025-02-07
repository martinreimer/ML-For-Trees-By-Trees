### Get the weather forecast for the next 7 days from yesterday for all sensors and save them 
import requests
from auth import get_headers
import pandas as pd
from datetime import datetime
import os
import json
import numpy as np

URL = "https://iot.climavi.eu/api/entitiesQuery/find"

def get_data_dump(headers, entity_ids):
    url = URL
    # Body data
    body = {
        "entityFilter": {
            "type": "entityList",
            "entityType": "DEVICE",
            "entityList": entity_ids
        },
          "entityFields": [
      {
          "type": "ENTITY_FIELD",
          "key": "name"
      },
      {
          "type": "ENTITY_FIELD",
          "key": "label"
      }
  ],
  "latestValues": [
      {
          "type": "ATTRIBUTE",
          "key": "FORECAST"
      },
  ],
  "pageLink": {
      "page": 0,
      "pageSize": 100,
      "sortOrder": {
          "key": {
              "key": "label",
              "type": "ENTITY_FIELD"
          },
          "direction": "ASC"
      }
  }
}
    # Make the API call to get time-series data
    response = requests.post(url=url, headers=headers, json=body)

    if response.status_code == 200:
        print("Success HTTP")
        return response.json()
    else:
        print("Error: HTTP", response.status_code)
        response.raise_for_status()


def get_forecast_data(headers, entity_ids):
    data = get_data_dump(headers, entity_ids)
    # Extract the "FORECAST" data
    data_per_sensor = data["data"]
    # iterate over all sensors and save in dict
    forecast_data = []
    # dict to save all data per sensor
    sensors_dict = {}
    
    for sensor in data_per_sensor:
        # get the forecast data
        entity_id = sensor["entityId"]["id"]
        forecast_data = sensor["latest"]["ATTRIBUTE"]["FORECAST"]["value"]
        # Parse the forecast data from JSON string to a Python dictionary
        forecast_data_dict = json.loads(forecast_data)

        # Create an empty list to store forecast data
        forecast_list = []

        # Iterate through the forecast data and extract timestamp and values
        for forecast_item in forecast_data_dict:
            ts = forecast_item["ts"]
            values = forecast_item["values"]
            forecast_list.append({"Timestamp": ts, **values})

        # Create a DataFrame from the forecast list
        forecast_df = pd.DataFrame(forecast_list)

        # Convert the timestamp column to datetime
        forecast_df["Timestamp"] = pd.to_datetime(forecast_df["Timestamp"], unit="ms")

        sensors_dict[entity_id] = forecast_df        
    return sensors_dict


# main method
if __name__ == "__main__":
    # Get the directory of the current script
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    forecasts_path = os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "data", "raw", "climavi", "forecasts"))
    sensor_table_path = os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "data", "climavi_sensor_table.csv"))
    # get absolute path of a folder in a nother super folder

    # try to create a folder for the date and then save the data in that folder
    # current date - 1 day
    date_yesterday = datetime.now().date() - pd.Timedelta(days=1)
    # create folder
    date_folder_path = os.path.join(forecasts_path, str(date_yesterday))
    # check if the folder already exists
    if os.path.exists(date_folder_path):
        print(f"Data for {str(date_yesterday)} already exists")
        exit()
    os.makedirs(date_folder_path, exist_ok=True)

    station_df = pd.read_csv(sensor_table_path)
    entity_ids = list(station_df["entityId"])
    print(f"Fetching data for {len(entity_ids)} sensors")

    headers = get_headers()

    data = get_forecast_data(headers, entity_ids)
    # show first 5 rows of the data
    # show all cols in df print
    pd.set_option('display.max_columns', None)

    # iterate though data dict
    for entity_id in data:
        print(entity_id)
        # get the forecast data
        print(f"Saving data for sensor: {entity_id}")
        # get the sensor name
        sensor_name = station_df[station_df["entityId"] == entity_id]["label"].values[0]
        print(f"Saving data for sensor: {sensor_name}")
        forecast_data = data[entity_id]
        #print(f"Saving data for sensor: {sensor_name} - path: ../data/forecasts/{date_yesterday}/{sensor_name}.csv")
        # save data to csv w/ name containing the current date - 1 day 
        sensor_file_path = os.path.join(date_folder_path, f"{sensor_name}.csv")
        forecast_data.to_csv(sensor_file_path, index=False)

    print("Data saved successfully")