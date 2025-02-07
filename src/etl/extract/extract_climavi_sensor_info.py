import requests
import pandas as pd
from auth import get_headers
import os

def get_sensor_info(headers):
    query_url = f"https://iot.climavi.eu:443/api/entitiesQuery/find"
    payload = {
        "entityFilter": {
            "type": "entityType",
            "resolveMultiple": True,
            "entityType": "DEVICE"
        },
        "entityFields": [
            {"type": "ENTITY_FIELD", "key": "name"},
            {"type": "ENTITY_FIELD", "key": "label"}
        ],
        "latestValues": [
            {"type": "ATTRIBUTE", "key": "latitude"},
            {"type": "ATTRIBUTE", "key": "longitude"},
        ],
        "pageLink": {
            "page": 0,
            "pageSize": 100,
            "sortOrder": {
                "key": {"key": "label", "type": "ENTITY_FIELD"},
                "direction": "ASC"
            }
        }
    }
    response = requests.post(query_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

def get_sensor_df(headers):
    sensor_dict = get_sensor_info(headers)
    # Extract information
    records = []
    for sensor in sensor_dict['data']:
        entity_id = sensor['entityId']['id']
        latitude = sensor['latest']['ATTRIBUTE']['latitude']['value']
        longitude = sensor['latest']['ATTRIBUTE']['longitude']['value']
        label = sensor['latest']['ENTITY_FIELD']['label']['value']
        name_id = sensor['latest']['ENTITY_FIELD']['name']['value']
        is_favorite = False
        is_weather_station = label.startswith("Wetter ")        
        records.append({'entityId': entity_id, 'nameId': name_id, 'label': label, 'is_weather_station': is_weather_station, 'latitude': latitude, 'longitude': longitude, 'is_favorite': is_favorite})

    # Create DataFrame
    df = pd.DataFrame(records)
    return df


headers = get_headers() 
df = get_sensor_df(headers=headers)
# Export to CSV
cur_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(cur_dir, "..", "..", "..", "data")
file_path = os.path.join(base_dir, f"sensor_table1.csv")
df.to_csv(file_path, index=False)