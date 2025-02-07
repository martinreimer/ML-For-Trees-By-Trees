import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import os

# Constants + Parameters
url = "https://archive-api.open-meteo.com/v1/archive"
# Specify date range
start_date = "2023-01-01"
end_date = pd.Timestamp.now(tz = "Europe/Berlin").date() # current date
# specify coordinates
latitude = 49.591
longitude = 11.0078
# Specify parameters for the API call
hourly_params = ["temperature_2m", "relative_humidity_2m", "precipitation", "weather_code", "cloud_cover", "et0_fao_evapotranspiration", "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm", "is_day", "sunshine_duration", "direct_radiation"]
daily_params = ["weather_code", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "sunshine_duration", "precipitation_sum", "precipitation_hours", "et0_fao_evapotranspiration"]
# Specify timezone
timezone = "Europe/Berlin"
# Specify export location
cur_dir = os.path.dirname(os.path.abspath(__file__))
export_dir = os.path.join(cur_dir, "..", "..", "..", "data", "raw", "openmeteo")


def get_api_response(url, params):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)
    responses = openmeteo.weather_api(url, params=params)
    return responses


def process_hourly_response(hourly):
    # Process hourly data. The order of variables needs to be the same as requested.
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
    hourly_weather_code = hourly.Variables(3).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(4).ValuesAsNumpy()
    hourly_et0_fao_evapotranspiration = hourly.Variables(5).ValuesAsNumpy()
    hourly_soil_moisture_0_to_7cm = hourly.Variables(6).ValuesAsNumpy()
    hourly_soil_moisture_7_to_28cm = hourly.Variables(7).ValuesAsNumpy()
    hourly_soil_moisture_28_to_100cm = hourly.Variables(8).ValuesAsNumpy()
    hourly_is_day = hourly.Variables(9).ValuesAsNumpy()
    hourly_sunshine_duration = hourly.Variables(10).ValuesAsNumpy()
    hourly_direct_radiation = hourly.Variables(11).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["weather_code"] = hourly_weather_code
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data["et0_fao_evapotranspiration"] = hourly_et0_fao_evapotranspiration
    hourly_data["soil_moisture_0_to_7cm"] = hourly_soil_moisture_0_to_7cm
    hourly_data["soil_moisture_7_to_28cm"] = hourly_soil_moisture_7_to_28cm
    hourly_data["soil_moisture_28_to_100cm"] = hourly_soil_moisture_28_to_100cm
    hourly_data["is_day"] = hourly_is_day
    hourly_data["sunshine_duration"] = hourly_sunshine_duration
    hourly_data["direct_radiation"] = hourly_direct_radiation

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    return hourly_dataframe


def process_daily_response(response):
    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_weather_code = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(2).ValuesAsNumpy()
    daily_temperature_2m_mean = daily.Variables(3).ValuesAsNumpy()
    daily_sunshine_duration = daily.Variables(4).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(5).ValuesAsNumpy()
    daily_precipitation_hours = daily.Variables(6).ValuesAsNumpy()
    daily_et0_fao_evapotranspiration = daily.Variables(7).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}
    daily_data["weather_code"] = daily_weather_code
    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_min"] = daily_temperature_2m_min
    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data["sunshine_duration"] = daily_sunshine_duration
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["precipitation_hours"] = daily_precipitation_hours
    daily_data["et0_fao_evapotranspiration"] = daily_et0_fao_evapotranspiration

    daily_dataframe = pd.DataFrame(data = daily_data)
    return daily_dataframe


if __name__ == "__main__":
    print("Fetching weather data from Open-Meteo API...")
    params = {
        # Coordinates of the location: Erlangen, Germany
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_params,
        "daily": daily_params,
        "timezone": timezone
    }
    responses = get_api_response(url, params)    

    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data
    hourly = response.Hourly()
    df_hourly = process_hourly_response(hourly)
    # Process daily data
    df_daily = process_daily_response(response)

    # export to csv
    hourly_path = os.path.join(export_dir, f"Weather_Hourly.csv")
    daily_path = os.path.join(export_dir, f"Weather_Daily.csv")
    df_hourly.to_csv(hourly_path, index = False)
    df_daily.to_csv(daily_path, index = False)
    print(f"Data exported to {export_dir}.")

