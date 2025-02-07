import requests
import json
import os 
import time

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_PATH = FOLDER_PATH + "\\" + "credentials.json"
URL="https://iot.climavi.eu:443/api/auth/login"


def get_bearer_token(username, password, url=URL, headers=None):
    """
    Logs in to the Climavi API and returns a bearer token.

    :param username: The username for the Climavi API
    :param password: The password for the Climavi API
    :param url: The login URL for the Climavi API 
    :return: The bearer token if login is successful, otherwise raises an exception
    """

    data = {
        'username': username,
        'password': password
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        token = response.json().get('token')
        if token:
            return token
        else:
            raise Exception("Token not found in the response")
    else:
        error_message = response.json().get('message', 'Login failed')
        raise Exception(f"Error {response.status_code}: {error_message}")

# Function to authenticate and get a new token
def get_new_token():

    with open(CREDENTIALS_PATH, 'r') as jsonfile:
        credentials = json.load(jsonfile)

    headers = {
        'Content-Type': 'application/json'
    }
    # Define the login credentials
    token = get_bearer_token(username=credentials['username'], password=credentials['password'], headers=headers)
    return token


def get_headers():
    # Retrieve token
    token = get_new_token()
    headers = {
        'Content-Type': 'application/json',
        "X-Authorization": f"Bearer {token}"
    }
    return headers

if __name__ == "__main__":
    headers = get_headers()
    print(f"Headers: {headers}")