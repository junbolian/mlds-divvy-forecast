import requests

Divvy_URL = "https://api.citybik.es/v2/networks/divvy"

def fetch_divvy_data():
    response = requests.get(Divvy_URL, timeout=10)
    response.raise_for_status()
    data = response.json()["network"]
    return data