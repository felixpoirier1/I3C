import requests

def fetch_elevation(lon: float, lat: float):
    res = requests.get(f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}")
    return res.json()["results"][0]["elevation"]

if __name__ == "__main__":
    #random point in the middle of the ocean
    fetch_elevation(31.416944, -56.999877)