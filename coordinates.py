from geopy.geocoders import Nominatim
import shapely

geolocator = Nominatim(user_agent="My app")

def fetch_coordinates(name: str, country: str) -> tuple:
    geo = geolocator.geocode(f"{name}, {country}")
    return (geo.point.latitude, geo.point.longitude)

if __name__ == "__main__":
    print(fetch_coordinates("New Jersey (Central)", "US"))