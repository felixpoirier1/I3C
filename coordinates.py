from geopy.geocoders import Nominatim
import shapely

geolocator = Nominatim(user_agent="My app")

def fetch_coordinates(name: str) -> tuple:
    geo = geolocator.geocode(f"Tulsa")
    return (geo.point.latitude, geo.point.longitude)

if __name__ == "__main__":
    print(fetch_coordinates("Tulsa"))