from geopy.geocoders import Nominatim
import random


def get_random_coordinate(zipcode):
    """
    returns a randomized lat, lon based on zipcode
    """
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode({'postalcode': zipcode}, exactly_one=True, country_codes='us')
    if location is None:
        raise ValueError(f"Zip code {zipcode} not found.")

    bounds = location.raw['boundingbox']
    lat = random.uniform(float(bounds[0]), float(bounds[1]))
    lon = random.uniform(float(bounds[2]), float(bounds[3]))
    return lat, lon