from geopy.geocoders import Nominatim
import random
import json

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

def build_zipcode_bounds(zipcodes, output_path=f"zip_bounds.json"): #, output_path="zip_bounds.json"):
    geolocator = Nominatim(user_agent="zip-bounds")
    bounds_dict = {}

    for zip_code in zipcodes:
        try:
            location = geolocator.geocode({'postalcode': zip_code}, exactly_one=True, country_codes='us')
            if location and 'boundingbox' in location.raw:
                bounds = list(map(float, location.raw['boundingbox']))  # [lat_min, lat_max, lon_min, lon_max]
                bounds_dict[zip_code] = bounds
                print(f"{zip_code}: {bounds}")
            else:
                print(f"[!] No bounding box for {zip_code}")
        except Exception as e:
            print(f"[x] Failed for {zip_code}: {e}")
        # time.sleep(1)  # respect Nominatim rate limits

    if output_path:
        with open(output_path, "w") as f:
            json.dump(bounds_dict, f, indent=2)

    return bounds_dict

def get_random_coordinate_cached(zipcode, zipcode_bounds):
    zipcode = str(zipcode).zfill(5)
    if str(zipcode).zfill(5) not in zipcode_bounds:
        raise ValueError(f"Zipcode {zipcode} not in bounds cache.")

    lat_min, lat_max, lon_min, lon_max = zipcode_bounds[zipcode]
    lat = random.uniform(lat_min, lat_max)
    lon = random.uniform(lon_min, lon_max)
    return lat, lon

def load_zipcode_bounds(path="zip_bounds.json"):
    with open(path, "r") as f:
        return json.load(f)