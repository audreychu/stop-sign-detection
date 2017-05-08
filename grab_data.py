import requests
import json
import pandas
from datetime import datetime
import numpy as np
import urllib
import os
import random


def GetStreet(geom,fov,name, SaveLoc ,heading = None,key = "AIzaSyC8gYKEg9WRbKU-abM0dAgMENvk8frwjAU"):
    """Takes in latitude, longitude, fov, name of file, save location, heading (optional), and key (optional). It will randomly change lat and lon, and grab streetview image from there."""
    split = geom.split(",")
    lat = float(split[0][1:])
    lon = float(split[1][:-1])
    lat += random.uniform(-.00015,.00015)
    lon += random.uniform(-.00015,.00015)
    if heading == None:
        pot_head = [0,90,180,270]
    for heading in pot_head:
        MyUrl = "https://maps.googleapis.com/maps/api/streetview?size=640x436&location={0},{1}&fov={2}&heading={3}&key={4}".format(lat,lon,fov,heading,key)

        fi = name +str(heading) + ".jpg"
        urllib.request.urlretrieve(MyUrl, os.path.join(SaveLoc,fi))

def ReverseGeo(lat=35.1330343, lon=-90.0625056, api='AIzaSyBwTLTIHYJU_osZ-KKE-HlTH9EcowYJjDs'):
    """Takes lat (float), lon (float), and api key (string) (optional), returns zipcode for now, plan to return distance to nearest city in the future"""
    sensor = 'false'
    base = "https://maps.googleapis.com/maps/api/geocode/json?"
    params = "latlng={lat},{lon}&sensor={sen}&result_type=postal_code&key={api}".format(
        lat=lat,
        lon=lon,
        sen=sensor,
        api=api
    )
    url = "{base}{params}".format(base=base, params=params)
    response = requests.get(url)
    x = json.loads(response.text)
    try:
        zipcode = x['results'][0]['address_components'][0]['short_name']
        return zipcode
    except:
        return np.nan
    def main():
        pass

__all__ = ('GetStreet','ReverseGeo')


