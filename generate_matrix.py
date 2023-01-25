import os

import numpy as np
import json
import time
import matplotlib.pyplot as plt
import geopandas as gpd
import shapely
from tqdm import tqdm
from PIL import Image

class xMap():
    def __init__(self, lon: list, lat: list):
        assert len(lon) == 2, "lon must be a list of two elements"
        assert len(lat) == 2, "lat must be a list of two elements"
        self.lon_left = lon[0]
        self.lon_right = lon[1]
        self.lat_bottom = lat[0]
        self.lat_top = lat[1]

    def generate_matrix(self, precision: float) -> np.ndarray:
        self.precision = precision
        self.lon_qty = int(np.ceil((self.lon_right - self.lon_left)/self.precision))
        self.lat_qty = int(np.ceil((self.lat_top - self.lat_bottom)/self.precision))

        self.xmap = np.zeros((self.lat_qty, self.lon_qty, 2))

        self.latitude = np.linspace(self.lat_bottom, self.lat_top, self.lat_qty, endpoint=False, dtype=float)
        self.longitude = np.linspace(self.lon_left, self.lon_right, self.lon_qty, endpoint=False, dtype=float)

        for i, lat in enumerate(self.latitude):
            for j, lon in enumerate(self.longitude):
                self.xmap[i,j,0] = lat
                self.xmap[i,j,1] = lon

        return self.xmap




class landMass():
    def __init__(self, xmap: np.ndarray, land: shapely.geometry.Polygon):
        self.xmap = xmap
        self.land = land
        self.lat_qty = xmap.shape[0]
        self.lon_qty = xmap.shape[1]
        self.lat_range = [xmap[:,:,0].max(), xmap[:,:,0].min()]
        self.lon_range = [xmap[:,:,1].max(), xmap[:,:,1].min()]

    def findlandmass(self) -> np:
        '''
        takes a shapely.geometry.Polygon and returns a vector of 0 and 1, 0 means water, 1 means land
        for every coordinate within the xmap
        '''

        shapely.speedups.enable()

        start_time = time.time()

        # create empty matrix
        self.landmass = np.zeros((self.xmap.shape[0], self.xmap.shape[1]), dtype=bool)

        # iterate over all points in the xmap
        for i in tqdm(range(self.xmap.shape[0]), desc='latitude', ascii=True):
            for j in range(self.xmap.shape[1]):
                # find the closest point in the xmap
                lat = self.xmap[i,j,0]
                lon = self.xmap[i,j,1]
                # set the point to land
                self.landmass[i,j] = self.land.contains(shapely.geometry.Point(lon, lat))

        print(f"--- {time.time() - start_time} seconds ---")
        return self.landmass

    def classify_landmass(self, landmass: np.ndarray, classes: dict) -> np.ndarray:
        '''
        takes a landmass vector and a dictionary of classes and returns a landmass vector with the classes
        '''
        self.landmass = landmass
        self.classes = classes

        # create empty matrix
        self.classified_landmass = np.zeros((self.xmap.shape[0], self.xmap.shape[1]), dtype=int)

        # iterate over all points in the xmap
        #for i in tqdm(range(self.xmap.shape[0]), desc='latitude', ascii=True):
            


def populate(xmap : np.ndarray, lon: list, lat: list, val: list):
    layer = np.zeros((xmap.shape[0], xmap.shape[1]), dtype=float)
    for i in range(len(lon)):
        # given latitute and longitude, find the closest point in the xmap
        latitude = lat[i] if lat[i] > xMap[:,:, 0].min() and lat[i] < xMap[:,:, 0].max() else False
        longitude = lon[i] if lon[i] > xMap[:,:, 1].min() and lon[i] < xMap[:,:, 1].max() else False
        if latitude and longitude:
            x, y = np.unravel_index(np.argmin(np.sqrt((xMap[:, :, 0] - latitude) ** 2 + (xMap[:, :, 1] - longitude) ** 2)), xMap.shape[:2])
            if x != last_x and y != last_y:
                print(x, y)
            layer[x, y] += val[i]
        last_x = x; last_y = y
    return layer

    
def populate(xmap : np.ndarray, lon: list, lat: list, val: list):
    layer = np.zeros((xmap.shape[0], xmap.shape[1]), dtype=float)
    last_x = None
    last_y = None
    for i in range(len(lon)):
        # given latitute and longitude, find the closest point in the xmap
        latitude = lat[i] if lat[i] > xmap[:,:, 0].min() and lat[i] < xmap[:,:, 0].max() else False
        longitude = lon[i] if lon[i] > xmap[:,:, 1].min() and lon[i] < xmap[:,:, 1].max() else False
        if latitude and longitude:
            x, y = np.unravel_index(np.argmin(np.sqrt((xmap[:, :, 0] - latitude) ** 2 + (xmap[:, :, 1] - longitude) ** 2)), xmap.shape[:2])
            if x != last_x and y != last_y:
                layer[x, y] += val[i]
            last_x = x; last_y = y
    return layer



if __name__ == '__main__':
    
    lon_left = -125
    lon_right = -66
    lat_bottom = 24
    lat_top = 50

    # precision = 0.5
    # xmap = xMap([lon_left, lon_right], [lat_bottom, lat_top])
    # us_map = xmap.generate_matrix(precision)
    # np.save('output/us_map.npy', us_map)

    # #USA
    # print("USA -- MAIN MAP")
    # world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # us = world[world.name == 'United States of America'].iloc[0]['geometry'].geoms[0]
    # us_landmap = landMass(us_map, us)
    # us_landmass = us_landmap.findlandmass()
    # np.save('output/us_landmass.npy', us_landmass[::-1])

    # #CANADA
    # print("CANADA")
    # can = world[world.name == 'Canada'].iloc[0]['geometry'].geoms[0]
    # can_landmap = landMass(us_map, can)
    # can_landmass = can_landmap.findlandmass()
    # np.save('output/can_landmass.npy', can_landmass[::-1])

    # #MEXICO
    # print("MEXICO")
    # mex = world[world.name == 'Mexico'].iloc[0]['geometry']
    # mex_landmap = landMass(us_map, mex)
    # mex_landmass = mex_landmap.findlandmass()
    # np.save('output/mex_landmass.npy', mex_landmass[::-1])
    for i in range(1, 100):
        if i%5 == 0:
            precision = i/100
            xmap = xMap([lon_left, lon_right], [lat_bottom, lat_top])
            us_map = xmap.generate_matrix(precision)

            #USA
            print(f"USA -- MAIN MAP -- precision of {precision}")
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            us = world[world.name == 'United States of America'].iloc[0]['geometry'].geoms[0]
            us_landmap = landMass(us_map, us)
            us_landmass = us_landmap.findlandmass()
            Image.fromarray(us_landmass[::-1]).resize((1200, 532)).save(f'usmaps/usmap_{str(i).zfill(2)}.png')        