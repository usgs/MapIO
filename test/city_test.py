#!/usr/bin/env python

from __future__ import print_function

#stdlib imports
import os.path
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

#hack the path so that I can debug these functions if I need to
homedir = os.path.dirname(os.path.abspath(__file__)) #where is this script?
mapiodir = os.path.abspath(os.path.join(homedir,'..'))
sys.path.insert(0,mapiodir) #put this at the front of the system path, ignoring any installed mapio stuff

from mapio.dataset import DataSetException
from mapio.city import Cities

def test(cityfile=None):
    print('Test loading geonames cities file from the web...')
    cities = Cities.loadFromGeoNames(cityfile=cityfile) #load from the web
    print('Passed loading geonames cities file from the web.')

    print('Test getting city names and coordinates...')
    lat,lon,names = cities.getCities()
    assert len(lat) == len(cities)
    print('Passed getting city names and coordinates...')
    
    print('Test limiting cities using California bounds...')
    ymin,ymax = 32.394, 42.062
    xmin,xmax = -125.032, -114.002
    bcities = cities.limitByBounds((xmin,xmax,ymin,ymax))
    bounds = bcities.getBounds()
    assert bounds[0] > xmin and bounds[1] < xmax and bounds[2] > ymin and bounds[3] < ymax
    print('Passed limiting cities using California bounds.')

    print('Test limiting cities using a 4 by 4 grid...')
    gcities = bcities.limitByGrid(nx=2,ny=4,cities_per_grid = 10)
    assert len(gcities) <= 2*4*10
    print('Passed limiting cities using California bounds.')
    
       
    print('Test getting cities by name (Los Angeles)...')
    cityofangels = bcities.limitByName('Los Angeles').limitByPopulation(1000000)
    assert len(cityofangels) == 1
    print('Passed getting cities by name (Los Angeles).')

    print('Test limiting cities 50 km radius around LA...')
    clat,clon = 34.048019,-118.244133
    rcities = bcities.limitByRadius(clat,clon,50)
    print('Passed limiting cities using California bounds.')

    print('Test limiting cities above population above 50,000...')
    popthresh = 50000
    bigcities = rcities.limitByPopulation(popthresh)
    df = bigcities.getDataFrame()
    assert df['pop'].max() >= popthresh
    print('Test limiting cities above population above 50,000.')
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        cfile = sys.argv[1]
        test(cityfile=cfile)
    else:
        test()
