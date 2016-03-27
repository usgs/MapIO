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
from mapio.mapcity import MapCities
    
def test(cityfile=None):
    print('Test loading geonames cities file from the web...')
    cities = MapCities.loadFromGeoNames(cityfile=cityfile) #load from the web
    print('Passed loading geonames cities file from the web.')

    print('Test limiting cities using California bounds...')
    ymin,ymax = 32.394, 42.062
    xmin,xmax = -125.032, -114.002
    bcities = cities.limitByBounds((xmin,xmax,ymin,ymax))
    print('Done limiting cities using California bounds.')
    
    print('Test removing cities with collisions...')
    ymin,ymax = 32.394, 42.062
    xmin,xmax = -125.032, -114.002
    clat = (ymin+ymax)/2.0
    clon = (xmin+xmax)/2.0
    f = plt.figure(figsize=(8,8))
    ax = f.add_axes([0.1,0.1,0.8,0.8])
    BASEMAP_RESOLUTION = 'l'
    m = Basemap(llcrnrlon=xmin,llcrnrlat=ymin,urcrnrlon=xmax,urcrnrlat=ymax,
                rsphere=(6378137.00,6356752.3142),
                resolution=BASEMAP_RESOLUTION,projection='merc',
                lat_0=clat,lon_0=clon,lat_ts=clat)
    m.drawcoastlines() #have to draw something on map before axis limits are set...
    axmin,axmax,aymin,aymax = m.llcrnrx,m.urcrnrx,m.llcrnry,m.urcrnry
    bcities.project(m)
    bigcities = bcities.limitByMapCollision('Arial',10,ax)
    #how to test this?
    print('Passed removing cities with collisions.')
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        cfile = sys.argv[1]
        test(cityfile=cfile)
    else:
        test()
