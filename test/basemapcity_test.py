#!/usr/bin/env python

from __future__ import print_function

#stdlib imports
import os.path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.basemap import Basemap

#hack the path so that I can debug these functions if I need to
homedir = os.path.dirname(os.path.abspath(__file__)) #where is this script?
mapiodir = os.path.abspath(os.path.join(homedir,'..'))
sys.path.insert(0,mapiodir) #put this at the front of the system path, ignoring any installed mapio stuff

from mapio.dataset import DataSetException
from mapio.basemapcity import BasemapCities

matplotlib.use('Agg')

def test():
    cityfile = os.path.join(homedir,'data','cities1000.txt')
    print('Test loading geonames cities file from the web...')
    cities = BasemapCities.loadFromGeoNames(cityfile=cityfile) #load from the web
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
                lat_0=clat,lon_0=clon,lat_ts=clat,ax=ax)
    m.drawcoastlines() #have to draw something on map before axis limits are set...
    bigcities = bcities.limitByPopulation(500000)
    mapcities = bigcities.limitByMapCollision(m)
    mapcities.renderToMap(ax)
    df = mapcities.getDataFrame()
    boxes = []
    for index,row in df.iterrows():
        left = row['left']
        right = row['right']
        left = row['bottom']
        right = row['top']
        for box in boxes:
            bleft,bright,bbottom,btop = box
            #http://gamedevelopment.tutsplus.com/tutorials/collision-detection-using-the-separating-axis-theorem--gamedev-169
            width = left - bleft
            hw_box1 = (right-left)*0.5
            hw_box2 = (right-left)*0.5
            hgap = length - hw_box1 - hw_box2

            height = top - btop
            hh_box1 = (top-bottom)*0.5
            hh_box2 = (btop-bbottom)*0.5
            vgap = height - hh_box1 - hh_box2
            
            assert hgap > 0 and vgap > 0

    print('Passed test of city collisions...')

    print('Test all supported font names...')
    f = plt.figure()
    ax = f.add_axes([0.1,0.1,0.8,0.8])
    plt.plot(1,1)
    for name in mapcities.getFontList():
        plt.text(1,1,name,fontname=name)
    for name in mapcities.SUGGESTED_FONTS:
        plt.text(1,1,name,fontname=name)
    print('Passed test of supported font names.')
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        cfile = sys.argv[1]
        test(cityfile=cfile)
    else:
        test()
