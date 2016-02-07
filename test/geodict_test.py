#!/usr/bin/env python

from __future__ import print_function

#stdlib imports
import os.path
import sys

import numpy as np

#hack the path so that I can debug these functions if I need to
homedir = os.path.dirname(os.path.abspath(__file__)) #where is this script?
mapiodir = os.path.abspath(os.path.join(homedir,'..'))
sys.path.insert(0,mapiodir) #put this at the front of the system path, ignoring any installed mapio stuff

from mapio.dataset import DataSetException
from mapio.geodict import GeoDict

def test():
    #these values taken from the shakemap header of: 
    #http://earthquake.usgs.gov/realtime/product/shakemap/ak12496371/ak/1453829475592/download/grid.xml

    print('Testing various dictionaries for consistency...')

    print('Testing consistent dictionary...')
    #this should pass, and will serve as the comparison from now on
    gdict = {'xmin':-160.340600,'xmax':-146.340600,
             'ymin':54.104700,'ymax':65.104700,
             'dx':0.025000,'dy':0.025000,
             'ny':441,'nx':561}
    gd = GeoDict(gdict,adjust=None)
    print('Consistent dictionary passed.')

    print('Testing dictionary with inconsistent dimensions...')
    #this should pass
    gdict = {'xmin':-160.340600,'xmax':-146.340600,
             'ymin':54.104700,'ymax':65.104700,
             'dx':0.026000,'dy':0.026000,
             'ny':441,'nx':561}
    gd3 = GeoDict(gdict,adjust='res')
    assert gd3 == gd
    print('Dimensions modification passed.')

    print('Testing dictionary with inconsistent lower right corner...')
    #this should pass
    gdict = {'xmin':-160.340600,'xmax':-146.350600,
             'ymin':54.103700,'ymax':65.104700,
             'dx':0.025000,'dy':0.025000,
             'ny':441,'nx':561}
    gd4 = GeoDict(gdict,adjust='bounds')
    assert gd4 == gd
    print('Corner modification passed.')

    print('Testing to make sure lat/lon and row/col calculations are correct...')
    #make sure the lat/lon row/col calculations are correct
    ndec = int(np.abs(np.log10(GeoDict.EPS)))
    lat,lon = gd.getLatLon(0,0)
    dlat = np.abs(lat-gd.ymax)
    dlon = np.abs(lon-gd.xmin)
    assert dlat < GeoDict.EPS and dlon < GeoDict.EPS
    row,col = gd.getRowCol(lat,lon)
    assert row == 0 and col == 0

    lat,lon = gd.getLatLon(gd.ny-1,gd.nx-1)
    dlat = np.abs(lat-gd.ymin)
    dlon = np.abs(lon-gd.xmax)
    assert dlat < GeoDict.EPS and dlon < GeoDict.EPS
    row,col = gd.getRowCol(lat,lon)
    assert row == (gd.ny-1) and col == (gd.nx-1)
    print('lat/lon and row/col calculations are correct.')

    print('Testing a dictionary for a global grid...')
    #this is the file geodict for Landscan - should pass muster
    globaldict = {'nx': 43200,
                  'ny': 20880,
                  'dx': 0.00833333333333,
                  'xmax': 179.99583333318935,
                  'xmin': -179.99583333333334,
                  'dy': 0.00833333333333,
                  'ymax': 83.99583333326376,
                  'ymin': -89.99583333333334}
    gd5 = GeoDict(globaldict)
    lat,lon = gd5.getLatLon(gd5.ny-1,gd5.nx-1)
    dlat = np.abs(lat-gd5.ymin)
    dlon = np.abs(lon-gd5.xmax)
    assert dlat < GeoDict.EPS and dlon < GeoDict.EPS
    print('Global grid is internally consistent.')

    #Test class methods for creating a GeoDict
    print('Testing whether GeoDict creator class methods work...')
    xmin = -121.05333277776235
    xmax = -116.03833388890432
    ymin = 32.138334444506171
    ymax = 36.286665555493826
    dx = 0.0083333333333333332
    dy = 0.0083333333333333332
    gd6 = GeoDict.createDictFromBox(xmin,xmax,ymin,ymax,dx,dy,inside=False)
    assert gd6.xmax > xmax
    assert gd6.ymin < ymin
    print('Created dictionary (outside) is correct.')
    gd7 = GeoDict.createDictFromBox(xmin,xmax,ymin,ymax,dx,dy,inside=True)
    assert gd7.xmax < xmax
    assert gd7.ymin > ymin
    print('Created dictionary (inside) is correct.')
    xspan = 2.5
    yspan = 2.5
    gd8 = GeoDict.createDictFromCenter(xmin,ymin,dx,dy,xspan,yspan)
    print('Created dictionary (from center point) is valid.')

    #test getBoundsWithin
    #use global grid, and then a shakemap grid that we can get
    print('Testing getBoundsWithin...')
    grussia = {'xmin':155.506400,'xmax':161.506400,
               'ymin':52.243000,'ymax':55.771000,
               'dx':0.016667,'dy':0.016642,
               'nx':361,'ny':213}
    gdrussia = GeoDict(grussia,adjust='res')
    sampledict = gd5.getBoundsWithin(gdrussia)
    xSmaller = sampledict.xmin > grussia['xmin'] and sampledict.xmax < grussia['xmax']
    ySmaller = sampledict.ymin > grussia['ymin'] and sampledict.ymax < grussia['ymax']
    assert xSmaller and ySmaller
    print('getBoundsWithin returned correct result.')
    
if __name__ == '__main__':
    test()
    #test_fail()

        

        
