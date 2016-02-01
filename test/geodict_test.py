#!/usr/bin/env python

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
    
    #this should pass, and will serve as the comparison from now on
    gdict = {'xmin':-160.340600,'xmax':-146.340600,
             'ymin':54.104700,'ymax':65.104700,
             'xdim':0.025000,'ydim':0.025000,
             'nrows':441,'ncols':561}
    gd = GeoDict(gdict,preserve=None)

    #this should pass
    gdict = {'xmin':-160.340600,'xmax':-146.340600,
             'ymin':54.104700,'ymax':65.104700,
             'xdim':0.025000,'ydim':0.025000,
             'nrows':440,'ncols':560}
    gd2 = GeoDict(gdict,preserve='dims')
    assert gd2 == gd

    #this should pass
    gdict = {'xmin':-160.340600,'xmax':-146.340600,
             'ymin':54.104700,'ymax':65.104700,
             'xdim':0.026000,'ydim':0.026000,
             'nrows':441,'ncols':561}
    gd3 = GeoDict(gdict,preserve='shape')
    assert gd3 == gd

    #this should pass
    gdict = {'xmin':-160.340600,'xmax':-146.350600,
             'ymin':54.103700,'ymax':65.104700,
             'xdim':0.025000,'ydim':0.025000,
             'nrows':441,'ncols':561}
    gd4 = GeoDict(gdict,preserve='corner')
    assert gd4 == gd
    
    #make sure the lat/lon row/col calculations are correct
    ndec = int(np.abs(np.log10(GeoDict.EPS)))
    lat,lon = gd.getLatLon(0,0)
    dlat = np.abs(lat-gd.ymax)
    dlon = np.abs(lon-gd.xmin)
    assert dlat < GeoDict.EPS and dlon < GeoDict.EPS
    row,col = gd.getRowCol(lat,lon)
    assert row == 0 and col == 0

    lat,lon = gd.getLatLon(gd.nrows-1,gd.ncols-1)
    dlat = np.abs(lat-gd.ymin)
    dlon = np.abs(lon-gd.xmax)
    assert dlat < GeoDict.EPS and dlon < GeoDict.EPS
    row,col = gd.getRowCol(lat,lon)
    assert row == (gd.nrows-1) and col == (gd.ncols-1)

    #this is the file geodict for Landscan - should pass muster
    globaldict = {'ncols': 43200,
                  'nrows': 20880,
                  'xdim': 0.00833333333333,
                  'xmax': 179.99583333318935,
                  'xmin': -179.99583333333334,
                  'ydim': 0.00833333333333,
                  'ymax': 83.99583333326376,
                  'ymin': -89.99583333333334}
    gd5 = GeoDict(globaldict)
    lat,lon = gd5.getLatLon(gd5.nrows-1,gd5.ncols-1)
    dlat = np.abs(lat-gd5.ymin)
    dlon = np.abs(lon-gd5.xmax)
    assert dlat < GeoDict.EPS and dlon < GeoDict.EPS

if __name__ == '__main__':
    test()

        

        
