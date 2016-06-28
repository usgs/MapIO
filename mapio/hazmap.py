#!/usr/bin/env python

#python 3 compatibility
from __future__ import print_function

#stdlib imports
import abc
import textwrap
import glob
import os
import sys

#third party imports
from .grid2d import Grid2D
from .dataset import DataSetException
from .geodict import GeoDict

import numpy as np
from scipy import interpolate
import shapely
from affine import Affine
from rasterio import features
from shapely.geometry import MultiPoint,Polygon,mapping

def read_grid_array(filename):
    #read in lat/lon/z values from text file
    fpath,ffile = os.path.split(filename)
    ffile,fext = os.path.splitext(ffile)
    lats = []
    lons = []
    z = []
    f = open(filename,'rt')
    for line in f.readlines():
        parts = line.split()
        lons.append(float(parts[0]))
        lats.append(float(parts[1]))
        z.append(float(parts[2]))
    f.close()

    #turn those lists into 2D numpy arrays
    lats = np.array(lats)
    lons = np.array(lons)
    griddata = np.array(z)

    #figure out the number of rows and columns we have
    ulat = np.unique(lats)
    ulon = np.unique(lons)
    xdim = ulon[1] - ulon[0]
    ydim = ulat[1] - ulat[0]
    xmin = lons.min()
    xmax = lons.max()
    ymin = lats.min()
    ymax = lats.max()
    ncols = np.int32(np.floor(((xmax+xdim)-xmin)/xdim))
    nrows = np.int32(np.floor(((ymax+ydim)-ymin)/ydim))

    #create a dictionary containing the upper left corner, resolution, and dimensions
    geodict = {}
    geodict['xmin'] = xmin
    geodict['xmax'] = xmax
    geodict['ymin'] = ymin
    geodict['ymax'] = ymax
    geodict['dx'] = xdim
    geodict['dy'] = ydim
    geodict['nx'] = ncols
    geodict['ny'] = nrows

    ulx = self.geodict['xmin']
    uly = self.geodict['ymax']
    xdim = self.geodict['xdim']
    ydim = self.geodict['ydim']
    cols = np.floor((lons-xmin)/xdim)
    rows = np.floor((ymax-lats)/ydim)
    index = np.int32(np.multi_ravel_index((cols,rows),(ncols,nrows)))
    griddata[index] = griddata
    griddata = np.reshape(griddata,(nrows,ncols))

    return (griddata,geodict)

class HazardGrid(Grid2D):
    reqfields = set(['xmin','xmax','ymin','ymax','dx','dy','nx','ny'])
    
    @classmethod
    def load(filename):
        griddata,geodict = read_grid_array(filename)
        gd = GeoDict(geodict)
        return HazardGrid(griddata,gd)

    #This should be a @classmethod in subclasses
    @abc.abstractmethod
    def save(self,filename): #would we ever want to save a subset of the data?
        raise NotImplementedError('Save method not implemented.')
    


        
        
