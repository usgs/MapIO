#!/usr/bin/env python

#python 3 compatibility
from __future__ import print_function

#stdlib imports
import os.path
import sys
from collections import OrderedDict
import warnings

#third party imports
import rasterio
import numpy as np

#hack the path so that I can debug these functions if I need to
homedir = os.path.dirname(os.path.abspath(__file__)) #where is this script?
mapiodir = os.path.abspath(os.path.join(homedir,'..'))
sys.path.insert(0,mapiodir) #put this at the front of the system path, ignoring any installed mapio stuff

from mapio.grid2d import Grid2D
from mapio.gdal import GDALGrid
from mapio.dataset import DataSetException,DataSetWarning
from mapio.geodict import GeoDict

def createSample(M,N):
    data = np.arange(0,M*N).reshape(M,N)
    data = data.astype(np.int32) #arange gives int64 by default, not supported by netcdf3
    xvar = np.arange(0.5,0.5+N,1.0)
    yvar = np.arange(0.5,0.5+M,1.0)
    geodict = {'nrows':N,
               'ncols':N,
               'xmin':0.5,
               'xmax':xvar[-1],
               'ymin':0.5,
               'ymax':yvar[-1],
               'xdim':1.0,
               'ydim':1.0}
    gmtgrid = GDALGrid(data,geodict)
    return gmtgrid

def test_format():
    try:
        for dtype in [np.uint8,np.uint16,np.uint32,np.int8,np.int16,np.int32,np.float32,np.float64]:
            print('Testing saving/loading of data with type %s...' % str(dtype))
            data = np.arange(0,16).reshape(4,4).astype(dtype)
            if dtype in [np.float32,np.float64]:
                data[1,1] = np.nan
            geodict = GeoDict({'xmin':0.5,'xmax':3.5,'ymin':0.5,'ymax':3.5,'xdim':1.0,'ydim':1.0,'nrows':4,'ncols':4})
            gdalgrid = GDALGrid(data,geodict)
            gdalgrid.save('test.bil')
            gdalgrid2 = GDALGrid.load('test.bil')
            np.testing.assert_almost_equal(gdalgrid2.getData(),gdalgrid.getData())
            print('Passed saving/loading of data with type %s...' % str(dtype))

    except Exception as obj:
        print('Failed tests with message: "%s"' % str(obj))
    os.remove('test.bil')
    os.remove('test.hdr')

def test_pad():
    try:
        print('Test padding data with null values...')
        data,geodict = Grid2D._createSampleData(4,4)
        gdalgrid = GDALGrid(data,geodict)
        gdalgrid.save('test.bil')

        newdict = GeoDict({'xmin':-0.5,'xmax':4.5,'ymin':-0.5,'ymax':4.5,'xdim':1.0,'ydim':1.0,'nrows':2,'ncols':2},preserve='dims')
        gdalgrid2 = GDALGrid.load('test.bil',samplegeodict=newdict,doPadding=True)
        output = np.array([[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                           [np.nan,0.0,1.0,2.0,3.0,np.nan],
                           [np.nan,4.0,5.0,6.0,7.0,np.nan],
                           [np.nan,8.0,9.0,10.0,11.0,np.nan],
                           [np.nan,12.0,13.0,14.0,15.0,np.nan],
                           [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]])
        np.testing.assert_almost_equal(gdalgrid2._data,output)
        print('Passed padding data null values.')
    except AssertionError as error:
        print('Failed padding test:\n %s' % error)
    if os.path.isfile('test.bil'):
        os.remove('test.bil')
        os.remove('test.hdr')

def test_subset():
    try:
        print('Testing subsetting of non-square grid...')
        data,geodict = Grid2D._createSampleData(6,4)
        gdalgrid = GDALGrid(data,geodict)
        gdalgrid.save('test.bil')
        newdict = GeoDict({'xmin':1.5,'xmax':2.5,
                           'ymin':1.5,'ymax':3.5,
                           'xdim':1.0,'ydim':1.0,
                           'ncols':2,'nrows':2},
                           preserve='dims')
        gdalgrid3 = GDALGrid.load('test.bil',samplegeodict=newdict)
        output = np.array([[9,10],
                           [13,14],
                           [17,18]])
        np.testing.assert_almost_equal(gdalgrid3._data,output)
        print('Passed subsetting of non-square grid.')
        
    except AssertionError as error:
        print('Failed subset test:\n %s' % error)

    os.remove('test.bil')
    os.remove('test.hdr')

def test_resample():
    try:
        print('Test resampling data without padding...')
        data,geodict = Grid2D._createSampleData(9,7)
        gdalgrid = GDALGrid(data,geodict)
        gdalgrid.save('test.bil')

        bounds = (3.0,4.0,3.0,4.0)
        #newdict = Grid2D.fixGeoDict(bounds,1.0,1.0,-1,-1,preserve='dims')
        newdict = GeoDict({'xmin':3.0,'xmax':4.0,
                           'ymin':3.0,'ymax':4.0,
                           'xdim':1.0,'ydim':1.0,
                           'nrows':2,'ncols':2},preserve='dims')
        gdalgrid3 = GDALGrid.load('test.bil',samplegeodict=newdict,resample=True)
        output = np.array([[34,35],
                           [41,42]])
        np.testing.assert_almost_equal(gdalgrid3._data,output)
        print('Passed resampling data without padding...')
        
        print('Test resampling data with padding...')
        data,geodict = Grid2D._createSampleData(4,4)
        gdalgrid = GDALGrid(data,geodict)
        gdalgrid.save('test.bil')
        newdict = {'xmin':0.0,'xmax':4.0,'ymin':0.0,'ymax':4.0,'xdim':1.0,'ydim':1.0}
        bounds = (0.0,4.0,0.0,4.0)
        xdim,ydim = (1.0,1.0)
        nrows,ncols = (-1,-1)
        newdict = GeoDict({'xmin':0.0,'xmax':4.0,
                          'ymin':0.0,'ymax':4.0,
                          'xdim':xdim,'ydim':ydim,
                          'nrows':2,'ncols':ncols},preserve='dims')
        gdalgrid3 = GDALGrid.load('test.bil',samplegeodict=newdict,resample=True,doPadding=True)
        output = np.array([[np.nan,np.nan,np.nan,np.nan,np.nan],
                           [np.nan,2.5,3.5,4.5,np.nan],
                           [np.nan,6.5,7.5,8.5,np.nan],
                           [np.nan,10.5,11.5,12.5,np.nan],
                           [np.nan,np.nan,np.nan,np.nan,np.nan]])
        np.testing.assert_almost_equal(gdalgrid3._data,output)
        print('Passed resampling data with padding...')
    except AssertionError as error:
        print('Failed resample test:\n %s' % error)

    os.remove('test.bil')
    os.remove('test.hdr')

def test_meridian():
    try:
        print('Testing resampling of global grid where sample crosses 180/-180 meridian...')
        data = np.arange(0,84).astype(np.int32).reshape(7,12)
        geodict = GeoDict({'xmin':-180.0,'xmax':150.0,'ymin':-90.0,'ymax':90.0,'xdim':30,'ydim':30,'nrows':7,'ncols':12})
        gdalgrid = GDALGrid(data,geodict)
        gdalgrid.save('test.bil')

        sampledict = GeoDict({'xmin':105,'xmax':-105,
                              'ymin':-15.0,'ymax':15.0,
                              'xdim':30.0,'ydim':30.0,
                              'nrows':2,'ncols':5},preserve='dims')
        gdalgrid5 = GDALGrid.load('test.bil',samplegeodict=sampledict,resample=True,doPadding=True)

        output = np.array([[ 39.5,40.5,35.5,30.5,31.5,32.5],
                           [ 51.5,52.5,47.5,42.5,43.5,44.5]])
        #output = np.random.rand(2,6) #this will fail assertion test
        np.testing.assert_almost_equal(gdalgrid5._data,output)
        print('Passed resampling of global grid where sample crosses 180/-180 meridian...')

        print('Testing resampling of global grid where sample crosses 180/-180 meridian and first column is duplicated by last...')
        data = np.arange(0,84).astype(np.int32).reshape(7,12)
        data = np.hstack((data,data[:,0].reshape(7,1)))
        geodict = GeoDict({'xmin':-180.0,'xmax':180.0,
                           'ymin':-90.0,'ymax':90.0,
                           'xdim':30,'ydim':30,
                           'nrows':7,'ncols':13})
        gdalgrid = GDALGrid(data,geodict)
        gdalgrid.save('test.bil')

        sampledict = GeoDict({'xmin':105,'xmax':-105,
                              'ymin':-15.0,'ymax':15.0,
                              'xdim':30.0,'ydim':30.0,
                              'nrows':2,'ncols':5},preserve='dims')
        gdalgrid5 = GDALGrid.load('test.bil',samplegeodict=sampledict,resample=True,doPadding=True)

        output = np.array([[ 39.5,40.5,35.5,30.5,31.5,32.5],
                           [ 51.5,52.5,47.5,42.5,43.5,44.5]])
        #output = np.random.rand(2,6) #this will fail assertion test
        np.testing.assert_almost_equal(gdalgrid5._data,output)
        print('Passed resampling of global grid where sample crosses 180/-180 meridian and first column is duplicated by last...')
        
    except AssertionError as error:
        print('Failed meridian test:\n %s' % error)
    os.remove('test.bil')
    os.remove('test.hdr')

    
if __name__ == '__main__':
    homedir = os.path.dirname(os.path.abspath(__file__)) #where is this script?
    mapiodir = os.path.abspath(os.path.join(homedir,'..','mapio'))
    sys.path.append(mapiodir)
    if len(sys.argv) > 1:
        gdalfile = sys.argv[1]
        sampledict = None
        if len(sys.argv) > 2:
            xmin = float(sys.argv[2])
            xmax = float(sys.argv[3])
            ymin = float(sys.argv[4])
            ymax = float(sys.argv[5])
            xdim = float(sys.argv[6])
            ydim = float(sys.argv[7])
            fgeodict,xvar,yvar = GDALGrid.getFileGeoDict(gdalfile)
            sampledict1 = GeoDict({'xmin':xmin,'xmax':xmax,
                                   'ymin':ymin,'ymax':ymax,
                                   'xdim':xdim,'ydim':ydim,
                                   'nrows':2,'ncols':2},preserve='dims')
            sampledict2 = GDALGrid.getBoundsWithin(gdalfile,sampledict1)
            grid = GDALGrid.load(gdalfile,samplegeodict=sampledict2)
    else:
        test_format()
        test_pad()
        test_subset()
        test_resample()
        test_meridian()
        

