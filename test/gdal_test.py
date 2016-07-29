#!/usr/bin/env python

#python 3 compatibility
from __future__ import print_function

#stdlib imports
import os.path
import sys
from collections import OrderedDict
import warnings
import tempfile
import shutil

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
    geodict = {'ny':N,
               'nx':N,
               'xmin':0.5,
               'xmax':xvar[-1],
               'ymin':0.5,
               'ymax':yvar[-1],
               'dx':1.0,
               'dy':1.0}
    gmtgrid = GDALGrid(data,geodict)
    return gmtgrid

def test_format():
    tdir = tempfile.mkdtemp()
    testfile = os.path.join(tdir,'test.bil')
    testhdr = os.path.join(tdir,'test.hdr')
    try:
        for dtype in [np.uint8,np.uint16,np.uint32,np.int8,np.int16,np.int32,np.float32,np.float64]:
            print('Testing saving/loading of data with type %s...' % str(dtype))
            data = np.arange(0,16).reshape(4,4).astype(dtype)
            if dtype in [np.float32,np.float64]:
                data[1,1] = np.nan
            geodict = GeoDict({'xmin':0.5,'xmax':3.5,'ymin':0.5,'ymax':3.5,'dx':1.0,'dy':1.0,'ny':4,'nx':4})
            gdalgrid = GDALGrid(data,geodict)
            gdalgrid.save(testfile)
            gdalgrid2 = GDALGrid.load(testfile)
            np.testing.assert_almost_equal(gdalgrid2.getData(),gdalgrid.getData())
            print('Passed saving/loading of data with type %s...' % str(dtype))

    except DataSetException as obj:
        print('Failure other than AssertionError: "%s"' % str(obj))
    finally:
        if os.path.isdir(tdir):
            shutil.rmtree(tdir)

def test_pad():
    tdir = tempfile.mkdtemp()
    testfile = os.path.join(tdir,'test.bil')
    testhdr = os.path.join(tdir,'test.hdr')
    try:
        print('Test padding data with null values...')
        data,geodict = Grid2D._createSampleData(4,4)
        gdalgrid = GDALGrid(data,geodict)
        gdalgrid.save(testfile)

        newdict = GeoDict({'xmin':-0.5,'xmax':4.5,'ymin':-0.5,'ymax':4.5,'dx':1.0,'dy':1.0,'ny':6,'nx':6})
        gdalgrid2 = GDALGrid.load(testfile,samplegeodict=newdict,doPadding=True)
        output = np.array([[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                           [np.nan,0.0,1.0,2.0,3.0,np.nan],
                           [np.nan,4.0,5.0,6.0,7.0,np.nan],
                           [np.nan,8.0,9.0,10.0,11.0,np.nan],
                           [np.nan,12.0,13.0,14.0,15.0,np.nan],
                           [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]])
        np.testing.assert_almost_equal(gdalgrid2._data,output)
        print('Passed padding data null values.')
    except DataSetException as error:
        print('Failure other than AssertionError: "%s"' % str(error))
    finally:
        if os.path.isdir(tdir):
            shutil.rmtree(tdir)

def test_subset():
    tdir = tempfile.mkdtemp()
    testfile = os.path.join(tdir,'test.bil')
    testhdr = os.path.join(tdir,'test.hdr')
    try:
        print('Testing subsetting of non-square grid...')
        data,geodict = Grid2D._createSampleData(6,4)
        gdalgrid = GDALGrid(data,geodict)
        gdalgrid.save(testfile)
        newdict = GeoDict({'xmin':1.5,'xmax':2.5,
                           'ymin':1.5,'ymax':3.5,
                           'dx':1.0,'dy':1.0,
                           'nx':2,'ny':3})
        gdalgrid3 = GDALGrid.load(testfile,samplegeodict=newdict)
        output = np.array([[9,10],
                           [13,14],
                           [17,18]])
        np.testing.assert_almost_equal(gdalgrid3._data,output)
        print('Passed subsetting of non-square grid.')
        
    except DataSetException as error:
        print('Failure other than AssertionError: "%s"' % str(error))
    finally:
        if os.path.isdir(tdir):
            shutil.rmtree(tdir)

def test_resample():
    tdir = tempfile.mkdtemp()
    testfile = os.path.join(tdir,'test.bil')
    testhdr = os.path.join(tdir,'test.hdr')
    try:
        print('Test resampling data without padding...')
        data,geodict = Grid2D._createSampleData(9,7)
        gdalgrid = GDALGrid(data,geodict)
        gdalgrid.save(testfile)

        bounds = (3.0,4.0,3.0,4.0)
        newdict = GeoDict({'xmin':3.0,'xmax':4.0,
                           'ymin':3.0,'ymax':4.0,
                           'dx':1.0,'dy':1.0,
                           'ny':2,'nx':2})
        gdalgrid3 = GDALGrid.load(testfile,samplegeodict=newdict,resample=True)
        output = np.array([[34,35],
                           [41,42]])
        np.testing.assert_almost_equal(gdalgrid3._data,output)
        print('Passed resampling data without padding...')
        
        print('Test resampling data with padding...')
        data,geodict = Grid2D._createSampleData(4,4)
        gdalgrid = GDALGrid(data,geodict)
        gdalgrid.save(testfile)
        newdict = {'xmin':0.0,'xmax':4.0,'ymin':0.0,'ymax':4.0,'dx':1.0,'dy':1.0}
        bounds = (0.0,4.0,0.0,4.0)
        dx,dy = (1.0,1.0)
        ny,nx = (-1,-1)
        newdict = GeoDict({'xmin':0.0,'xmax':4.0,
                          'ymin':0.0,'ymax':4.0,
                          'dx':dx,'dy':dy,
                          'ny':5,'nx':5})
        gdalgrid3 = GDALGrid.load(testfile,samplegeodict=newdict,resample=True,doPadding=True)
        output = np.array([[np.nan,np.nan,np.nan,np.nan,np.nan],
                           [np.nan,2.5,3.5,4.5,np.nan],
                           [np.nan,6.5,7.5,8.5,np.nan],
                           [np.nan,10.5,11.5,12.5,np.nan],
                           [np.nan,np.nan,np.nan,np.nan,np.nan]])
        np.testing.assert_almost_equal(gdalgrid3._data,output)
        print('Passed resampling data with padding...')
    except DataSetException as error:
        print('Failure other than AssertionError: "%s"' % str(error))
    finally:
        if os.path.isdir(tdir):
            shutil.rmtree(tdir)

def test_meridian():
    tdir = tempfile.mkdtemp()
    testfile = os.path.join(tdir,'test.bil')
    testhdr = os.path.join(tdir,'test.hdr')
    try:
        print('Testing resampling of global grid where sample crosses 180/-180 meridian...')
        data = np.arange(0,84).astype(np.int32).reshape(7,12)
        geodict = GeoDict({'xmin':-180.0,'xmax':150.0,'ymin':-90.0,'ymax':90.0,'dx':30,'dy':30,'ny':7,'nx':12})
        gdalgrid = GDALGrid(data,geodict)
        gdalgrid.save(testfile)

        sampledict = GeoDict({'xmin':105,'xmax':-105,
                              'ymin':-15.0,'ymax':15.0,
                              'dx':30.0,'dy':30.0,
                              'ny':2,'nx':6})
        gdalgrid5 = GDALGrid.load(testfile,samplegeodict=sampledict,resample=True,doPadding=True)

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
                           'dx':30,'dy':30,
                           'ny':7,'nx':13})
        gdalgrid = GDALGrid(data,geodict)
        gdalgrid.save(testfile)

        sampledict = GeoDict({'xmin':105,'xmax':-105,
                              'ymin':-15.0,'ymax':15.0,
                              'dx':30.0,'dy':30.0,
                              'ny':2,'nx':6})
        gdalgrid5 = GDALGrid.load(testfile,samplegeodict=sampledict,resample=True,doPadding=True)

        output = np.array([[ 39.5,40.5,35.5,30.5,31.5,32.5],
                           [ 51.5,52.5,47.5,42.5,43.5,44.5]])
        #output = np.random.rand(2,6) #this will fail assertion test
        np.testing.assert_almost_equal(gdalgrid5._data,output)
        print('Passed resampling of global grid where sample crosses 180/-180 meridian and first column is duplicated by last...')
        
    except DataSetException as error:
        print('Failure other than AssertionError: "%s"' % str(error))
    finally:
        if os.path.isdir(tdir):
            shutil.rmtree(tdir)

    
if __name__ == '__main__':
    test_format()
    test_pad()
    test_subset()
    test_resample()
    test_meridian()
        

