#!/usr/bin/env python

#python 3 compatibility
from __future__ import print_function

#stdlib imports
import struct
import os.path
import sys

#third party imports
import numpy as np
from scipy.io import netcdf

#hack the path so that I can debug these functions if I need to
homedir = os.path.dirname(os.path.abspath(__file__)) #where is this script?
mapiodir = os.path.abspath(os.path.join(homedir,'..'))
sys.path.insert(0,mapiodir) #put this at the front of the system path, ignoring any installed mapio stuff

from mapio.grid2d import Grid2D
from mapio.gridbase import Grid
from mapio.dataset import DataSetException
from mapio.gmt import GMTGrid,indexArray,sub2ind,NETCDF_TYPES,INVERSE_NETCDF_TYPES
from mapio.geodict import GeoDict
import h5py

def createSampleXRange(M,N,filename,bounds=None,dx=None,dy=None):
    if dx is None:
        dx = 1.0
    if dy is None:
        dy = 1.0
    if bounds is None:
        xmin = 0.5
        xmax = xmin + (N-1)*dx
        ymin = 0.5
        ymax = ymin + (M-1)*dy
    else:
        xmin,xmax,ymin,ymax = bounds
    data = np.arange(0,M*N).reshape(M,N).astype(np.int32)
    cdf = netcdf.netcdf_file(filename,'w')
    cdf.createDimension('side',2)
    cdf.createDimension('xysize',M*N)
    dim = cdf.createVariable('dimension','i',('side',))
    dim[:] = np.array([N,M])
    spacing = cdf.createVariable('spacing','i',('side',))
    spacing[:] = np.array([dx,dy])
    zrange = cdf.createVariable('z_range',INVERSE_NETCDF_TYPES[str(data.dtype)],('side',))
    zrange[:] = np.array([data.min(),data.max()])
    x_range = cdf.createVariable('x_range','d',('side',))
    x_range[:] = np.array([xmin,xmax])
    y_range = cdf.createVariable('y_range','d',('side',))
    y_range[:] = np.array([ymin,ymax])
    z = cdf.createVariable('z',INVERSE_NETCDF_TYPES[str(data.dtype)],('xysize',))
    z[:] = data.flatten()
    cdf.close()
    return data

def createSampleGrid(M,N):
    """Used for internal testing - create an NxN grid with lower left corner at 0.5,0.5, dx/dy = 1.0.
    :param M:
       Number of rows in output grid
    :param N:
       Number of columns in output grid
    :returns:
       GMTGrid object where data values are an NxN array of values from 0 to N-squared minus 1, and geodict
       lower left corner is at 0.5/0.5 and cell dimensions are 1.0.
    """
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
    gd = GeoDict(geodict)
    gmtgrid = GMTGrid(data,gd)
    return gmtgrid

def test_save():
    try:
        print('Testing saving and loading to/from NetCDF3...')
        #make a sample data set
        gmtgrid = createSampleGrid(4,4)

        #save it as netcdf3
        gmtgrid.save('test.grd',format='netcdf')
        gmtgrid2 = GMTGrid.load('test.grd')
        np.testing.assert_almost_equal(gmtgrid._data,gmtgrid2._data)
        print('Passed saving and loading to/from NetCDF3.')

        print('Testing saving and loading to/from NetCDF4 (HDF)...')
        #save it as HDF
        gmtgrid.save('test.grd',format='hdf')
        gmtgrid3 = GMTGrid.load('test.grd')
        np.testing.assert_almost_equal(gmtgrid._data,gmtgrid3._data)
        print('Passed saving and loading to/from NetCDF4 (HDF)...')

        print('Testing saving and loading to/from GMT native)...')
        gmtgrid.save('test.grd',format='native')
        gmtgrid4 = GMTGrid.load('test.grd')
        np.testing.assert_almost_equal(gmtgrid._data,gmtgrid4._data)
        print('Passed saving and loading to/from GMT native...')
    except AssertionError as error:
        print('Failed padding test:\n %s' % error)
    os.remove('test.grd')

def test_pad():
    try:
        for fmt in ['netcdf','hdf','native']:
            print('Test padding data with null values (format %s)...' % fmt)
            gmtgrid = createSampleGrid(4,4)
            gmtgrid.save('test.grd',format=fmt)

            newdict = GeoDict({'xmin':-0.5,'xmax':4.5,
                               'ymin':-0.5,'ymax':4.5,
                               'dx':1.0,'dy':1.0,
                               'ny':6,'nx':6})
            gmtgrid2 = GMTGrid.load('test.grd',samplegeodict=newdict,doPadding=True)
            output = np.array([[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                               [np.nan,0.0,1.0,2.0,3.0,np.nan],
                               [np.nan,4.0,5.0,6.0,7.0,np.nan],
                               [np.nan,8.0,9.0,10.0,11.0,np.nan],
                               [np.nan,12.0,13.0,14.0,15.0,np.nan],
                               [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]])
            np.testing.assert_almost_equal(gmtgrid2._data,output)
            print('Passed padding data with format %s.' % fmt)
    except AssertionError as error:
        print('Failed padding test:\n %s' % error)
    os.remove('test.grd')

def test_subset():
    try:
        for fmt in ['netcdf','hdf','native']:
            print('Testing subsetting of non-square grid (format %s)...' % fmt)
            data = np.arange(0,24).reshape(6,4).astype(np.int32)
            geodict = GeoDict({'xmin':0.5,'xmax':3.5,
                               'ymin':0.5,'ymax':5.5,
                               'dx':1.0,'dy':1.0,
                               'ny':6,'nx':4})
            gmtgrid = GMTGrid(data,geodict)
            gmtgrid.save('test.grd',format=fmt)
            newdict = GeoDict({'xmin':1.5,'xmax':2.5,
                               'ymin':1.5,'ymax':3.5,
                               'dx':1.0,'dy':1.0,
                               'ny':3,'nx':2})
            gmtgrid3 = GMTGrid.load('test.grd',samplegeodict=newdict)
            output = np.array([[9,10],
                               [13,14],
                               [17,18]])
            np.testing.assert_almost_equal(gmtgrid3._data,output)
            print('Passed subsetting of non-square grid (format %s)...' % fmt)
        
    except AssertionError as error:
        print('Failed subset test:\n %s' % error)

    os.remove('test.grd')

def test_resample():
    try:
        for fmt in ['netcdf','hdf','native']:
            print('Test resampling data without padding (format %s)...' % fmt)
            data = np.arange(0,63).astype(np.int32).reshape(9,7)
            geodict = GeoDict({'xmin':0.5,'xmax':6.5,
                               'ymin':0.5,'ymax':8.5,
                               'dx':1.0,'dy':1.0,
                               'ny':9,'nx':7})
            gmtgrid = GMTGrid(data,geodict)
            gmtgrid.save('test.grd',format=fmt)

            bounds = (3.0,4.0,3.0,4.0)
            dx,dy = (1.0,1.0)
            ny,nx = (-1,-1)
            newdict = GeoDict({'xmin':3.0,'xmax':4.0,
                               'ymin':3.0,'ymax':4.0,
                               'dx':1.0,'dy':1.0,
                               'ny':2,'nx':2})
            gmtgrid3 = GMTGrid.load('test.grd',samplegeodict=newdict,resample=True)
            output = np.array([[34,35],
                               [41,42]])
            np.testing.assert_almost_equal(gmtgrid3._data,output)
            print('Passed resampling data without padding (format %s)...' % fmt)

            print('Test resampling data with padding (format %s)...' % fmt)
            gmtgrid = createSampleGrid(4,4)
            gmtgrid.save('test.grd',format=fmt)
            newdict = {'xmin':0.0,'xmax':4.0,'ymin':0.0,'ymax':4.0,'dx':1.0,'dy':1.0}
            bounds = (0.0,4.0,0.0,4.0)
            dx,dy = (1.0,1.0)
            ny,nx = (-1,-1)
            newdict = GeoDict({'xmin':0.0,'xmax':4.0,
                               'ymin':0.0,'ymax':4.0,
                               'dx':1.0,'dy':1.0,
                               'ny':5,'nx':5})
            gmtgrid3 = GMTGrid.load('test.grd',samplegeodict=newdict,resample=True,doPadding=True)
            output = np.array([[np.nan,np.nan,np.nan,np.nan,np.nan],
                               [np.nan,2.5,3.5,4.5,np.nan],
                               [np.nan,6.5,7.5,8.5,np.nan],
                               [np.nan,10.5,11.5,12.5,np.nan],
                               [np.nan,np.nan,np.nan,np.nan,np.nan]])
            np.testing.assert_almost_equal(gmtgrid3._data,output)
            print('Passed resampling data with padding (format %s)...' % fmt)
    except AssertionError as error:
        print('Failed resample test:\n %s' % error)

    os.remove('test.grd')
    
def test_meridian():
    try:
        for fmt in ['netcdf','hdf','native']:
            print('Testing resampling of global grid where sample crosses 180/-180 meridian (format %s)...' % fmt)
            data = np.arange(0,84).astype(np.int32).reshape(7,12)
            geodict = GeoDict({'xmin':-180.0,'xmax':150.0,
                               'ymin':-90.0,'ymax':90.0,
                               'dx':30,'dy':30,
                               'ny':7,'nx':12})
            gmtgrid = GMTGrid(data,geodict)
            gmtgrid.save('test.grd',format=fmt)

            sampledict = GeoDict({'xmin':105,'xmax':-105,
                                  'ymin':-15.0,'ymax':15.0,
                                  'dx':30.0,'dy':30.0,
                                  'ny':2,'nx':6})
            gmtgrid5 = GMTGrid.load('test.grd',samplegeodict=sampledict,resample=True,doPadding=True)

            output = np.array([[ 39.5,40.5,35.5,30.5,31.5,32.5],
                               [ 51.5,52.5,47.5,42.5,43.5,44.5]])
            #output = np.random.rand(2,6) #this will fail assertion test
            np.testing.assert_almost_equal(gmtgrid5._data,output)
            print('Passed resampling of global grid where sample crosses 180/-180 meridian (format %s)...' % fmt)

            print('Testing resampling of global grid where sample crosses 180/-180 meridian and first column is duplicated by last (format %s)...' % fmt)
            data = np.arange(0,84).astype(np.int32).reshape(7,12)
            data = np.hstack((data,data[:,0].reshape(7,1)))
            geodict = GeoDict({'xmin':-180.0,'xmax':180.0,
                               'ymin':-90.0,'ymax':90.0,
                               'dx':30,'dy':30,
                               'ny':7,'nx':13})
            gmtgrid = GMTGrid(data,geodict)
            gmtgrid.save('test.grd')

            sampledict = GeoDict({'xmin':105,'xmax':-105,
                                  'ymin':-15.0,'ymax':15.0,
                                  'dx':30.0,'dy':30.0,
                                  'ny':2,'nx':6})
            gmtgrid5 = GMTGrid.load('test.grd',samplegeodict=sampledict,resample=True,doPadding=True)

            output = np.array([[ 39.5,40.5,35.5,30.5,31.5,32.5],
                               [ 51.5,52.5,47.5,42.5,43.5,44.5]])
            #output = np.random.rand(2,6) #this will fail assertion test
            np.testing.assert_almost_equal(gmtgrid5._data,output)
            print('Passed resampling of global grid where sample crosses 180/-180 meridian and first column is duplicated by last (format %s)...' % fmt)
        
    except AssertionError as error:
        print('Failed meridian test:\n %s' % error)
    os.remove('test.grd')

def test_index():
    data = np.arange(0,42).reshape(6,7)
    d2 = data.flatten()
    shp = data.shape
    res1 = data[1:3,1:3]
    res2 = indexArray(d2,shp,1,3,1,3)
    np.testing.assert_almost_equal(res1,res2)

def test_xrange():
    #there is a type of GMT netcdf file where the data is in scanline order
    #we don't care enough to support this in the save() method, but we do need a test for it.  Sigh.
    try:
        print('Testing loading whole x_range style grid...')
        data = createSampleXRange(6,4,'test.grd')
        gmtgrid = GMTGrid.load('test.grd')
        np.testing.assert_almost_equal(data,gmtgrid.getData())
        print('Passed loading whole x_range style grid...')

        print('Testing loading partial x_range style grid...')
        #test with subsetting
        newdict = GeoDict({'xmin':1.5,'xmax':2.5,
                           'ymin':1.5,'ymax':3.5,
                           'dx':1.0,'dy':1.0,
                           'ny':3,'nx':2})
        gmtgrid3 = GMTGrid.load('test.grd',samplegeodict=newdict)
        output = np.array([[9,10],
                           [13,14],
                           [17,18]])
        np.testing.assert_almost_equal(gmtgrid3._data,output)
        print('Passed loading partial x_range style grid...')

        print('Testing x_range style grid where we cross meridian...')
        data = createSampleXRange(7,12,'test.grd',(-180.,150.,-90.,90.),dx=30.,dy=30.)
        sampledict = GeoDict({'xmin':105,'xmax':-105,
                              'ymin':-15.0,'ymax':15.0,
                              'dx':30.0,'dy':30.0,
                              'ny':2,'nx':6})
        gmtgrid5 = GMTGrid.load('test.grd',samplegeodict=sampledict,resample=True,doPadding=True)
        print('Testing x_range style grid where we cross meridian...')
        
    except AssertionError as error:
        print('Failed an xrange test:\n %s' % error)
    os.remove('test.grd')    

    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        gmtfile = sys.argv[1]
        sampledict = None
        if len(sys.argv) == 6:
            xmin = float(sys.argv[2])
            xmax = float(sys.argv[3])
            ymin = float(sys.argv[4])
            ymax = float(sys.argv[5])
            sampledict = {'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax}
            grid = GMTGrid.load(gmtfile,samplegeodict=sampledict)
    else:
        test_index()
        test_save()
        test_resample()
        test_meridian()
        test_pad()
        test_subset()
        test_xrange()
        
        
