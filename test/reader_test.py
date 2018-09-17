#!/usr/bin/env python

import os.path
import time
import sys

import numpy as np
import rasterio

from mapio.reader import read, get_file_geodict
from mapio.geodict import GeoDict


def test_read_whole():
    files = {'ESRI Float': 'samplegrid_flt.flt',
             'NetCDF 3': 'samplegrid_cdf.cdf'}
    # where is this script?
    homedir = os.path.dirname(os.path.abspath(__file__))
    # this is an HDF 5 file
    for ftype, fname in files.items():
        datafile = os.path.join(homedir, 'data', fname)
        grid = read(datafile)
        assert grid._geodict.xmin == 5.0
        print('Successful read of %s' % ftype)


def test_read_subset_no_resample():
    # where is this script?
    homedir = os.path.dirname(os.path.abspath(__file__))
    # this is an HDF 5 file
    datafile = os.path.join(homedir, 'data', 'samplegrid_cdf.cdf')
    sdict = {'xmin': 6,
             'xmax': 7,
             'ymin': 5,
             'ymax': 6,
             'nx': 2,
             'ny': 2,
             'dx': 1,
             'dy': 1}
    sampledict = GeoDict(sdict)
    grid = read(datafile, samplegeodict=sampledict)
    tdata = np.array([[11, 12], [16, 17]])
    np.testing.assert_almost_equal(grid._data, tdata)


def test_resample():
    # this should fail
    # where is this script?
    homedir = os.path.dirname(os.path.abspath(__file__))
    # this is an HDF 5 file
    datafile = os.path.join(homedir, 'data', 'samplegrid_cdf.cdf')
    sdict = {'xmin': 6.5,
             'xmax': 7.5,
             'ymin': 5.5,
             'ymax': 6.5,
             'nx': 2,
             'ny': 2,
             'dx': 1,
             'dy': 1}
    sampledict = GeoDict(sdict)
    grid = read(datafile, samplegeodict=sampledict, resample=True)
    tdata = np.array([[9, 10], [14, 15]])
    np.testing.assert_almost_equal(grid._data, tdata)


def test_read_subset_with_resample_and_padding():
    # where is this script?
    homedir = os.path.dirname(os.path.abspath(__file__))
    # this is an HDF 5 file
    datafile = os.path.join(homedir, 'data', 'samplegrid_cdf.cdf')
    sdict = {'xmin': 4.5,
             'xmax': 5.5,
             'ymin': 7.5,
             'ymax': 8.5,
             'nx': 2,
             'ny': 2,
             'dx': 1,
             'dy': 1}
    sampledict = GeoDict(sdict)
    grid = read(datafile, samplegeodict=sampledict,
                resample=True, doPadding=True)
    atest = np.array([[np.nan, np.nan],
                      [np.inf,  3.]])
    np.testing.assert_almost_equal(grid._data, atest)


def test_read_subset_with_padding():
    # where is this script?
    homedir = os.path.dirname(os.path.abspath(__file__))
    # this is an HDF 5 file
    datafile = os.path.join(homedir, 'data', 'samplegrid_cdf.cdf')
    sdict = {'xmin': 4.5,
             'xmax': 5.5,
             'ymin': 7.5,
             'ymax': 8.5,
             'nx': 2,
             'ny': 2,
             'dx': 1,
             'dy': 1}
    sampledict = GeoDict(sdict)
    grid = read(datafile, samplegeodict=sampledict,
                resample=False, doPadding=True)
    assert grid._data.shape == (3, 3)
    assert grid._data[1, 1] == 0


def test_read_meridian():
    # where is this script?
    homedir = os.path.dirname(os.path.abspath(__file__))
    # this is an HDF 5 file
    datafile = os.path.join(homedir, 'data', 'globalgrid_cdf.cdf')
    sdict = {'xmin': 180,
             'xmax': -120,
             'ymin': 0,
             'ymax': 30,
             'nx': 2,
             'ny': 2,
             'dx': 60,
             'dy': 30}
    sampledict = GeoDict(sdict)
    grid = read(datafile, samplegeodict=sampledict)
    assert np.nansum(grid._data) == 50.0


def read_user_file_test(fname, xmin, xmax, ymin, ymax):
    gd = get_file_geodict(fname)
    sample = GeoDict.createDictFromBox(xmin, xmax, ymin, ymax, gd.dx, gd.dy)
    t1 = time.time()
    grid = read(fname, samplegeodict=sample)
    t2 = time.time()
    nrows, ncols = grid._data.shape
    npixels = nrows*ncols
    print('%.2f seconds to read %i pixels using h5py' % (t2-t1, npixels))

    west, east, south, north = (-105.00416666665,
                                -102.98750000804999,
                                34.98750000805,
                                37.00416666665)
    src = rasterio.open(fname, 'r')
    window = src.window(west, south, east, north)
    t1 = time.time()
    data = src.read(window=window)
    t2 = time.time()
    print('%.2f seconds to read %i pixels using rasterio' % (t2-t1, npixels))
    ratio = grid._data.sum()/data.sum()
    print('Ratio of h5py data to rasterio data is %.4f' % ratio)
    src.close()


if __name__ == '__main__':
    test_read_whole()
    test_read_subset_no_resample()
    test_resample()
    test_read_subset_with_padding()
    test_read_subset_with_resample_and_padding()
    test_read_meridian()

    if len(sys.argv) > 1:
        fname = sys.argv[1]
        xmin = float(sys.argv[2])
        xmax = float(sys.argv[3])
        ymin = float(sys.argv[4])
        ymax = float(sys.argv[5])
        read_user_file_test(fname, xmin, xmax, ymin, ymax)
