#!/usr/bin/env python

# python 3 compatibility
from __future__ import print_function

# stdlib imports
import os.path
import sys
from collections import OrderedDict
import warnings
import tempfile
import shutil

# third party imports
import rasterio
import numpy as np

# hack the path so that I can debug these functions if I need to
homedir = os.path.dirname(os.path.abspath(__file__))  # where is this script?
mapiodir = os.path.abspath(os.path.join(homedir, ".."))
sys.path.insert(
    0, mapiodir
)  # put this at the front of the system path, ignoring any installed mapio stuff

from mapio.grid2d import Grid2D
from mapio.gdal import GDALGrid
from mapio.gmt import GMTGrid
from mapio.dataset import DataSetException, DataSetWarning
from mapio.geodict import GeoDict

FORMATS = {GDALGrid: ["EHdr"], GMTGrid: ["netcdf", "hdf", "native"]}


def test_simple_subset():
    gridclasses = [GDALGrid, GMTGrid]
    for gridclass in gridclasses:
        for fileformat in FORMATS[gridclass]:
            tdir = None
            try:
                geodict = GeoDict(
                    {
                        "xmin": 0,
                        "xmax": 4,
                        "ymin": 0,
                        "ymax": 4,
                        "dx": 1,
                        "dy": 1,
                        "nx": 5,
                        "ny": 5,
                    }
                )
                data = np.arange(1, 26, dtype=np.float32).reshape((5, 5))
                tdir = tempfile.mkdtemp()
                testfile = os.path.join(tdir, "test.bil")
                testhdr = os.path.join(tdir, "test.hdr")
                srcgrid = gridclass(data, geodict)

                srcgrid.save(testfile, format=fileformat)
                sampledict = GeoDict(
                    {
                        "xmin": 1,
                        "xmax": 3,
                        "ymin": 1,
                        "ymax": 3,
                        "dx": 1,
                        "dy": 1,
                        "nx": 3,
                        "ny": 3,
                    }
                )
                testdata = np.array(
                    [[7, 8, 9], [12, 13, 14], [17, 18, 19]], dtype=np.float32
                )
                testdict = GeoDict(
                    {
                        "xmin": 1,
                        "xmax": 3,
                        "ymin": 1,
                        "ymax": 3,
                        "dx": 1,
                        "dy": 1,
                        "nx": 3,
                        "ny": 3,
                    }
                )
                samplegrid = gridclass.load(testfile, sampledict)
                np.testing.assert_almost_equal(samplegrid.getData(), testdata)
                assert samplegrid.getGeoDict() == testdict
            except Exception as e:
                raise (e)
            finally:
                if os.path.isdir(tdir):
                    shutil.rmtree(tdir)


def test_simple_pad():
    gridclasses = [GDALGrid, GMTGrid]
    for gridclass in gridclasses:
        for fileformat in FORMATS[gridclass]:
            tdir = None
            try:
                geodict = GeoDict(
                    {
                        "xmin": 0,
                        "xmax": 4,
                        "ymin": 0,
                        "ymax": 4,
                        "dx": 1,
                        "dy": 1,
                        "nx": 5,
                        "ny": 5,
                    }
                )
                data = np.arange(1, 26, dtype=np.float32).reshape((5, 5))
                tdir = tempfile.mkdtemp()
                testfile = os.path.join(tdir, "test.bil")
                testhdr = os.path.join(tdir, "test.hdr")
                srcgrid = gridclass(data, geodict)
                srcgrid.save(testfile, format=fileformat)
                sampledict = GeoDict(
                    {
                        "xmin": -1,
                        "xmax": 1,
                        "ymin": 1,
                        "ymax": 3,
                        "dx": 1,
                        "dy": 1,
                        "nx": 3,
                        "ny": 3,
                    }
                )
                testdata = np.array(
                    [[np.nan, 6, 7], [np.nan, 11, 12], [np.nan, 16, 17]],
                    dtype=np.float32,
                )
                testdict = GeoDict(
                    {
                        "xmin": -1,
                        "xmax": 1,
                        "ymin": 1,
                        "ymax": 3,
                        "dx": 1,
                        "dy": 1,
                        "nx": 3,
                        "ny": 3,
                    }
                )
                samplegrid = gridclass.load(testfile, sampledict, doPadding=True)
                np.testing.assert_almost_equal(samplegrid.getData(), testdata)
                assert samplegrid.getGeoDict() == testdict
            except Exception as e:
                raise (e)
            finally:
                if os.path.isdir(tdir):
                    shutil.rmtree(tdir)


def block_test_simple_meridian():
    gridclasses = [GDALGrid, GMTGrid]
    for gridclass in gridclasses:
        for fileformat in FORMATS[gridclass]:
            tdir = None
            try:
                geodict = GeoDict(
                    {
                        "xmin": -180,
                        "xmax": 120,
                        "ymin": -90,
                        "ymax": 90,
                        "dx": 60,
                        "dy": 45,
                        "nx": 6,
                        "ny": 5,
                    }
                )
                data = np.arange(1, 31, dtype=np.float32).reshape((5, 6))
                tdir = tempfile.mkdtemp()
                testfile = os.path.join(tdir, "test.bil")
                testhdr = os.path.join(tdir, "test.hdr")
                srcgrid = gridclass(data, geodict)
                srcgrid.save(testfile, format=fileformat)
                sampledict = GeoDict(
                    {
                        "xmin": 60,
                        "xmax": -120,
                        "ymin": 0,
                        "ymax": 45,
                        "dx": 60,
                        "dy": 45,
                        "nx": 4,
                        "ny": 2,
                    }
                )
                testdata = np.array(
                    [
                        [11, 12, 7, 8],
                        [17, 18, 13, 14],
                    ],
                    dtype=np.float32,
                )
                testdict = GeoDict(
                    {
                        "xmin": 60,
                        "xmax": -120,
                        "ymin": 0,
                        "ymax": 45,
                        "dx": 60,
                        "dy": 45,
                        "nx": 4,
                        "ny": 2,
                    }
                )
                samplegrid = gridclass.load(testfile, sampledict)
                np.testing.assert_almost_equal(samplegrid.getData(), testdata)
                assert samplegrid.getGeoDict() == testdict
            except Exception as e:
                raise (e)
            finally:
                if os.path.isdir(tdir):
                    shutil.rmtree(tdir)


def test_simple_interp():
    gridclasses = [GDALGrid, GMTGrid]
    for gridclass in gridclasses:
        for fileformat in FORMATS[gridclass]:
            tdir = None
            try:
                geodict = GeoDict(
                    {
                        "xmin": -180,
                        "xmax": 120,
                        "ymin": -90,
                        "ymax": 90,
                        "dx": 60,
                        "dy": 45,
                        "nx": 6,
                        "ny": 5,
                    }
                )
                data = np.arange(1, 31, dtype=np.float32).reshape((5, 6))
                tdir = tempfile.mkdtemp()
                testfile = os.path.join(tdir, "test.bil")
                testhdr = os.path.join(tdir, "test.hdr")
                srcgrid = gridclass(data, geodict)
                srcgrid.save(testfile, format=fileformat)
                sampledict = GeoDict(
                    {
                        "xmin": -90,
                        "xmax": 30,
                        "ymin": -22.5,
                        "ymax": 22.5,
                        "dx": 60,
                        "dy": 45,
                        "nx": 3,
                        "ny": 2,
                    }
                )
                testdata = np.array(
                    [
                        [11.5, 12.5, 13.5],
                        [17.5, 18.5, 19.5],
                    ],
                    dtype=np.float32,
                )
                testdict = GeoDict(
                    {
                        "xmin": -90,
                        "xmax": 30,
                        "ymin": -22.5,
                        "ymax": 22.5,
                        "dx": 60,
                        "dy": 45,
                        "nx": 3,
                        "ny": 2,
                    }
                )
                samplegrid = gridclass.load(testfile, sampledict, resample=True)
                np.testing.assert_almost_equal(samplegrid.getData(), testdata)
                assert samplegrid.getGeoDict() == testdict
            except Exception as e:
                raise (e)
            finally:
                if os.path.isdir(tdir):
                    shutil.rmtree(tdir)


def block_test_meridian_interp():
    gridclasses = [GDALGrid, GMTGrid]
    for gridclass in gridclasses:
        for fileformat in FORMATS[gridclass]:
            tdir = None
            try:
                geodict = GeoDict(
                    {
                        "xmin": -180,
                        "xmax": 120,
                        "ymin": -90,
                        "ymax": 90,
                        "dx": 60,
                        "dy": 45,
                        "nx": 6,
                        "ny": 5,
                    }
                )
                data = np.arange(1, 31, dtype=np.float32).reshape((5, 6))
                tdir = tempfile.mkdtemp()
                testfile = os.path.join(tdir, "test.bil")
                testhdr = os.path.join(tdir, "test.hdr")
                srcgrid = gridclass(data, geodict)
                srcgrid.save(testfile, format=fileformat)
                sampledict = GeoDict(
                    {
                        "xmin": 90,
                        "xmax": -150,
                        "ymin": -22.5,
                        "ymax": 22.5,
                        "dx": 60,
                        "dy": 45,
                        "nx": 3,
                        "ny": 2,
                    }
                )
                testdata = np.array(
                    [
                        [14.5, 12.5, 10.5],
                        [20.5, 18.5, 16.5],
                    ],
                    dtype=np.float32,
                )
                testdict = GeoDict(
                    {
                        "xmin": 90,
                        "xmax": -150,
                        "ymin": -22.5,
                        "ymax": 22.5,
                        "dx": 60,
                        "dy": 45,
                        "nx": 3,
                        "ny": 2,
                    }
                )
                samplegrid = gridclass.load(testfile, sampledict, resample=True)
                np.testing.assert_almost_equal(samplegrid.getData(), testdata)
                assert samplegrid.getGeoDict() == testdict
            except Exception as e:
                raise (e)
            finally:
                if os.path.isdir(tdir):
                    shutil.rmtree(tdir)


# def test_360():
#     gridclasses = [GDALGrid,GMTGrid]
#     for gridclass in gridclasses:
#         for fileformat in FORMATS[gridclass]:
#             tdir = None
#             try:
#                 geodict = GeoDict({'xmin':-180,
#                                    'xmax':120,
#                                    'ymin':-90,
#                                    'ymax':90,
#                                    'dx':60,
#                                    'dy':45,
#                                    'nx':6,
#                                    'ny':5})
#                 data = np.arange(1,31,dtype=np.float32).reshape((5,6))
#                 tdir = tempfile.mkdtemp()
#                 testfile = os.path.join(tdir,'test.bil')
#                 testhdr = os.path.join(tdir,'test.hdr')
#                 srcgrid = gridclass(data,geodict)
#                 srcgrid.save(testfile,format=fileformat)
#                 sampledict = GeoDict({'xmin':-90,
#                                       'xmax':30,
#                                       'ymin':-22.5,
#                                       'ymax':22.5,
#                                       'dx':60,
#                                       'dy':45,
#                                       'nx':3,
#                                       'ny':2})
#                 testdata = np.array([[11.5,12.5,13.5],
#                                      [17.5,18.5,19.5],
#                                      ],dtype=np.float32)
#                 testdict = GeoDict({'xmin':-90,
#                                     'xmax':30,
#                                     'ymin':-22.5,
#                                     'ymax':22.5,
#                                     'dx':60,
#                                     'dy':45,
#                                     'nx':3,
#                                     'ny':2})
#                 samplegrid = gridclass.load(testfile,sampledict,resample=True)
#                 np.testing.assert_almost_equal(samplegrid.getData(),testdata)
#                 assert samplegrid.getGeoDict() == testdict
#             except Exception as e:
#                 raise(e)
#             finally:
#                 if os.path.isdir(tdir):
#                     shutil.rmtree(tdir)

if __name__ == "__main__":
    test_simple_interp()
    test_simple_subset()
    # test_simple_meridian()
    # test_meridian_interp()
    test_simple_pad()
