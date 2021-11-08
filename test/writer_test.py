#!/usr/bin/env python

import tempfile
import shutil
import os.path
import pathlib

import rasterio
import numpy as np

from mapio.writer import write
from mapio.reader import read
from mapio.geodict import GeoDict
from mapio.grid2d import Grid2D


def test_write():
    data = np.arange(0, 25).reshape(5, 5).astype(np.float32)
    gdict = {
        "xmin": 5.0,
        "xmax": 9.0,
        "ymin": 4.0,
        "ymax": 8.0,
        "dx": 1.0,
        "dy": 1.0,
        "nx": 5,
        "ny": 5,
    }
    gd = GeoDict(gdict)
    grid = Grid2D(data, gd)

    for format_type in ["netcdf", "esri", "hdf", "tiff"]:
        tdir = tempfile.mkdtemp()
        fname = os.path.join(tdir, "tempfile.grd")
        try:
            write(grid, fname, format_type)
            src = rasterio.open(fname, "r")
            tdata = src.read(1)
            np.testing.assert_almost_equal(tdata, data)
        except Exception as e:
            raise e
        finally:
            shutil.rmtree(tdir)


def test_compress():
    data = np.arange(0, 10000).reshape(100, 100).astype(np.float32)
    nx, ny = data.shape
    gdict = {
        "xmin": 5.0,
        "xmax": 9.0,
        "ymin": 4.0,
        "ymax": 8.0,
        "dx": 1.0,
        "dy": 1.0,
        "nx": nx,
        "ny": ny,
    }
    gd = GeoDict(gdict)
    grid = Grid2D(data, gd)

    tdir = tempfile.mkdtemp()
    fname = os.path.join(tdir, "tempfile.tif")
    try:
        write(grid, fname, "tiff", do_compression=False)
        uncompressed_size = pathlib.Path(fname).stat().st_size
        write(grid, fname, "tiff", do_compression=True)
        compressed_size = pathlib.Path(fname).stat().st_size
        assert compressed_size < uncompressed_size
        src = rasterio.open(fname, "r")
        tdata = src.read(1)
        np.testing.assert_almost_equal(tdata, data)
    except Exception as e:
        raise e
    finally:
        shutil.rmtree(tdir)


def test_nan_write():
    data = np.arange(0, 25).reshape(5, 5).astype(np.float32)
    data[0, 0] = np.nan
    gdict = {
        "xmin": 5.0,
        "xmax": 9.0,
        "ymin": 4.0,
        "ymax": 8.0,
        "dx": 1.0,
        "dy": 1.0,
        "nx": 5,
        "ny": 5,
    }
    gd = GeoDict(gdict)
    grid = Grid2D(data, gd)
    for format_type in ["netcdf", "esri", "hdf", "tiff"]:
        tdir = tempfile.mkdtemp()
        fname = os.path.join(tdir, "tempfile.grd")
        try:
            write(grid, fname, format_type)
            src = rasterio.open(fname, "r")
            tdata = src.read(1)
            np.testing.assert_almost_equal(tdata, data)
        except Exception as e:
            raise e
        finally:
            shutil.rmtree(tdir)


def big_test():
    xmin = -180
    xmax = -170
    ymin = 30
    ymax = 40
    dx = 0.0083
    dy = 0.0083
    gd = GeoDict.createDictFromBox(xmin, xmax, ymin, ymax, dx, dy)
    data = np.random.rand(gd.ny, gd.nx)
    grid = Grid2D(data, gd)
    fname = os.path.join(os.path.expanduser("~"), "tempfile.grd")
    write(grid, fname, "hdf")
    print(fname)
    src = rasterio.open(fname, "r")
    # tdata = src.read(1)
    # np.testing.assert_almost_equal(tdata, data)


def test_meridian():
    xmin = 175
    xmax = -175
    ymin = 30
    ymax = 40
    dx = 1
    dy = 1
    nx = int((((360 + xmax) - xmin) / dx) + 1)
    ny = int(((ymax - ymin) / dy) + 1)
    geodict = GeoDict(
        {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "dx": dx,
            "dy": dy,
            "nx": nx,
            "ny": ny,
        }
    )
    data = np.arange(0, nx * ny, dtype=np.int32).reshape(ny, nx)
    grid = Grid2D(data=data, geodict=geodict)
    tdir = tempfile.mkdtemp()
    fname = os.path.join(tdir, "tempfile.grd")
    try:
        write(grid, fname, "netcdf")
        grid2 = read(fname)
        assert grid2._geodict == geodict
        write(grid, fname, "tiff")
        grid3 = read(fname)
        assert grid3._geodict == geodict
    except Exception as e:
        raise e
    finally:
        shutil.rmtree(tdir)


if __name__ == "__main__":
    # big_test()
    test_meridian()
    test_compress()
    test_write()
    test_nan_write()
