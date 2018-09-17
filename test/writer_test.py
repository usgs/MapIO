#!/usr/bin/env python

import tempfile
import shutil
import os.path

import rasterio
import numpy as np

from mapio.writer import write
from mapio.geodict import GeoDict
from mapio.grid2d import Grid2D


def test_write():
    data = np.arange(0, 25).reshape(5, 5).astype(np.float32)
    gdict = {'xmin': 5.0,
             'xmax': 9.0,
             'ymin': 4.0,
             'ymax': 8.0,
             'dx': 1.0,
             'dy': 1.0,
             'nx': 5,
             'ny': 5}
    gd = GeoDict(gdict)
    grid = Grid2D(data, gd)

    for format_type in ['netcdf', 'esri', 'hdf']:
        tdir = tempfile.mkdtemp()
        fname = os.path.join(tdir, 'tempfile.grd')
        try:
            write(grid, fname, format_type)
            src = rasterio.open(fname, 'r')
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
    fname = os.path.join(os.path.expanduser('~'), 'tempfile.grd')
    write(grid, fname, 'hdf')
    print(fname)
    src = rasterio.open(fname, 'r')
    # tdata = src.read(1)
    # np.testing.assert_almost_equal(tdata, data)


if __name__ == '__main__':
    # big_test()
    test_write()
