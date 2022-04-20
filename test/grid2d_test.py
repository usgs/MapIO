#!/usr/bin/env python

# python 3 compatibility
from __future__ import print_function
import os.path
import sys
import shutil
import time

# stdlib imports
import abc
import textwrap
import glob
import os
import tempfile

# hack the path so that I can debug these functions if I need to
homedir = os.path.dirname(os.path.abspath(__file__))  # where is this script?
mapiodir = os.path.abspath(os.path.join(homedir, ".."))
# put this at the front of the system path, ignoring any installed mapio stuff
sys.path.insert(0, mapiodir)


# third party imports
from mapio.gridbase import Grid
from mapio.grid2d import Grid2D
from mapio.gdal import GDALGrid, get_affine
from mapio.dataset import DataSetException
from mapio.geodict import GeoDict
import numpy as np
from scipy import interpolate
import shapely
from affine import Affine
from rasterio import features
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.crs import CRS
import rasterio
from shapely.geometry import MultiPoint, Polygon, mapping
import pyproj


def test_subdivide():
    print("Testing subdivide method - aligned grids...")
    data = np.arange(0, 4).reshape((2, 2))
    geodict = GeoDict(
        {
            "xmin": 0.0,
            "xmax": 1.0,
            "ymin": 0.0,
            "ymax": 1.0,
            "dx": 1.0,
            "dy": 1.0,
            "ny": 2,
            "nx": 2,
        }
    )
    hostgrid = Grid2D(data, geodict)
    finedict = GeoDict(
        {
            "xmin": 0.0 - (1.0 / 3.0),
            "xmax": 1.0 + (1.0 / 3.0),
            "ymin": 0.0 - (1.0 / 3.0),
            "ymax": 1.0 + (1.0 / 3.0),
            "dx": 1.0 / 3.0,
            "dy": 1.0 / 3.0,
            "ny": 6,
            "nx": 6,
        }
    )
    finegrid = hostgrid.subdivide(finedict)
    output = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
            [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
            [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        ]
    )
    np.testing.assert_almost_equal(finegrid.getData(), output)
    print("Passed subdivide method test - aligned grids.")

    print("Testing subdivide method - non-aligned grids...")
    data = np.arange(0, 9).reshape((3, 3))
    geodict = GeoDict(
        {
            "xmin": 0.0,
            "xmax": 10.0,
            "ymin": 0.0,
            "ymax": 10.0,
            "dx": 5.0,
            "dy": 5.0,
            "ny": 3,
            "nx": 3,
        }
    )
    hostgrid = Grid2D(data, geodict)
    finedict = GeoDict(
        {
            "xmin": -2.5,
            "xmax": 11.5,
            "ymin": -1.5,
            "ymax": 10.5,
            "dx": 2.0,
            "dy": 2.0,
            "nx": 8,
            "ny": 7,
        }
    )
    N = np.nan
    print("Testing subdivide with min parameter...")
    finegrid = hostgrid.subdivide(finedict, cellFill="min")
    output = np.array(
        [
            [N, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0],
            [N, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0],
            [N, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0],
            [N, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0],
            [N, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0],
            [N, 6.0, 6.0, 7.0, 7.0, 7.0, 8.0, 8.0],
            [N, 6.0, 6.0, 7.0, 7.0, 7.0, 8.0, 8.0],
        ]
    )
    np.testing.assert_almost_equal(finegrid.getData(), output)
    print("Passed subdivide with min parameter...")
    print("Testing subdivide with max parameter...")
    finegrid = hostgrid.subdivide(finedict, cellFill="max")
    output = np.array(
        [
            [N, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            [N, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            [N, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0],
            [N, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0],
            [N, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 8.0],
            [N, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 8.0],
            [N, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 8.0],
        ]
    )
    np.testing.assert_almost_equal(finegrid.getData(), output)
    print("Passed subdivide with max parameter...")
    print("Testing subdivide with mean parameter...")
    finegrid = hostgrid.subdivide(finedict, cellFill="mean")
    output = np.array(
        [
            [N, 0.0, 0.0, 1.0, 1.0, 1.5, 2.0, 2.0],
            [N, 0.0, 0.0, 1.0, 1.0, 1.5, 2.0, 2.0],
            [N, 3.0, 3.0, 4.0, 4.0, 4.5, 5.0, 5.0],
            [N, 3.0, 3.0, 4.0, 4.0, 4.5, 5.0, 5.0],
            [N, 4.5, 4.5, 5.5, 5.5, 6.0, 6.5, 6.5],
            [N, 6.0, 6.0, 7.0, 7.0, 7.5, 8.0, 8.0],
            [N, 6.0, 6.0, 7.0, 7.0, 7.5, 8.0, 8.0],
        ]
    )
    np.testing.assert_almost_equal(finegrid.getData(), output)
    print("Passed subdivide with mean parameter...")
    print("Passed subdivide method test - non-aligned grids.")


def test_basics():
    geodict = GeoDict(
        {
            "xmin": 0.5,
            "xmax": 3.5,
            "ymin": 0.5,
            "ymax": 3.5,
            "dx": 1.0,
            "dy": 1.0,
            "ny": 4,
            "nx": 4,
        }
    )
    data = np.arange(0, 16).reshape(4, 4).astype(np.float32)
    grid = Grid2D(data, geodict)
    print(
        "Testing basic Grid2D functionality (retrieving data, lat/lon to pixel coordinates, etc..."
    )
    np.testing.assert_almost_equal(grid.getData(), data)

    assert grid.getGeoDict() == geodict

    assert grid.getBounds() == (geodict.xmin, geodict.xmax, geodict.ymin, geodict.ymax)

    lat, lon = grid.getLatLon(0, 0)

    assert lat == 3.5 and lon == 0.5

    row, col = grid.getRowCol(lat, lon)

    assert row == 0 and col == 0

    value = grid.getValue(lat, lon)

    assert value == 0

    frow, fcol = grid.getRowCol(1.0, 3.0, returnFloat=True)

    assert frow == 2.5 and fcol == 2.5

    irow, icol = grid.getRowCol(1.0, 3.0, returnFloat=False)

    assert irow == 2 and icol == 2

    # test getting values in and outside of the grid bounds
    lat = np.array([0.0, 0.5, 2.5, 4.0])
    lon = np.array([0.0, 0.5, 2.5, 4.0])
    default = np.nan
    output = np.array([np.nan, 12, 6, np.nan])
    value = grid.getValue(lat, lon, default=default)

    np.testing.assert_almost_equal(value, output)

    print(
        "Passed basic Grid2D functionality (retrieving data, lat/lon to pixel coordinates, etc..."
    )


def test_getvalue():
    array = np.arange(1, 26).reshape(5, 5)
    gdict = GeoDict(
        {
            "xmin": 1.0,
            "xmax": 5.0,
            "ymin": 1.0,
            "ymax": 5.0,
            "dx": 1.0,
            "dy": 1.0,
            "nx": 5,
            "ny": 5,
        }
    )
    grid = Grid2D(array, gdict)
    assert grid.getValue(3.0, 3.0) == 13
    lat = np.array([3.0, 4.0])
    lon = np.array([3.0, 3.0])
    test = grid.getValue(lat, lon)
    np.testing.assert_almost_equal(test, np.array([13, 8]))
    lat = np.array([[3.0, 4.0], [4.0, 5.0]])
    lon = np.array([[3.0, 3.0], [4.0, 4.0]])
    test = grid.getValue(lat, lon)
    np.testing.assert_almost_equal(test, np.array([[13, 8], [9, 4]]))


def test_cut():
    geodict = GeoDict(
        {
            "xmin": 0.5,
            "xmax": 4.5,
            "ymin": 0.5,
            "ymax": 4.5,
            "dx": 1.0,
            "dy": 1.0,
            "ny": 5,
            "nx": 5,
        }
    )
    data = np.arange(0, 25).reshape(5, 5)

    print("Testing data extraction...")
    grid = Grid2D(data, geodict)
    xmin, xmax, ymin, ymax = (2.5, 3.5, 2.5, 3.5)
    newgrid = grid.cut(xmin, xmax, ymin, ymax)
    output = np.array([[7, 8], [12, 13]])
    np.testing.assert_almost_equal(newgrid.getData(), output)
    print("Passed data extraction...")

    print("Testing data trimming with resampling...")
    # make a more complicated test using getboundswithin
    data = np.arange(0, 84).reshape(7, 12)
    geodict = GeoDict(
        {
            "xmin": -180,
            "xmax": 150,
            "ymin": -90,
            "ymax": 90,
            "dx": 30,
            "dy": 30,
            "nx": 12,
            "ny": 7,
        }
    )
    grid = Grid2D(data, geodict)
    sampledict = GeoDict.createDictFromBox(-75, 45, -45, 75, geodict.dx, geodict.dy)
    cutdict = geodict.getBoundsWithin(sampledict)
    newgrid = grid.cut(cutdict.xmin, cutdict.xmax, cutdict.ymin, cutdict.ymax)
    output = np.array(
        [[16, 17, 18, 19], [28, 29, 30, 31], [40, 41, 42, 43], [52, 53, 54, 55]]
    )
    np.testing.assert_almost_equal(newgrid.getData(), output)
    print("Passed data trimming with resampling...")

    print("Test cut with self-alignment...")
    geodict = GeoDict(
        {
            "xmin": 0.5,
            "xmax": 4.5,
            "ymin": 0.5,
            "ymax": 6.5,
            "dx": 1.0,
            "dy": 1.0,
            "nx": 5,
            "ny": 7,
        }
    )
    data = np.arange(0, 35).astype(np.float32).reshape(7, 5)
    grid = Grid2D(data, geodict)
    cutxmin = 1.7
    cutxmax = 3.7
    cutymin = 1.7
    cutymax = 5.7
    cutgrid = grid.cut(cutxmin, cutxmax, cutymin, cutymax, align=True)
    output = np.array([[7, 8], [12, 13], [17, 18], [22, 23]])
    np.testing.assert_almost_equal(cutgrid.getData(), output)
    print("Passed cut with self-alignment.")


def test_interpolate():
    geodict = GeoDict(
        {
            "xmin": 0.5,
            "xmax": 6.5,
            "ymin": 1.5,
            "ymax": 6.5,
            "dx": 1.0,
            "dy": 1.0,
            "ny": 6,
            "nx": 7,
        }
    )
    data = np.arange(14, 56).reshape(6, 7)

    for method in ["nearest", "linear", "cubic"]:
        print('Testing interpolate with method "%s"...' % method)
        grid = Grid2D(data, geodict)
        sampledict = GeoDict(
            {
                "xmin": 3.1,
                "xmax": 4.1,
                "ymin": 3.1,
                "ymax": 4.1,
                "dx": 1.0,
                "dy": 1.0,
                "ny": 2,
                "nx": 2,
            }
        )
        grid = grid.interpolateToGrid(sampledict, method=method)
        tgrid = grid.interpolate2(sampledict, method=method)
        if method == "nearest":
            output = np.array([[31.0, 32.0], [38.0, 39.0]])
        elif method == "linear":
            output = np.array([[33.4, 34.4], [40.4, 41.4]])
        elif method == "cubic":
            output = np.array([[33.4, 34.4], [40.4, 41.4]])
        else:
            pass
        np.testing.assert_almost_equal(grid.getData(), output)
        print('Passed interpolate with method "%s".' % method)
        np.testing.assert_almost_equal(tgrid.getData(), output)
        print('Passed interpolate2 with method "%s".' % method)

    # speed test of interpolateToGrid and interpolate2
    geodict = GeoDict.createDictFromBox(0, 10, 0, 10, 0.01, 0.01)
    data = np.random.rand(geodict.ny, geodict.nx)
    grid = Grid2D(data, geodict)
    sampledict = GeoDict.createDictFromBox(2, 8, 2, 8, 0.098, 0.098)
    t1 = time.time()
    grid2 = grid.interpolateToGrid(sampledict, method="linear")
    t2 = time.time()
    grid3 = grid.interpolate2(sampledict, method="linear")
    t3 = time.time()
    # np.testing.assert_almost_equal(grid2._data.sum(),grid3._data.sum())
    print("scipy method: %.3f seconds" % (t2 - t1))
    print("gdal  method: %.3f seconds" % (t3 - t2))

    # test interpolate2 when called with geodict that is aligned with
    # enclosing geodict. This should just cut the grid.
    lon_min = -125.4500
    lat_min = 39.3667
    lon_max = -123.1000
    lat_max = 41.1667
    nominal_lon_spacing = 0.0083
    nominal_lat_spacing = 0.0083
    nlon = 283
    nlat = 217
    host_geodict = GeoDict(
        {
            "xmin": lon_min,
            "xmax": lon_max,
            "ymin": lat_min,
            "ymax": lat_max,
            "dx": nominal_lon_spacing,
            "dy": nominal_lat_spacing,
            "nx": nlon,
            "ny": nlat,
        }
    )
    sample_xmin = host_geodict.xmin + host_geodict.dx * 5
    sample_xmax = host_geodict.xmax - host_geodict.dx * 5
    sample_ymin = host_geodict.ymin + host_geodict.dy * 5
    sample_ymax = host_geodict.ymax - host_geodict.dy * 5
    sample_geodict = GeoDict(
        {
            "xmin": sample_xmin,
            "xmax": sample_xmax,
            "ymin": sample_ymin,
            "ymax": sample_ymax,
            "dx": host_geodict.dx,
            "dy": host_geodict.dy,
            "nx": nlon - 10,
            "ny": nlat - 10,
        }
    )
    assert host_geodict.isAligned(sample_geodict)
    host_data = np.random.rand(nlat, nlon)
    host_data = host_data.astype(np.float32)
    host_grid = Grid2D(data=host_data, geodict=host_geodict)
    sample_grid = host_grid.interpolate2(sample_geodict)
    assert sample_grid._data.shape == (sample_geodict.ny, sample_geodict.nx)
    # these should be identical - see notes below
    assert sample_grid._data.dtype == host_grid._data.dtype

    # test interpolate2 with different data types
    # every input data type except for float64 should return float32
    # unless the sample geodict is aligned, in which case data type should
    # be identical to input
    xmin = host_geodict.xmin + (host_geodict.xmax - host_geodict.xmin) / 5
    xmax = host_geodict.xmax - (host_geodict.xmax - host_geodict.xmin) / 5
    ymin = host_geodict.ymin + (host_geodict.ymax - host_geodict.ymin) / 5
    ymax = host_geodict.ymax - (host_geodict.ymax - host_geodict.ymin) / 5
    dx = host_geodict.dx * 1.1
    dy = host_geodict.dy * 1.1
    ncols = int(((xmax - xmin) / dx) + 1)
    nrows = int(((ymax - ymin) / dy) + 1)
    # right/bottom edges of geodict will be adjusted if necessary
    sample_geodict = GeoDict(
        {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "dx": dx,
            "dy": dy,
            "nx": ncols,
            "ny": nrows,
        },
        adjust="bounds",
    )
    assert not host_geodict.isAligned(sample_geodict)
    host_data = np.random.randint(0, 100, size=(nlat, nlon), dtype=np.int16)
    host_grid = Grid2D(data=host_data, geodict=host_geodict)
    igrid1 = host_grid.interpolate2(sample_geodict)
    assert igrid1._data.dtype == np.float32


def test_rasterize():
    geodict = GeoDict(
        {
            "xmin": 0.5,
            "xmax": 3.5,
            "ymin": 0.5,
            "ymax": 3.5,
            "dx": 1.0,
            "dy": 1.0,
            "ny": 4,
            "nx": 4,
        }
    )
    print(
        "Testing rasterizeFromGeometry() burning in values from a polygon sequence..."
    )
    # Define two simple polygons and assign them to shapes
    poly1 = [(0.25, 3.75), (1.25, 3.25), (1.25, 2.25)]
    poly2 = [
        (2.25, 3.75),
        (3.25, 3.75),
        (3.75, 2.75),
        (3.75, 1.50),
        (3.25, 0.75),
        (2.25, 2.25),
    ]
    shape1 = {"properties": {"value": 5}, "geometry": mapping(Polygon(poly1))}
    shape2 = {"properties": {"value": 7}, "geometry": mapping(Polygon(poly2))}
    shapes = [shape1, shape2]
    print("Testing burning in values where polygons need not contain pixel centers...")
    grid = Grid2D.rasterizeFromGeometry(
        shapes, geodict, fillValue=0, attribute="value", mustContainCenter=False
    )
    output = np.array([[5, 5, 7, 7], [5, 5, 7, 7], [0, 0, 7, 7], [0, 0, 0, 7]])
    np.testing.assert_almost_equal(grid.getData(), output)
    print("Passed burning in values where polygons need not contain pixel centers.")

    print("Testing burning in values where polygons must contain pixel centers...")
    grid2 = Grid2D.rasterizeFromGeometry(
        shapes, geodict, fillValue=0, attribute="value", mustContainCenter=True
    )
    output = np.array([[5, 0, 7, 0], [0, 0, 7, 7], [0, 0, 0, 7], [0, 0, 0, 0]])
    np.testing.assert_almost_equal(grid2.getData(), output)
    print("Passed burning in values where polygons must contain pixel centers.")


def test_copy():
    data = np.arange(0, 16).astype(np.float32).reshape(4, 4)
    geodict = GeoDict(
        {
            "xmin": 0.5,
            "xmax": 3.5,
            "ymin": 0.5,
            "ymax": 3.5,
            "dx": 1.0,
            "dy": 1.0,
            "ny": 4,
            "nx": 4,
        }
    )
    grid1 = Grid2D(data, geodict)
    grid2 = grid1.copyFromGrid(grid1)
    grid1._data[0, 0] = np.nan
    print(grid2._data)
    print(grid2._geodict)


def test_setData():
    data = np.arange(0, 16).astype(np.float32).reshape(4, 4)
    geodict = GeoDict(
        {
            "xmin": 0.5,
            "xmax": 3.5,
            "ymin": 0.5,
            "ymax": 3.5,
            "dx": 1.0,
            "dy": 1.0,
            "ny": 4,
            "nx": 4,
        }
    )
    grid1 = Grid2D(data, geodict)
    x = np.ones((4, 4))
    try:
        grid1.setData(x)  # this should pass
        print("setData test passed.")
    except DataSetException as dse:
        print("setData test failed.")
    try:
        x = np.ones((5, 5))
        grid1.setData(x)
        print("setData test did not fail when it should have.")
    except DataSetException as dse:
        print("setData test failed as expected.")

    try:
        x = "fred"
        grid1.setData(x)
        print("setData test did not fail when it should have.")
    except DataSetException as dse:
        print("setData test failed as expected.")


def get_data_range_test():
    # a standard global grid, going from -180 to 180
    normal_dict = GeoDict(
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

    # test a simple example which does NOT cross the 180 meridian
    sample1 = (-125, 65, -20, 20)
    dict1 = Grid2D.getDataRange(normal_dict, sample1)
    cdict1 = {"iulx1": 0, "iuly1": 1, "ilrx1": 6, "ilry1": 4}
    assert dict1 == cdict1

    # test a less-simple example which DOES cross the 180 meridian
    sample2 = (-235, -10, -20, 20)
    dict2 = Grid2D.getDataRange(normal_dict, sample2)
    cdict2 = {
        "iulx1": 5,
        "iuly1": 1,
        "ilrx1": 6,
        "ilry1": 4,
        "iulx2": 0,
        "iuly2": 1,
        "ilrx2": 4,
        "ilry2": 4,
    }
    assert dict2 == cdict2

    # test a less-simple example which DOES cross the 180 meridian, and xmin > xmax
    sample3 = (125, -10, -20, 20)
    dict3 = Grid2D.getDataRange(normal_dict, sample3)
    cdict3 = {
        "iulx1": 5,
        "iuly1": 1,
        "ilrx1": 6,
        "ilry1": 4,
        "iulx2": 0,
        "iuly2": 1,
        "ilrx2": 4,
        "ilry2": 4,
    }
    assert dict3 == cdict3

    # test an example where the sample bounds are from 0 to 360
    sample4 = (160, 200, -20, 20)
    dict4 = Grid2D.getDataRange(normal_dict, sample4)
    cdict4 = {
        "iulx1": 5,
        "iuly1": 1,
        "ilrx1": 6,
        "ilry1": 4,
        "iulx2": 0,
        "iuly2": 1,
        "ilrx2": 2,
        "ilry2": 4,
    }
    assert dict4 == cdict4

    # test an example where the sample bounds are from 0 to 360
    sample5 = (220, 260, -20, 20)
    dict5 = Grid2D.getDataRange(normal_dict, sample5)
    cdict5 = {"iulx1": 0, "iuly1": 1, "ilrx1": 3, "ilry1": 4}
    assert dict5 == cdict5


def test_project():
    # test projecting a grid that wraps the 180 meridian
    gd = GeoDict.createDictFromBox(175, -175, -5, 5, 1.0, 1.0)
    ncells = gd.ny * gd.nx
    data = np.arange(0.0, ncells).reshape(gd.ny, gd.nx)
    grid = GDALGrid(data, gd)
    projstr = "+proj=merc +lat_ts=55 +lon_0=180 +ellps=WGS84"
    newgrid = grid.project(projstr, method="nearest")
    proj = pyproj.Proj(projstr)
    # what would the ul/lr corners be?
    ulx, uly = proj(grid._geodict.xmin, grid._geodict.ymax)
    lrx, lry = proj(grid._geodict.xmax, grid._geodict.ymin)
    # what if we back-project?
    newxmin, newymax = proj(newgrid._geodict.xmin, newgrid._geodict.ymax, inverse=True)
    newxmax, newymin = proj(newgrid._geodict.xmax, newgrid._geodict.ymin, inverse=True)
    x = 1

    # test simple projection
    data = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.int32,
    )
    geodict = {
        "xmin": 50,
        "xmax": 50.4,
        "ymin": 50,
        "ymax": 50.4,
        "dx": 0.1,
        "dy": 0.1,
        "nx": 5,
        "ny": 5,
    }
    gd = GeoDict(geodict)
    grid = GDALGrid(data, gd)
    projstr = "+proj=utm +zone=40 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs "
    newgrid = grid.project(projstr, method="nearest")

    try:
        tdir = tempfile.mkdtemp()
        outfile = os.path.join(tdir, "output.bil")
        grid.save(outfile)
        with rasterio.open(outfile) as src:
            aff = get_affine(src)
            data = src.read(1)
            src_crs = CRS().from_string(GeoDict.DEFAULT_PROJ4).to_dict()
            dst_crs = CRS().from_string(projstr).to_dict()
            nrows, ncols = data.shape
            left = aff.xoff
            top = aff.yoff
            right, bottom = aff * (ncols - 1, nrows - 1)
            dst_transform, width, height = calculate_default_transform(
                src_crs, dst_crs, ncols, nrows, left, bottom, right, top
            )
            destination = np.zeros((height, width))
            reproject(
                data,
                destination,
                src_transform=aff,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
            x = 1
    except:
        pass
    finally:
        shutil.rmtree(tdir)

    # cmpdata = np.array([[ 0.,  0.,  1.,  0.],
    #                     [ 0.,  0.,  1.,  0.],
    #                     [ 0.,  0.,  1.,  0.],
    #                     [ 1.,  1.,  1.,  1.],
    #                     [ 0.,  1.,  1.,  1.],
    #                     [ 0.,  0.,  1.,  0.]],dtype=np.float64)
    # np.testing.assert_almost_equal(cmpdata,newgrid._data)

    # cmpdict = GeoDict({'ymax': 5608705.974598191,
    #                    'ny': 6,
    #                    'ymin': 5571237.8659376735,
    #                    'nx': 4,
    #                    'xmax': 21363.975311354592,
    #                    'dy': 7493.621732103531,
    #                    'dx': 7493.621732103531,
    #                    'xmin': -756.8898849560019})

    # assert cmpdict == newgrid._geodict


def test_fiji():
    host_dict = {
        "xmin": 176.8874999998576,
        "xmax": -178.23750000014437,
        "ymin": -20.770833333331773,
        "ymax": -16.154166666666953,
        "dx": 0.00833333333333,
        "dy": 0.00833333333333,
        "ny": 555,
        "nx": 586,
    }
    sample_dict = {
        "xmin": 176.90416666666522,
        "xmax": -178.25416666666814,
        "ymin": -20.729166666666934,
        "ymax": -16.154166666666953,
        "dx": 0.0083333333333333,
        "dy": 0.0083333333333333,
        "ny": 550,
        "nx": 582,
    }
    host_geodict = GeoDict(host_dict)
    sample_geodict = GeoDict(sample_dict)
    host_data = np.zeros((host_geodict.ny, host_geodict.nx))
    host_grid = Grid2D(host_data, host_geodict)
    xi, yi = host_grid._getInterpCoords(sample_geodict)
    xcmp, ycmp = (2.0000000169155823, 0.0)
    np.testing.assert_almost_equal(xi[0], xcmp)
    np.testing.assert_almost_equal(yi[0], ycmp)

    # this is a case where the host grid crosses
    # the meridian but the sample grid is only on
    # the east side of the meridian.
    host_dict = {
        "xmin": 176.11235968666102,
        "xmax": -179.99597366223898,
        "ymin": -21.212639164039008,
        "ymax": -17.537639178739006,
        "dx": 0.008333333299999992,
        "dy": 0.008333333300000002,
        "ny": 442,
        "nx": 468,
    }
    sample_dict = {
        "xmin": 176.15402635316102,
        "xmax": 179.94569300466102,
        "ymin": -21.162639164239007,
        "ymax": -17.587639178539007,
        "dx": 0.008333333299999992,
        "dy": 0.008333333300000002,
        "ny": 430,
        "nx": 456,
    }
    host_geodict = GeoDict(host_dict)
    sample_geodict = GeoDict(sample_dict)
    host_data = np.zeros((host_geodict.ny, host_geodict.nx))
    host_grid = Grid2D(host_data, host_geodict)
    xi, yi = host_grid._getInterpCoords(sample_geodict)
    xcmp, ycmp = (4.99999999999908, 6.000000000000082)
    np.testing.assert_almost_equal(xi[0], xcmp)
    np.testing.assert_almost_equal(yi[0], ycmp)


if __name__ == "__main__":
    test_fiji()
    test_getvalue()
    test_project()
    test_subdivide()
    test_rasterize()
    test_interpolate()
    test_basics()
    test_cut()
    test_copy()
    test_setData()
    # get_data_range_test()
