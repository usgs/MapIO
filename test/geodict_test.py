#!/usr/bin/env python

# stdlib imports
import os.path
import pathlib
import sys

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# third party imports
import numpy as np
import pandas as pd
import rasterio
from mapio.dataset import DataSetException
from mapio.geodict import (
    GeoDict,
    affine_from_geodict,
    geodict_from_affine,
    geodict_from_src,
    get_longitude_intersection,
)

# hack the path so that I can debug these functions if I need to
homedir = os.path.dirname(os.path.abspath(__file__))  # where is this script?
mapiodir = os.path.abspath(os.path.join(homedir, ".."))
# put this at the front of the system path, ignoring any installed mapio stuff
sys.path.insert(0, mapiodir)


def test():
    # these values taken from the shakemap header of:
    # http://earthquake.usgs.gov/realtime/product/shakemap/ak12496371/ak/1453829475592/download/grid.xml

    print("Testing various dictionaries for consistency...")

    print("Testing consistent dictionary...")
    # this should pass, and will serve as the comparison from now on
    gdict = {
        "xmin": -160.340600,
        "xmax": -146.340600,
        "ymin": 54.104700,
        "ymax": 65.104700,
        "dx": 0.025000,
        "dy": 0.025000,
        "ny": 441,
        "nx": 561,
    }
    gd = GeoDict(gdict)
    print("Consistent dictionary passed.")

    print("Testing dictionary with inconsistent resolution...")
    # this should pass
    gdict = {
        "xmin": -160.340600,
        "xmax": -146.340600,
        "ymin": 54.104700,
        "ymax": 65.104700,
        "dx": 0.026000,
        "dy": 0.026000,
        "ny": 441,
        "nx": 561,
    }
    gd3 = GeoDict(gdict, adjust="res")
    assert gd3 == gd
    print("Dimensions modification passed.")

    print("Testing dictionary with inconsistent lower right corner...")
    # this should pass
    gdict = {
        "xmin": -160.340600,
        "xmax": -146.350600,
        "ymin": 54.103700,
        "ymax": 65.104700,
        "dx": 0.025000,
        "dy": 0.025000,
        "ny": 441,
        "nx": 561,
    }
    gd4 = GeoDict(gdict, adjust="bounds")
    assert gd4 == gd
    print("Corner modification passed.")

    print("Testing to make sure lat/lon and row/col calculations are correct...")
    # make sure the lat/lon row/col calculations are correct
    ndec = int(np.abs(np.log10(GeoDict.EPS)))
    lat, lon = gd.getLatLon(0, 0)
    dlat = np.abs(lat - gd.ymax)
    dlon = np.abs(lon - gd.xmin)
    assert dlat < GeoDict.EPS and dlon < GeoDict.EPS
    row, col = gd.getRowCol(lat, lon)
    assert row == 0 and col == 0

    lat, lon = gd.getLatLon(gd.ny - 1, gd.nx - 1)
    dlat = np.abs(lat - gd.ymin)
    dlon = np.abs(lon - gd.xmax)
    assert dlat < GeoDict.EPS and dlon < GeoDict.EPS
    row, col = gd.getRowCol(lat, lon)
    assert row == (gd.ny - 1) and col == (gd.nx - 1)
    print("lat/lon and row/col calculations are correct.")

    print("Testing a dictionary for a global grid...")
    # this is the file geodict for Landscan - should pass muster
    globaldict = {
        "nx": 43200,
        "ny": 20880,
        "dx": 0.00833333333333,
        "xmax": 179.99583333318935,
        "xmin": -179.99583333333334,
        "dy": 0.00833333333333,
        "ymax": 83.99583333326376,
        "ymin": -89.99583333333334,
    }
    gd5 = GeoDict(globaldict)
    lat, lon = gd5.getLatLon(gd5.ny - 1, gd5.nx - 1)
    dlat = np.abs(lat - gd5.ymin)
    dlon = np.abs(lon - gd5.xmax)
    assert dlat < GeoDict.EPS and dlon < GeoDict.EPS
    print("Global grid is internally consistent.")

    # Test class methods for creating a GeoDict
    print("Testing whether GeoDict creator class methods work...")
    xmin = -121.05333277776235
    xmax = -116.03833388890432
    ymin = 32.138334444506171
    ymax = 36.286665555493826
    dx = 0.0083333333333333332
    dy = 0.0083333333333333332
    gd6 = GeoDict.createDictFromBox(xmin, xmax, ymin, ymax, dx, dy, inside=False)
    assert gd6.xmax > xmax
    assert gd6.ymin < ymin
    print("Created dictionary (outside) is correct.")
    gd7 = GeoDict.createDictFromBox(xmin, xmax, ymin, ymax, dx, dy, inside=True)
    assert gd7.xmax < xmax
    assert gd7.ymin > ymin
    print("Created dictionary (inside) is correct.")
    xspan = 2.5
    yspan = 2.5
    gd8 = GeoDict.createDictFromCenter(xmin, ymin, dx, dy, xspan, yspan)
    print("Created dictionary (from center point) is valid.")

    print("Testing a geodict with dx/dy values that are NOT the same...")
    xmin, xmax, ymin, ymax = (
        -121.06166611109568,
        -116.03000055557099,
        32.130001111172838,
        36.294998888827159,
    )
    dx, dy = (0.009999722214505959, 0.009999444413578534)
    td = GeoDict.createDictFromBox(xmin, xmax, ymin, ymax, dx, dy)
    print("Passed testing a geodict with dx/dy values that are NOT the same...")

    # test getBoundsWithin
    # use global grid, and then a shakemap grid that we can get
    print("Testing getBoundsWithin...")
    grussia = {
        "xmin": 155.506400,
        "xmax": 161.506400,
        "ymin": 52.243000,
        "ymax": 55.771000,
        "dx": 0.016667,
        "dy": 0.016642,
        "nx": 361,
        "ny": 213,
    }
    gdrussia = GeoDict(grussia, adjust="res")
    sampledict = gd5.getBoundsWithin(gdrussia)
    xSmaller = sampledict.xmin > grussia["xmin"] and sampledict.xmax < grussia["xmax"]
    ySmaller = sampledict.ymin > grussia["ymin"] and sampledict.ymax < grussia["ymax"]
    assert xSmaller and ySmaller
    assert gd5.isAligned(sampledict)
    print("getBoundsWithin returned correct result.")

    print("Testing isAligned() method...")
    gd = GeoDict(
        {
            "xmin": 0.5,
            "xmax": 3.5,
            "ymin": 0.5,
            "ymax": 3.5,
            "dx": 1.0,
            "dy": 1.0,
            "nx": 4,
            "ny": 4,
        }
    )

    inside_aligned = GeoDict(
        {
            "xmin": 1.5,
            "xmax": 2.5,
            "ymin": 1.5,
            "ymax": 2.5,
            "dx": 1.0,
            "dy": 1.0,
            "nx": 2,
            "ny": 2,
        }
    )
    inside_not_aligned = GeoDict(
        {
            "xmin": 2.0,
            "xmax": 3.0,
            "ymin": 2.0,
            "ymax": 3.0,
            "dx": 1.0,
            "dy": 1.0,
            "nx": 2,
            "ny": 2,
        }
    )
    assert gd.isAligned(inside_aligned)
    assert not gd.isAligned(inside_not_aligned)
    print("Passed isAligned() method...")

    print("Testing getAligned method...")
    popdict = GeoDict(
        {
            "dx": 0.00833333333333,
            "dy": 0.00833333333333,
            "nx": 43200,
            "ny": 20880,
            "xmax": 179.99583333318935,
            "xmin": -179.99583333333334,
            "ymax": 83.99583333326376,
            "ymin": -89.99583333333334,
        }
    )
    sampledict = GeoDict(
        {
            "dx": 0.008333333333333333,
            "dy": 0.008336693548387094,
            "nx": 601,
            "ny": 497,
            "xmax": -116.046,
            "xmin": -121.046,
            "ymax": 36.2785,
            "ymin": 32.1435,
        }
    )
    aligndict = popdict.getAligned(sampledict)
    assert popdict.isAligned(aligndict)

    print("Testing geodict intersects method...")
    gd1 = GeoDict(
        {
            "xmin": 0.5,
            "xmax": 3.5,
            "ymin": 0.5,
            "ymax": 3.5,
            "dx": 1.0,
            "dy": 1.0,
            "nx": 4,
            "ny": 4,
        }
    )

    print("Testing geodict intersects method...")
    gd2 = GeoDict(
        {
            "xmin": 2.5,
            "xmax": 5.5,
            "ymin": 2.5,
            "ymax": 5.5,
            "dx": 1.0,
            "dy": 1.0,
            "nx": 4,
            "ny": 4,
        }
    )
    gd3 = GeoDict(
        {
            "xmin": 4.5,
            "xmax": 7.5,
            "ymin": 4.5,
            "ymax": 7.5,
            "dx": 1.0,
            "dy": 1.0,
            "nx": 4,
            "ny": 4,
        }
    )
    gd4 = GeoDict(
        {
            "xmin": 1.5,
            "xmax": 2.5,
            "ymin": 1.5,
            "ymax": 2.5,
            "dx": 1.0,
            "dy": 1.0,
            "nx": 2,
            "ny": 2,
        }
    )
    assert gd1.intersects(gd2)
    assert not gd1.intersects(gd3)
    print("Passed intersects method...")

    print("Testing geodict intersects method with real geographic data...")
    gda = GeoDict(
        {
            "ymax": 83.62083333333263,
            "nx": 43201,
            "ny": 20835,
            "dx": 0.00833333333333,
            "dy": 0.00833333333333,
            "xmin": -179.99583333333334,
            "ymin": -89.99583333326461,
            "xmax": -179.99583333347732,
        }
    )
    gdb = GeoDict(
        {
            "ymax": 28.729166666619193,
            "nx": 300,
            "ny": 264,
            "dx": 0.00833333333333,
            "dy": 0.00833333333333,
            "xmin": 84.08749999989436,
            "ymin": 26.537499999953404,
            "xmax": 86.57916666656007,
        }
    )
    assert gda.intersects(gdb)
    print("Passed geodict intersects method with real geographic data.")

    print("Testing geodict doesNotContain method...")
    assert gd1.doesNotContain(gd3)
    assert not gd1.doesNotContain(gd4)

    print("Passed doesNotContain method...")

    print("Testing geodict contains method...")

    assert gd1.contains(gd4)
    assert not gd1.contains(gd3)
    print("Passed contains method...")

    # print('Testing to see if getIntersection() method works...')
    # gd5 = GeoDict({'xmin':0.5,'xmax':6.5,
    #                'ymin':0.5,'ymax':8.5,
    #                'dx':1.0,'dy':1.0,
    #                'nx':7,'ny':9})
    # gd6 = GeoDict({'xmin':3.0,'xmax':8.0,
    #                'ymin':5.0,'ymax':10.0,
    #                'dx':1.0,'dy':1.0,
    #                'nx':6,'ny':6})
    # print('Passed test of getIntersection() method')


def test_longitude_intersection():
    array_xmin = [
        155.0000000,
        -180.0000000,
        -10.0000000,
        -155.0000000,
        -179.9999867,
        -180.0010417,
    ]
    array_xmax = [
        -155.0000000,
        180.0000000,
        10.0000000,
        155.0000000,
        180.0005032,
        180.0010417,
    ]
    array_fxmin = [
        174.0000000,
        174.0000000,
        -20.0000000,
        -10.0000000,
        -174.0000000,
        174.0000000,
    ]
    array_fxmax = [
        -174.0000000,
        -174.0000000,
        5.0000000,
        20.0000000,
        174.0000000,
        -174.0000000,
    ]
    array_txmin = [
        174.0000000,
        174.0000000,
        -10.0000000,
        -10.0000000,
        -174.0000000,
        174.0000000,
    ]
    array_txmax = [
        186.0000000,
        186.0000000,
        5.0000000,
        20.0000000,
        174.0000000,
        186.0000000,
    ]
    for xmin, xmax, fxmin, fxmax, cmp_txmin, cmp_txmax in zip(
        array_xmin, array_xmax, array_fxmin, array_fxmax, array_txmin, array_txmax
    ):
        txmin, txmax = get_longitude_intersection(xmin, xmax, fxmin, fxmax)
        assert cmp_txmin == txmin
        assert cmp_txmax == txmax
    # test a case we know does not intersect
    xmin = 30
    xmax = 40
    fxmin = 60
    fxmax = 70
    try:
        txmin, txmax = get_longitude_intersection(xmin, xmax, fxmin, fxmax)
    except Exception as e:
        assert str(e).find("No longitude intersection") > -1
        return
    raise AssertionError("Longitudes intersected when they should not")


def test_intersect_meridian2():
    dict1 = {
        "xmin": 179.2833,
        "xmax": -174.7333,
        "ymin": 50.1,
        "ymax": 53.8,
        "dx": 0.008333426183843995,
        "dy": 0.008333333333333325,
        "ny": 445,
        "nx": 719,
    }
    geodict1 = GeoDict(dict1)
    dict2 = {
        "xmin": -180.0,
        "xmax": 180.0,
        "ymin": -56.0,
        "ymax": 84.0,
        "dx": 0.0020833333333333333,
        "dy": 0.0020833333333333333,
        "ny": 67201,
        "nx": 172801,
    }
    geodict2 = GeoDict(dict2)
    intersection = geodict1.getIntersection(geodict2)
    cmp_dict = GeoDict(
        {
            "xmin": 179.2833333333333,
            "xmax": -174.73333333333338,
            "ymin": 50.099999999999994,
            "ymax": 53.8,
            "dx": 0.0020833333333333333,
            "dy": 0.0020833333333333333,
            "ny": 1777,
            "nx": 2873,
        }
    )
    assert intersection == cmp_dict


def test_bounds_within_meridian():
    host = GeoDict(
        {
            "xmin": -180,
            "xmax": 150,
            "ymin": -90,
            "ymax": 90,
            "dx": 30,
            "dy": 45,
            "nx": 12,
            "ny": 5,
        }
    )
    sample = GeoDict(
        {
            "xmin": 75,
            "xmax": -135,
            "ymin": -67.5,
            "ymax": 67.5,
            "dx": 30,
            "dy": 45,
            "nx": 6,
            "ny": 4,
        }
    )

    result = GeoDict(
        {
            "xmin": 90,
            "xmax": -150,
            "ymin": -45,
            "ymax": 45,
            "dx": 30,
            "dy": 45,
            "nx": 5,
            "ny": 3,
        }
    )

    inside = host.getBoundsWithin(sample)
    assert inside == result


def test_intersection():
    fxmin, fxmax = (178.311, -179.189)
    fymin, fymax = (50.616, 52.176)
    fdx, fdy = (0.025, 0.02516129032258068)
    fnx, fny = (101, 63)
    host = GeoDict(
        {
            "xmin": fxmin,
            "xmax": fxmax,
            "ymin": fymin,
            "ymax": fymax,
            "dx": fdx,
            "dy": fdy,
            "nx": fnx,
            "ny": fny,
        }
    )
    sxmin, sxmax = (178.31249999999858, -179.19583333333335)
    symin, symax = (50.62083333333279, 52.17083333333278)
    sdx, sdy = (0.0083333333333333, 0.0083333333333333)
    snx, sny = (300, 187)
    sample = GeoDict(
        {
            "xmin": sxmin,
            "xmax": sxmax,
            "ymin": symin,
            "ymax": symax,
            "dx": sdx,
            "dy": sdy,
            "nx": snx,
            "ny": sny,
        }
    )
    ixmin, ixmax = (178.31249999999858, -179.19583333333478)
    iymin, iymax = (50.62083333333278, 52.17083333333278)
    idx, idy = (0.0083333333333333, 0.0083333333333333)
    inx, iny = (300, 187)
    result = GeoDict(
        {
            "xmin": ixmin,
            "xmax": ixmax,
            "ymin": iymin,
            "ymax": iymax,
            "dx": idx,
            "dy": idy,
            "nx": inx,
            "ny": iny,
        }
    )
    intersection = host.getIntersection(sample)
    np.testing.assert_allclose(intersection.xmin, ixmin)
    np.testing.assert_allclose(intersection.xmax, ixmax)
    np.testing.assert_allclose(intersection.ymin, iymin)
    np.testing.assert_allclose(intersection.ymax, iymax)


def test_bounds_within():
    host = GeoDict(
        {
            "xmin": -180,
            "xmax": 150,
            "ymin": -90,
            "ymax": 90,
            "dx": 30,
            "dy": 45,
            "nx": 12,
            "ny": 5,
        }
    )
    sample = GeoDict(
        {
            "xmin": -75,
            "xmax": 45,
            "ymin": -67.5,
            "ymax": 67.5,
            "dx": 30,
            "dy": 45,
            "nx": 5,
            "ny": 4,
        }
    )

    result = GeoDict(
        {
            "xmin": -60,
            "xmax": 30,
            "ymin": -45,
            "ymax": 45,
            "dx": 30,
            "dy": 45,
            "nx": 4,
            "ny": 3,
        }
    )

    inside = host.getBoundsWithin(sample)
    assert inside == result

    # test degenerate case where first pass at getting inside bounds fails (xmax)
    host = GeoDict(
        {
            "ymax": 84.0,
            "dx": 0.008333333333333333,
            "ny": 16801,
            "xmax": 179.99166666666667,
            "xmin": -180.0,
            "nx": 43200,
            "dy": 0.008333333333333333,
            "ymin": -56.0,
        }
    )

    sample = GeoDict(
        {
            "ymax": 18.933333333333334,
            "dx": 0.008333333333333333,
            "ny": 877,
            "xmax": -90.28333333333333,
            "xmin": -97.86666666666666,
            "nx": 911,
            "dy": 0.008333333333333333,
            "ymin": 11.633333333333333,
        }
    )
    inside = host.getBoundsWithin(sample)
    assert sample.contains(inside)

    # this tests some logic that I can't figure out
    # a way to make happen inside getBoundsWithin().
    newymin, ymin = (11.63333333333334, 11.633333333333333)
    fdy = 0.008333333333333333
    yminrow = 8684.0
    fymax = 84.0
    newymin -= fdy / 2  # bump it down
    while newymin <= ymin:
        yminrow = yminrow - 1
        newymin = fymax - yminrow * fdy
    assert newymin > ymin


def test_bounds_within_real():
    fxmin, fxmax = (-179.995833333333, 179.99583333189372)
    fymin, fymax = (-89.99583333333332, 89.9958333326134)
    fdx, fdy = (0.0083333333333, 0.0083333333333)
    fnx, fny = (43200, 21600)
    xmin, xmax = (177.75, -179.75)
    ymin, ymax = (50.41625, 51.98375)
    dx, dy = (0.025, 0.02488095238095242)
    nx, ny = (101, 64)
    host = GeoDict(
        {
            "xmin": fxmin,
            "xmax": fxmax,
            "ymin": fymin,
            "ymax": fymax,
            "dx": fdx,
            "dy": fdy,
            "nx": fnx,
            "ny": fny,
        }
    )
    sample = GeoDict(
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
    result = GeoDict(
        {
            "xmin": 60,
            "xmax": -120,
            "ymin": -30,
            "ymax": 30,
            "dx": 60,
            "dy": 30,
            "nx": 4,
            "ny": 3,
        }
    )
    inside = host.getBoundsWithin(sample)
    ixmin, ixmax = (177.75416666523603, -179.7541666666673)
    iymin, iymax = (50.4208333327717, 51.9791666660988)
    idx, idy = (0.0083333333333, 0.0083333333333)
    inx, iny = (300, 188)
    result = GeoDict(
        {
            "xmin": ixmin,
            "xmax": ixmax,
            "ymin": iymin,
            "ymax": iymax,
            "dx": idx,
            "dy": idy,
            "nx": inx,
            "ny": iny,
        }
    )
    assert inside == result


def test_bounds_within_again():
    fxmin, fxmax = (-179.995833333333, 179.99583333189372)
    fymin, fymax = (-89.99583333333332, 89.9958333326134)
    fdx, fdy = (0.0083333333333, 0.0083333333333)
    fnx, fny = (43200, 21600)

    xmin, xmax = (97.233, 99.733)
    ymin, ymax = (84.854, 85.074)
    dx, dy = (0.025, 0.024444444444444317)
    nx, ny = (101, 10)

    host = GeoDict(
        {
            "xmin": fxmin,
            "xmax": fxmax,
            "ymin": fymin,
            "ymax": fymax,
            "dx": fdx,
            "dy": fdy,
            "nx": fnx,
            "ny": fny,
        }
    )
    sample = GeoDict(
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
    inside = host.getBoundsWithin(sample)


def test_contains():
    fxmin, fxmax = (-179.995833333333, 179.99583333189372)
    fymin, fymax = (-89.99583333333332, 89.9958333326134)
    fdx, fdy = (0.0083333333333, 0.0083333333333)
    fnx, fny = (43200, 21600)
    xmin, xmax = (-179.996, -177.496)
    ymin, ymax = (-21.89175, -19.55425)
    dx, dy = (0.025, 0.02513440860215052)
    nx, ny = (101, 94)
    host = GeoDict(
        {
            "xmin": fxmin,
            "xmax": fxmax,
            "ymin": fymin,
            "ymax": fymax,
            "dx": fdx,
            "dy": fdy,
            "nx": fnx,
            "ny": fny,
        }
    )
    sample = GeoDict(
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
    assert host.contains(sample)

    fxmin, fxmax = (179.9874, -174.5126)
    fymin, fymax = (-25.4997, -20.4413)
    fdx, fdy = (0.0083, 0.0083)
    fny, fnx = (608, 661)
    host = GeoDict(
        {
            "xmin": fxmin,
            "xmax": fxmax,
            "ymin": fymin,
            "ymax": fymax,
            "dx": fdx,
            "dy": fdy,
            "nx": fnx,
            "ny": fny,
        }
    )
    xmin, xmax = (-179.9667, -174.5667)
    ymin, ymax = (-25.4500, -20.4833)
    dx, dy = (0.0083, 0.0083)
    ny, nx = (597, 649)
    sample = GeoDict(
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
    assert host.contains(sample)


def test_shapes():
    gd = GeoDict.createDictFromBox(100.0, 102.0, 32.0, 34.0, 0.08, 0.08)

    # pass in scalar values
    inrow, incol = (10, 10)
    lat, lon = gd.getLatLon(inrow, incol)  # should get scalar results
    assert np.isscalar(lat) and np.isscalar(lon)

    # pass in array values
    inrow = np.array([10, 11, 12])
    incol = np.array([10, 11, 12])
    lat, lon = gd.getLatLon(inrow, incol)  # should get array results
    c1 = isinstance(lat, np.ndarray) and lat.shape == inrow.shape
    c2 = isinstance(lon, np.ndarray) and lon.shape == incol.shape
    assert c1 and c2

    # this should fail, because inputs are un-dimensioned numpy arrays
    inrow = np.array(10)
    incol = np.array(10)
    try:
        lat, lon = gd.getLatLon(inrow, incol)  # should get array results
        assert 1 == 0  # this should never happen
    except DataSetException as dse:
        pass


def test_bounds_meridian2():
    host = {
        "ny": 797,
        "dx": 0.0083333333,
        "ymin": -21.445972496439,
        "dy": 0.0083333333,
        "xmin": 178.33735967776101,
        "nx": 841,
        "xmax": -174.66264035023897,
        "ymax": -14.812639189639,
    }
    sample = {
        "ny": 773,
        "dx": 0.008333333333333333,
        "ymin": -21.35,
        "dy": 0.008333333333333333,
        "xmin": 178.43333333333334,
        "nx": 817,
        "xmax": -174.76666666666665,
        "ymax": -14.916666666666666,
    }

    host_geodict = GeoDict(host)
    sample_geodict = GeoDict(sample)
    within = host_geodict.getBoundsWithin(sample_geodict)
    assert within.xmin == 178.437359677361
    assert within.xmax == -174.770973683139

    fxmin, fxmax = (179.9874, -174.5126)
    fymin, fymax = (-25.4997, -20.4413)
    fdx, fdy = (0.0083, 0.0083)
    fny, fnx = (608, 661)
    host = GeoDict(
        {
            "xmin": fxmin,
            "xmax": fxmax,
            "ymin": fymin,
            "ymax": fymax,
            "dx": fdx,
            "dy": fdy,
            "nx": fnx,
            "ny": fny,
        }
    )
    xmin, xmax = (-179.9667, -174.5667)
    ymin, ymax = (-25.4500, -20.4833)
    dx, dy = (0.0083, 0.0083)
    ny, nx = (597, 649)
    sample = GeoDict(
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
    assert host.getBoundsWithin(sample)


def test_lat_lon_array():
    gd = GeoDict(
        {
            "xmin": 0,
            "xmax": 4,
            "ymin": 0,
            "ymax": 3,
            "dx": 1.0,
            "dy": 1.0,
            "nx": 5,
            "ny": 4,
        }
    )
    lat = np.array([0, 1, 2], dtype=np.float32)
    lon = np.array([2, 2, 2], dtype=np.float32)
    trow = np.array([3, 2, 1])
    tcol = np.array([2, 2, 2])
    row, col = gd.getRowCol(lat, lon)
    np.testing.assert_almost_equal(trow, row)
    np.testing.assert_almost_equal(tcol, col)

    gd = GeoDict(
        {
            "xmin": 176,
            "xmax": -176,
            "ymin": 0,
            "ymax": 6,
            "nx": 5,
            "ny": 4,
            "dx": 2,
            "dy": 2,
        }
    )
    lat = np.array([2, 4, 6], dtype=np.float32)
    lon = np.array([178, 180, -178], dtype=np.float32)
    row, col = gd.getRowCol(lat, lon)
    trow = np.array([2, 1, 0])
    tcol = np.array([1, 2, 3])
    np.testing.assert_almost_equal(trow, row)
    np.testing.assert_almost_equal(tcol, col)


def test_intersect_meridian():
    popd = {
        "xmin": -179.99583333333334,
        "xmax": 179.99583333333192,
        "ymin": -89.99583333333334,
        "ymax": 89.99583333333263,
        "dx": 0.0083333333333333,
        "dy": 0.0083333333333333,
        "ny": 21600,
        "nx": 43200,
    }
    shaked = {
        "xmin": 180.0,
        "xmax": -177.1333,
        "ymin": -21.6667,
        "ymax": -19.0,
        "dx": 0.008333430232558165,
        "dy": 0.008333437499999995,
        "ny": 321,
        "nx": 345,
    }
    popdict = GeoDict(popd)
    shakedict = GeoDict(shaked)
    assert popdict.intersects(shakedict)


def test_affine():
    lon_min = -125.4500
    lat_min = 39.3667
    lon_max = -123.1000
    lat_max = 41.1667
    dx = 0.0083
    dy = 0.0083
    nlon = 283
    nlat = 217
    geodict = GeoDict(
        {
            "xmin": lon_min,
            "xmax": lon_max,
            "ymin": lat_min,
            "ymax": lat_max,
            "dx": dx,
            "dy": dy,
            "nx": nlon,
            "ny": nlat,
        }
    )
    affine = affine_from_geodict(geodict)
    geodict2 = geodict_from_affine(affine, nlat, nlon)
    assert geodict2 == geodict

    # where is this script?
    homedir = os.path.dirname(os.path.abspath(__file__))
    # this is an HDF 5 file
    datafile = os.path.join(homedir, "data", "samplegrid_cdf.cdf")
    src = rasterio.open(datafile)
    cmpgeodict = GeoDict(
        {
            "xmin": 5.0,
            "xmax": 9.0,
            "ymin": 4.0,
            "ymax": 8.0,
            "dx": 1.0,
            "dy": 1.0,
            "nx": 5,
            "ny": 5,
        }
    )
    geodict = geodict_from_src(src)
    assert geodict == cmpgeodict


def map_test_intersections():
    # this function won't be called by py.test, only when run from cmd line
    array_xmin = [
        155.0000000,
        -180.0000000,
        -10.0000000,
        -155.0000000,
        -179.9999867,
        -180.0010417,
    ]
    array_xmax = [
        -155.0000000,
        180.0000000,
        10.0000000,
        155.0000000,
        180.0005032,
        180.0010417,
    ]
    array_fxmin = [
        174.0000000,
        174.0000000,
        -20.0000000,
        -10.0000000,
        -174.0000000,
        174.0000000,
    ]
    array_fxmax = [
        -174.0000000,
        -174.0000000,
        5.0000000,
        20.0000000,
        174.0000000,
        -174.0000000,
    ]
    array_txmin = [
        174.0000000,
        174.0000000,
        -10.0000000,
        -10.0000000,
        -174.0000000,
        174.0000000,
    ]
    array_txmax = [
        186.0000000,
        186.0000000,
        5.0000000,
        20.0000000,
        174.0000000,
        186.0000000,
    ]
    ymin = -20
    ymax = 20
    dx1 = 0.08
    dy1 = 0.08
    dx2 = 0.008
    dy2 = 0.008
    idx = 1
    for xmin, xmax, fxmin, fxmax, cmp_txmin, cmp_txmax in zip(
        array_xmin,
        array_xmax,
        array_fxmin,
        array_fxmax,
        array_txmin,
        array_txmax,
    ):
        dict1 = GeoDict.createDictFromBox(xmin, xmax, ymin, ymax, dx1, dy1)
        dict2 = GeoDict.createDictFromBox(fxmin, fxmax, ymin, ymax, dx2, dy2)
        try:
            intersection = dict1.getIntersection(dict2)
        except:
            intersection = dict1.getIntersection(dict2)

        # which hemisphere are we in?
        int_xmin = intersection.xmin
        int_xmax = intersection.xmax
        if int_xmin > int_xmax:
            int_xmax += 360
        x_av = (int_xmax + int_xmin) / 2
        clon = 180
        if x_av >= -90 and x_av < 90:
            clon = 0

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=clon))

        # make the map global rather than have it zoom in to
        # the extents of any plotted data
        ax.set_global()

        ax.stock_img()
        ax.coastlines()

        # draw the central meridian
        ax.plot([clon, clon], [-90, 90], "k", transform=ccrs.PlateCarree())
        # draw the dict1 box
        if xmin > xmax:
            xmax += 360
        if fxmin > fxmax:
            fxmax += 360
        dict1_xbox = [xmin, xmax, xmax, xmin, xmin]
        dict1_ybox = [ymax, ymax, ymin, ymin, ymax]
        dict2_xbox = [fxmin, fxmax, fxmax, fxmin, fxmin]
        dict2_ybox = [ymax, ymax, ymin, ymin, ymax]
        ax.plot(
            dict1_xbox,
            dict1_ybox,
            "r",
            transform=ccrs.PlateCarree(),
        )
        ax.plot(
            dict2_xbox,
            dict2_ybox,
            "b",
            transform=ccrs.PlateCarree(),
        )
        int_xmin = intersection.xmin
        int_xmax = intersection.xmax
        int_ymin = intersection.ymin
        int_ymax = intersection.ymax
        if int_xmin > int_xmax:
            int_xmax += 360
        int_xbox = [int_xmin, int_xmax, int_xmax, int_xmin, int_xmin]
        int_ybox = [int_ymax, int_ymax, int_ymin, int_ymin, int_ymax]
        fillcolor = np.array([235, 52, 198, 128]) / 255
        ax.fill(
            int_xbox,
            int_ybox,
            color=fillcolor,
            transform=ccrs.PlateCarree(),
        )
        fname = pathlib.Path.home() / f"example_{idx:02d}.png"
        tstr = f"Intersection: {int_xmin}:{int_xmax}"
        ax.set_title(tstr)
        fig.savefig(fname)
        print(f"Saved {fname}")
        idx += 1


def test_robust_intersections():
    # this function won't be called by py.test, only when run from cmd line
    array_xmin = [
        155.0000000,
        -180.0000000,
        -10.0000000,
        -155.0000000,
        -179.9999867,
        -180.0010417,
    ]
    array_xmax = [
        -155.0000000,
        180.0000000,
        10.0000000,
        155.0000000,
        180.0005032,
        180.0010417,
    ]
    array_fxmin = [
        174.0000000,
        174.0000000,
        -20.0000000,
        -10.0000000,
        -174.0000000,
        174.0000000,
    ]
    array_fxmax = [
        -174.0000000,
        -174.0000000,
        5.0000000,
        20.0000000,
        174.0000000,
        -174.0000000,
    ]
    array_txmin = [
        174.0000000,
        174.0000000,
        -10.0000000,
        -10.0000000,
        -174.0000000,
        174.0000000,
    ]
    array_txmax = [
        186.0000000,
        186.0000000,
        5.0000000,
        20.0000000,
        174.0000000,
        186.0000000,
    ]
    ymin = -20
    ymax = 20
    dx1 = 0.08
    dy1 = 0.08
    dx2 = 0.008
    dy2 = 0.008
    for xmin, xmax, fxmin, fxmax, cmp_txmin, cmp_txmax in zip(
        array_xmin,
        array_xmax,
        array_fxmin,
        array_fxmax,
        array_txmin,
        array_txmax,
    ):
        dict1 = GeoDict.createDictFromBox(xmin, xmax, ymin, ymax, dx1, dy1)
        dict2 = GeoDict.createDictFromBox(fxmin, fxmax, ymin, ymax, dx2, dy2)
        try:
            assert dict1.intersects(dict2)
        except:
            dict1 = GeoDict.createDictFromBox(xmin, xmax, ymin, ymax, dx1, dy1)
            dict1.intersects(dict2)
    x = 1


if __name__ == "__main__":
    test_robust_intersections()
    test_longitude_intersection()
    map_test_intersections()
    test_intersect_meridian2()
    test_affine()
    test_intersect_meridian()
    test_lat_lon_array()
    test_bounds_meridian2()
    test_shapes()
    test()
    test_contains()
    test_bounds_within_again()
    test_bounds_within_real()
    test_intersection()
    test_bounds_within()
    test_bounds_within_meridian()
