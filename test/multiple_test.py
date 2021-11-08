#!/usr/bin/env python

# python 3 compatibility
from __future__ import print_function
import os.path
import sys

# stdlib imports
import abc
import textwrap
import glob
import os
from collections import OrderedDict

# hack the path so that I can debug these functions if I need to
homedir = os.path.dirname(os.path.abspath(__file__))  # where is this script?
mapiodir = os.path.abspath(os.path.join(homedir, ".."))
sys.path.insert(
    0, mapiodir
)  # put this at the front of the system path, ignoring any installed mapio stuff


# third party imports
from mapio.gridbase import Grid
from mapio.multiple import MultiGrid
from mapio.grid2d import Grid2D
from mapio.dataset import DataSetException
from mapio.geodict import GeoDict
import numpy as np
from scipy import interpolate
import shapely
from affine import Affine
from rasterio import features
from shapely.geometry import MultiPoint, Polygon, mapping


def test():
    print("Testing MultiGrid interpolate...")
    data = np.arange(14, 56).reshape(6, 7)
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
    layers = OrderedDict()
    layers["layer1"] = Grid2D(data, geodict)
    mgrid = MultiGrid(layers)
    sampledict = GeoDict(
        {
            "xmin": 3.0,
            "xmax": 4.0,
            "ymin": 3.0,
            "ymax": 4.0,
            "dx": 1.0,
            "dy": 1.0,
            "ny": 2,
            "nx": 2,
        }
    )
    for method in ["nearest", "linear", "cubic"]:
        mgrid2 = mgrid.interpolateToGrid(sampledict, method=method)
        if method == "nearest":
            output = np.array([[30.0, 32.0], [37.0, 39.0]])
        elif method == "linear":
            output = np.array([[34.0, 35.0], [41.0, 42.0]])
        elif method == "cubic":
            output = np.array([[34.0, 35.0], [41.0, 42.0]])
        else:
            pass
        np.testing.assert_almost_equal(mgrid2.getLayer("layer1").getData(), output)
    print("Passed MultiGrid interpolate test.")

    print("Testing bounds retrieval...")
    b1 = np.array(mgrid.getBounds())
    b2 = np.array((geodict.xmin, geodict.xmax, geodict.ymin, geodict.ymax))
    np.testing.assert_almost_equal(b1, b2)
    print("Passed bounds retrieval...")

    print("Testing MultiGrid subdivide test...")
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
    layers = OrderedDict()
    layers["layer1"] = Grid2D(data, geodict)
    hostgrid = MultiGrid(layers)
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
    np.testing.assert_almost_equal(finegrid.getLayer("layer1").getData(), output)
    print("Passed MultiGrid subdivide test.")


if __name__ == "__main__":
    test()
