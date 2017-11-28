# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2012-2017 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
This module is taken from OpenQuake, and methods that we do not need have
been removed. We chose to do this rather than install openquake because
this is such a small amount of the total openquake code.

The OpenQuake licence is included above in the comments.
"""

import operator

import numpy
try:
    import rtree
except ImportError:
    rtree = None

from openquake.baselib.python3compat import range, round

#: Earth radius in km.
EARTH_RADIUS = 6371.0

#: Maximum elevation on Earth in km.
EARTH_ELEVATION = -8.848


def geodetic_distance(lons1, lats1, lons2, lats2, diameter=2*EARTH_RADIUS):
    """
    Calculate the geodetic distance between two points or two collections
    of points.

    Parameters are coordinates in decimal degrees. They could be scalar
    float numbers or numpy arrays, in which case they should "broadcast
    together".

    Implements http://williams.best.vwh.net/avform.htm#Dist

    :returns:
        Distance in km, floating point scalar or numpy array of such.
    """
    lons1, lats1, lons2, lats2 = _prepare_coords(lons1, lats1, lons2, lats2)
    distance = numpy.arcsin(numpy.sqrt(
        numpy.sin((lats1 - lats2) / 2.0) ** 2.0
        + numpy.cos(lats1) * numpy.cos(lats2)
        * numpy.sin((lons1 - lons2) / 2.0) ** 2.0
    ))
    return diameter * distance


def azimuth(lons1, lats1, lons2, lats2):
    """
    Calculate the azimuth between two points or two collections of points.

    Parameters are the same as for :func:`geodetic_distance`.

    Implements an "alternative formula" from
    http://williams.best.vwh.net/avform.htm#Crs

    :returns:
        Azimuth as an angle between direction to north from first point and
        direction to the second point measured clockwise in decimal degrees.
    """
    lons1, lats1, lons2, lats2 = _prepare_coords(lons1, lats1, lons2, lats2)
    cos_lat2 = numpy.cos(lats2)
    true_course = numpy.degrees(numpy.arctan2(
        numpy.sin(lons1 - lons2) * cos_lat2,
        numpy.cos(lats1) * numpy.sin(lats2)
        - numpy.sin(lats1) * cos_lat2 * numpy.cos(lons1 - lons2)
    ))
    return (360 - true_course) % 360


def distance(lons1, lats1, depths1, lons2, lats2, depths2):
    """
    Calculate a distance between two points (or collections of points)
    considering points' depth.

    Calls :func:`geodetic_distance`, finds the "vertical" distance between
    points by subtracting one depth from another and combine both using
    Pythagoras theorem.

    :returns:
        Distance in km, a square root of sum of squares of :func:`geodetic
        <geodetic_distance>` distance and vertical distance, which is just
        a difference between depths.
    """
    hdist = geodetic_distance(lons1, lats1, lons2, lats2)
    vdist = depths1 - depths2
    return numpy.sqrt(hdist ** 2 + vdist ** 2)


def _prepare_coords(lons1, lats1, lons2, lats2):
    """
    Convert two pairs of spherical coordinates in decimal degrees
    to numpy arrays of radians. Makes sure that respective coordinates
    in pairs have the same shape.
    """
    lons1 = numpy.radians(lons1)
    lats1 = numpy.radians(lats1)
    assert lons1.shape == lats1.shape
    lons2 = numpy.radians(lons2)
    lats2 = numpy.radians(lats2)
    assert lons2.shape == lats2.shape
    return lons1, lats1, lons2, lats2
