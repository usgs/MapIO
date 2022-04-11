#!/usr/bin/env python

# third party imports
import numpy as np
from .dataset import DataSetException
import pyproj
from rasterio.transform import Affine
import rasterio

GLOBE_SPAN = 359.0


def is_valid_projection(projstr):
    try:
        pyproj.Proj(projstr)
    except BaseException:
        return False
    return True


class GeoDict(object):
    EPS = 1e-12
    DEFAULT_PROJ4 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    DIST_THRESH = 0.01 / (111.191 * 1000)  # 1 centimeter in decimal degrees
    REQ_KEYS = ["xmin", "xmax", "ymin", "ymax", "dx", "dy", "ny", "nx"]

    def __init__(self, geodict, adjust="bounds"):
        """
        An object which represents the spatial information for a grid and is guaranteed
        to be self-consistent.

        Args:
            geodict (GeoDict):
                contains at least some of the following fields:
                    - xmin Longitude minimum (decimal degrees) (Center of upper left
                    cell)
                    - xmax Longitude maximum (decimal degrees) (Center of upper right
                    cell)
                    - ymin Longitude minimum (decimal degrees) (Center of lower left
                    cell)
                    - ymax Longitude maximum (decimal degrees) (Center of lower right
                    cell)
                    - dx Cell width (decimal degrees)
                    - dy Cell height (decimal degrees)
                    - ny Number of rows of input data (must match input data dimensions)
                    - nx Number of columns of input data (must match input data
                    dimensions).
                    - projection proj4 string defining projection.  Optional. If not
                    specified,
                    assumed to be
                    "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
                    (WGS-84 geographic).
            adjust (:obj:`str`, optional):
                String (one of 'bounds','res')
                    'bounds': dx/dy, nx/ny, xmin/ymax are assumed to be correct,
                    xmax/ymin will be recalculated.
                    'res': nx/ny, xmin/ymax, xmax/ymin and assumed to be correct,
                    dx/dy will be recalculated.
        """
        for key in self.REQ_KEYS:
            if key not in geodict.keys():
                raise DataSetException(
                    'Missing required key "%s" from input geodict dictionary' % key
                )

        self._xmin = float(geodict["xmin"])
        self._xmax = float(geodict["xmax"])
        self._ymin = float(geodict["ymin"])
        self._ymax = float(geodict["ymax"])
        self._dx = float(geodict["dx"])
        self._dy = float(geodict["dy"])
        self._ny = int(geodict["ny"])
        self._nx = int(geodict["nx"])
        self._nodata = None
        if "nodata" in geodict:
            self._nodata = geodict["nodata"]

        if "projection" in geodict:
            projstr = geodict["projection"]
            if is_valid_projection(projstr):
                self._projection = projstr
            else:
                raise DataSetException(f"Projection string '{projstr}' is invalid.")
        else:
            self._projection = self.DEFAULT_PROJ4
        self.validate(adjust)

    @classmethod
    def createDictFromBox(cls, xmin, xmax, ymin, ymax, dx, dy, inside=False):
        """Create GeoDict from two corner points and an x/y resolution.

        Args:
            xmin (float):
                X coordinate of center of upper left pixel.
            xmax (float):
                X coordinate of center of lower right pixel.
            ymin(float):
                Y coordinate of center of lower right pixel.
            ymax (float):
                Y coordinate of center of upper left pixel.
            dx (float):
                Width of pixels.
            dy (float):
                Height of pixels.
            inside (bool):
                indicating whether to ensure that resulting GeoDict
                falls inside or outside the input bounds.
        Returns:
          GeoDict object.
        """
        if xmin > xmax:
            txmax = xmax + 360
        else:
            txmax = xmax
        if inside:
            nx = np.floor(((txmax - xmin + GeoDict.EPS) / dx) + 1)
            ny = np.floor(((ymax - ymin + GeoDict.EPS) / dy) + 1)
        else:
            nx = np.ceil(((txmax - xmin - GeoDict.EPS) / dx) + 1)
            ny = np.ceil(((ymax - ymin - GeoDict.EPS) / dy) + 1)
        xmax2 = xmin + (nx - 1) * dx
        ymin2 = ymax - (ny - 1) * dy
        return cls(
            {
                "xmin": xmin,
                "xmax": xmax2,
                "ymin": ymin2,
                "ymax": ymax,
                "dx": dx,
                "dy": dy,
                "nx": nx,
                "ny": ny,
            }
        )

    @classmethod
    def createDictFromCenter(cls, cx, cy, dx, dy, xspan, yspan):
        """Create GeoDict from a center point, dx/dy and a width and height.

        Args:
            cx (float):
                X coordinate of center point.
            cy (float):
                Y coordinate of center point.
            dx (float):
                Width of pixels.
            dy (float):
                Height of pixels.
            xspan (float):
                Width of desired box.
            yspan (float):
                Height of desired box.
        Returns:
            GeoDict object.
        """
        xmin = cx - xspan / 2.0
        xmax = cx + xspan / 2.0
        ymin = cy - yspan / 2.0
        ymax = cy + yspan / 2.0
        return cls.createDictFromBox(xmin, xmax, ymin, ymax, dx, dy)

    def setProjection(self, projection):
        """Set a new projection for the GeoDict.

        Args:
            projection (str):
                Valid proj4 string.
        Raises:
            DataSetException:
                When input is not valid proj4.
        """
        try:
            pyproj.Proj(projection)
        except BaseException:
            raise DataSetException("%s is not a valid proj4 string." % projection)
        self._projection = projection

    def getAligned(self, geodict, inside=False):
        """Return a geodict that is close to the input geodict bounds but aligned with
        this GeoDict.

        Args:
            geodict (GeoDict):
                Input GeoDict object, whose bounds will be used, but dx/dy and nx/ny
                ignored.
            inside (bool):
                indicating whether the aligned geodict should be inside or
                outside input geodict.
        Returns:
            GeoDict object which is guaranteed to be grid-aligned with this GeoDict.
        """
        dx = self.dx
        dy = self.dy
        # how many columns are there between the host and input geodict left edges?
        if inside:
            falsenx = np.ceil((geodict.xmin - self.xmin) / dx)
        else:
            falsenx = np.floor((geodict.xmin - self.xmin) / dx)
        # use that number of rows to calculate what aligned xmin should be
        newxmin = self.xmin + falsenx * dx
        # how many columns are there between the host and input geodict right edges?
        if inside:
            falsenx = np.floor((geodict.xmax - self.xmax) / dx)
        else:
            falsenx = np.ceil((geodict.xmax - self.xmax) / dx)
        # use that number of rows to calculate what aligned xmax should be
        newxmax = self.xmax + falsenx * dx

        # if we wound up going past 180, correct x values to be within -180->180.
        if newxmin > 180:
            newxmin -= 360
        if newxmax > 180:
            newxmax -= 360

        # how many columns are there between the host and input geodict bottom edges?
        if inside:
            falseny = np.ceil((geodict.ymin - self.ymin) / dy)
        else:
            falseny = np.floor((geodict.ymin - self.ymin) / dy)
        # use that number of rows to calculate what aligned ymin should be
        newymin = self.ymin + falseny * dy
        # how many columns are there between the host and input geodict top edges?
        if inside:
            falseny = np.floor((geodict.ymax - self.ymax) / dy)
        else:
            falseny = np.ceil((geodict.ymax - self.ymax) / dy)
        # use that number of rows to calculate what aligned ymax should be
        newymax = self.ymax + falseny * dy

        nx = int(np.round((newxmax - newxmin) / dx + 1))
        ny = int(np.round((newymax - newymin) / dy + 1))

        gd = GeoDict(
            {
                "xmin": newxmin,
                "xmax": newxmax,
                "ymin": newymin,
                "ymax": newymax,
                "dx": dx,
                "dy": dy,
                "nx": nx,
                "ny": ny,
            }
        )
        return gd

    def getIntersection(self, geodict):
        """Return a geodict defining intersected area, retaining resolution of the input
        geodict.

        Args:
            geodict (GeoDict):
                Input GeoDict object, which should intersect with this GeoDict.
        Returns:
            GeoDict which represents the intersected area, and is aligned with the input
            geodict.
        Raises:
            DataSetException:
                When input geodict does not intersect at all with this GeoDict.
        """

        # return modified input geodict where bounds have been adjusted to be inside
        # self. output should align with input.
        if not self.intersects(geodict):
            raise DataSetException("Input geodict has no overlap.")
        fxmin, fxmax, fymin, fymax = (self.xmin, self.xmax, self.ymin, self.ymax)
        xmin, xmax, ymin, ymax = (
            geodict.xmin,
            geodict.xmax,
            geodict.ymin,
            geodict.ymax,
        )

        # there are some weird special cases here with longitudes (180 meridian)
        # handling these in separate testable function
        txmin, txmax = get_longitude_intersection(xmin, xmax, fxmin, fxmax)

        # calculate latitude intersection
        tymin = max(fymin, ymin)
        tymax = min(fymax, ymax)

        # we're using the input geodict resolution for output
        dx, dy = (geodict.dx, geodict.dy)

        # now align those bounds with the input geodict
        trow, tcol = geodict.getRowCol(tymax, txmin, returnFloat=True)
        fleftcol = int(np.round(tcol))
        ftoprow = int(np.round(trow))
        newymax, newxmin = geodict.getLatLon(ftoprow, fleftcol)

        trow, tcol = geodict.getRowCol(tymin, txmax, returnFloat=True)
        frightcol = int(np.round(tcol))
        fbottomrow = int(np.round(trow))
        newymin, newxmax = geodict.getLatLon(fbottomrow, frightcol)

        nx = int(np.round((newxmax - newxmin) / dx + 1))
        ny = int(np.round((newymax - newymin) / dy + 1))

        if newxmax > 180:
            newxmax = newxmax - 360

        outdict = GeoDict(
            {
                "xmin": newxmin,
                "xmax": newxmax,
                "ymin": newymin,
                "ymax": newymax,
                "dx": dx,
                "dy": dy,
                "nx": nx,
                "ny": ny,
            }
        )
        return outdict

    def getBoundsWithin(self, geodict):
        """Create a GeoDict by finding the maximum bounding
           box aligned with enclosing GeoDict that is guaranteed
           to be inside input GeoDict.

        Args:
            geodict (GeoDict):
                GeoDict object that output GeoDict will be contained by.
        Raises:
            DataSetException:
                When input geodict is not fully contained by this GeoDict, or
                if the output GeoDict cannot be aligned with this GeoDict (this
                shouldn't happen).
        """
        if not self.contains(geodict):
            raise DataSetException(
                "Input geodict not fully contained by this GeoDict object."
            )

        # if the input geodict is identical to the host grid, then just return
        # that geodict.
        if self == geodict:
            return geodict.copy()

        # fxmin etc. are the host bounds
        fxmin, fymax = (self.xmin, self.ymax)

        # xmin etc. are the geodict bounds
        xmin, xmax, ymin, ymax = (
            geodict.xmin,
            geodict.xmax,
            geodict.ymin,
            geodict.ymax,
        )
        fdx, fdy = (self.dx, self.dy)
        fnx = self.nx

        # how many columns from the host xmin is the sample xmin?
        xmincol = np.ceil((xmin - fxmin) / fdx)
        # how many columns from the host xmin is the sample xmax?
        if xmax < fxmin and xmax < 0:
            xmaxcol = np.floor(((xmax + 360) - fxmin) / fdx)
        else:
            xmaxcol = np.floor((xmax - fxmin) / fdx)

        newxmin = fxmin + xmincol * fdx
        newxmax = fxmin + xmaxcol * fdx
        yminrow = np.floor((fymax - ymin) / fdy)
        ymaxrow = np.ceil((fymax - ymax) / fdy)

        newymin = fymax - yminrow * fdy
        newymax = fymax - ymaxrow * fdy

        # if our original geodict crossed the 180 meridian, then
        # let's set our xmin/xmax back to the xmin > xmax scenario.
        if xmax < fxmin:
            newxmax -= 360

        # in certain edge cases, the "inside" bounds can be equal (or nearly)
        # to input bounds. Check for this here, and nudge in one pixel until
        # fixed
        while newxmin <= xmin:
            xmincol = xmincol + 1
            newxmin = fxmin + xmincol * fdx

        while newxmax >= xmax:
            xmaxcol = xmaxcol - 1
            newxmax = fxmin + xmaxcol * fdx

        while newymin <= ymin:
            yminrow = yminrow - 1
            newymin = fymax - yminrow * fdy

        while newymax >= ymax:
            ymaxrow = ymaxrow + 1
            newymax = fymax - ymaxrow * fdy

        ny = int((yminrow - ymaxrow) + 1)
        # if the input geodict crosses the 180 meridian, deal with this...
        if xmin > xmax:
            nx = ((fnx - xmincol)) + (xmaxcol + 1)
        else:
            nx = int((xmaxcol - xmincol) + 1)

        # because validate() calculates ymin and xmax differently,
        # let's re-calculate those that way and see if we get the same results
        tmp_new_ymin = newymax - fdy * (ny - 1)
        tmp_new_xmax = newxmin + fdx * (nx - 1)
        tmp_nx = nx
        tmp_ny = ny
        tmp_xmax = xmax
        if tmp_new_xmax > 180:
            tmp_xmax += 360

        # if we don't get the same results, adjust nx/ny until we do.
        while tmp_new_ymin < ymin:
            tmp_ny = tmp_ny - 1
            tmp_new_ymin = newymax - fdy * (tmp_ny - 1)
        while tmp_new_xmax > tmp_xmax:
            tmp_nx = tmp_nx - 1
            tmp_new_xmax = newxmin + fdx * (tmp_nx - 1)

        if tmp_new_ymin != newymin:
            newymin = tmp_new_ymin
            ny = tmp_ny

        if tmp_new_xmax != newxmax:
            newxmax = tmp_new_xmax
            nx = tmp_nx

        outgeodict = GeoDict(
            {
                "xmin": newxmin,
                "xmax": newxmax,
                "ymin": newymin,
                "ymax": newymax,
                "dx": fdx,
                "dy": fdy,
                "ny": ny,
                "nx": nx,
            }
        )
        isaligned = self.isAligned(outgeodict)
        if not isaligned:
            raise DataSetException(
                "getBoundsWithin() cannot create an aligned geodict."
            )

        isinside = geodict.contains(outgeodict)
        if not isinside:
            raise DataSetException(
                "getBoundsWithin() cannot create an geodict inside input."
            )

        return outgeodict

    def isAligned(self, geodict):
        """Determines if input geodict is grid-aligned with this GeoDict.

        Args:
            geodict (GeoDict):
                Input geodict whose cells must all be grid-aligned with this GeoDict.
        Returns:
            True when geodict is grid-aligned, and False if not.
        """
        dx1 = self._dx
        dx2 = geodict.dx
        dy1 = self._dy
        dy2 = geodict.dy

        dx_close = np.isclose(dx1, dx2, atol=self.EPS)
        dy_close = np.isclose(dy1, dy2, atol=self.EPS)

        if not dx_close or not dy_close:
            return False

        xmin1 = self._xmin
        xmin2 = geodict.xmin
        ymin1 = self._ymin
        ymin2 = geodict.ymin
        t1 = (xmin2 - xmin1) / dx1
        t2 = np.round(t1)
        x_close = np.isclose(t1, t2)
        t1 = (ymin2 - ymin1) / dy1
        t2 = np.round(t1)
        y_close = np.isclose(t1, t2)
        # x_rem = ((xmin2-xmin1) / dx1) < self.EPS
        # y_rem = ((ymin2-ymin1) % dy1) < self.EPS

        if x_close and y_close:
            return True
        return False

    def doesNotContain(self, geodict):
        """Determine if input geodict is completely outside this GeoDict.

        Args:
            geodict (GeoDict):
                Input GeoDict object.
        Returns:
            True if input geodict is completely outside this GeoDict,
            False if not.
        """
        a, b = (self.xmin, self.ymax)
        e, f = (geodict.xmin, geodict.ymax)
        c, _ = (self.xmax, self.ymin)
        g, h = (geodict.xmax, geodict.ymin)
        outside_x = e > c or a > g
        outside_y = f > b or b > h
        # outside_x = geodict.xmin < self._xmin and geodict.xmax > self._xmax
        # outside_y = geodict.ymin < self._ymin and geodict.ymax > self._ymax
        if outside_x and outside_y:
            return True
        return False

    def intersects(self, geodict):
        """Determine if input geodict intersects this GeoDict.

        Args:
            geodict (GeoDict):
                Input GeoDict object.
        Returns:
            True if input geodict intersects with this GeoDict,
            False if not.
        """
        try:
            xmin = self.xmin
            xmax = self.xmax
            fxmin = geodict.xmin
            fxmax = geodict.xmax
            txmin, txmax = get_longitude_intersection(xmin, xmax, fxmin, fxmax)
        except Exception:
            return False
        ymin = self.ymin
        ymax = self.ymax
        fymin = geodict.ymin
        fymax = geodict.ymax
        self_inside_input = ymin <= fymin and ymax >= fymin
        input_inside_self = fymin <= ymin and fymax >= ymin
        if self_inside_input or input_inside_self:
            return True

    def contains(self, geodict):
        """Determine if input geodict is completely outside this GeoDict.

        Args:
            geodict (GeoDict):
                Input GeoDict object.
        Returns:
            True if input geodict is completely outside this GeoDict,
            False if not.
        """
        inside_x = False
        if np.abs((self.xmax + self.dx) - (self.xmin + 360)) < self.dx * 0.01:
            inside_x = True
        if not inside_x:
            if geodict.xmin < self.xmin and (geodict.xmin + 360) < self.xmax:
                inside_x = True
            else:
                if self.xmax > self.xmin:
                    xmin_inside = (geodict.xmin + self.EPS) >= self._xmin
                    xmax_inside = (geodict.xmax - self.EPS) <= self._xmax
                else:
                    xmin_inside = (geodict.xmin + self.EPS) >= self._xmin
                    xmax_inside = (geodict.xmax - self.EPS) <= self._xmax + 360
                inside_x = xmin_inside and xmax_inside
        ymin_inside = (geodict.ymin + self.EPS) >= self._ymin
        ymax_inside = (geodict.ymax - self.EPS) <= self._ymax
        inside_y = ymin_inside and ymax_inside
        if inside_x and inside_y:
            return True
        return False

    def asDict(self):
        """Return GeoDict object in dictionary representation.

        Returns:
            Dictionary containing the same fields as found in constructor.
        """
        tdict = {}
        tdict["xmin"] = self._xmin
        tdict["xmax"] = self._xmax
        tdict["ymin"] = self._ymin
        tdict["ymax"] = self._ymax
        tdict["dx"] = self._dx
        tdict["dy"] = self._dy
        tdict["ny"] = self._ny
        tdict["nx"] = self._nx
        return tdict

    def __repr__(self):
        """Return a string representation of the object."""

        rfmt = """Bounds: (%.4f,%.4f,%.4f,%.4f)\nDims: (%.4f,%.4f)\nShape: (%i,%i)"""
        rtpl = (
            self._xmin,
            self._xmax,
            self._ymin,
            self._ymax,
            self._dx,
            self._dy,
            self._ny,
            self._nx,
        )
        return rfmt % rtpl

    def copy(self):
        """Return an object that is a complete copy of this GeoDict.

        Returns:
            A GeoDict object whose elements (xmin, xmax, ...) are an exact copy of
            this object's elements.
        """
        if not hasattr(self.__class__, "projection"):
            self._projection = self.DEFAULT_PROJ4
        geodict = {
            "xmin": self._xmin,
            "xmax": self._xmax,
            "ymin": self._ymin,
            "ymax": self._ymax,
            "dx": self._dx,
            "dy": self._dy,
            "ny": self._ny,
            "nx": self._nx,
            "projection": self._projection,
        }
        return GeoDict(geodict)

    def __eq__(self, other):
        """Check for equality between one GeoDict object and another.

        Args:
            other (GeoDict):
                Another GeoDict object.
        Returns:
            True when all GeoDict parameters are no more different than 1e-12, False
            otherwise.
        """
        if np.abs(self._xmin - other._xmin) > self.EPS:
            return False
        if np.abs(self._ymin - other.ymin) > self.EPS:
            return False
        if np.abs(self._xmax - other.xmax) > self.EPS:
            return False
        if np.abs(self._ymax - other.ymax) > self.EPS:
            return False
        if np.abs(self._dx - other.dx) > self.EPS:
            return False
        if np.abs(self._dy - other.dy) > self.EPS:
            return False
        if np.abs(self._ny - other.ny) > self.EPS:
            return False
        if np.abs(self._nx - other.nx) > self.EPS:
            return False
        return True

    def getLatLon(self, row, col):
        """Return geographic coordinates (lat/lon decimal degrees) for given data row
        and column.

        Args:
            row (int):
                Row dimension index into internal data array.
            col (int):
                Column dimension index into internal data array.
        Returns:
            Tuple of latitude and longitude.
        """
        scalar_types = (int, float)
        if type(row) != type(col):
            raise DataSetException("Input row/col types must be the same")

        if not isinstance(row, scalar_types):  # is this a sequence type thing?
            if isinstance(row, np.ndarray):
                if row.shape == () or col.shape == ():
                    raise DataSetException(
                        "Input row/col values must be scalars or dimensioned numpy "
                        "arrays."
                    )
            else:
                raise DataSetException(
                    "Input row/col values must be scalars or dimensioned numpy arrays."
                )

        ulx = self._xmin
        uly = self._ymax
        dx = self._dx
        dy = self._dy
        lon = ulx + col * dx
        lat = uly - row * dy
        return (lat, lon)

    def getRowCol(self, lat, lon, returnFloat=False, intMethod="round"):
        """Return data row and column from given geographic coordinates (lat/lon decimal
        degrees).

        Args:
            lat (int,float):
                Input latitude.
            lon (int,float):
                Input longitude.
            returnFloat (bool):
                indicating whether floating point row/col coordinates should be
                returned.
            intMethod (:obj:`str`, optional):
                String indicating the desired method by which floating point row/col
                values should be converted to integers.
                Choices are 'round' (default), 'floor', or 'ceil'.
        Returns:
            Tuple of row and column.
        """
        scalar_types = (int, float)
        if type(lat) != type(lon):
            raise DataSetException("Input lat/lon types must be the same")

        if not isinstance(lat, scalar_types):  # is this a sequence type thing?
            if isinstance(lat, np.ndarray):
                if lat.shape == () or lon.shape == ():
                    raise DataSetException(
                        "Input lat/lon values must be scalars or dimensioned numpy "
                        "arrays."
                    )
            else:
                raise DataSetException(
                    "Input lat/lon values must be scalars or dimensioned numpy arrays."
                )

        if intMethod not in ["round", "floor", "ceil"]:
            raise DataSetException("intMethod %s is not supported." % intMethod)

        ulx = self._xmin
        uly = self._ymax
        dx = self._dx
        dy = self._dy
        # check to see if we're in a scenario where the grid crosses the meridian
        if self._xmax < ulx and np.any(lon < 0):
            if not isinstance(lat, scalar_types):
                lon[lon < 0] += 360
            else:
                lon += 360
        col = (lon - ulx) / dx
        row = (uly - lat) / dy
        if returnFloat:
            return (row, col)
        if intMethod == "round":
            return (np.round(row).astype(int), np.round(col).astype(int))
        elif intMethod == "floor":
            return (np.floor(row).astype(int), np.floor(col).astype(int))
        else:
            return (np.ceil(row).astype(int), np.ceil(col).astype(int))

    # define setter and getter methods for all of the geodict parameters
    @property
    def xmin(self):
        """Get xmin value.

        Returns:
            xmin value.
        """
        return self._xmin

    @property
    def xmax(self):
        """Get xmin value.

        Returns:
            xmin value.
        """
        return self._xmax

    @property
    def ymin(self):
        """Get xmax value.

        Returns:
            xmax value.
        """
        return self._ymin

    @property
    def ymax(self):
        """Get xmax value.

        Returns:
            xmax value.
        """
        return self._ymax

    @property
    def dx(self):
        """Get dx value.

        Returns:
            dx value.
        """
        return self._dx

    @property
    def dy(self):
        """Get dy value.

        Returns:
            dy value.
        """
        return self._dy

    @property
    def ny(self):
        """Get ny value.

        Returns:
            ny value.
        """
        return self._ny

    @property
    def nx(self):
        """Get nx value.

        Returns:
            nx value.
        """
        return self._nx

    @property
    def projection(self):
        """Get projection Proj4 string.

        Returns:
            Valid Proj4 string.
        """
        return self._projection

    @property
    def nodata(self):
        """Get projection Proj4 string.

        Returns:
            Valid nodata value.
        """
        return self._nodata

    @xmin.setter
    def xmin(self, value):
        """Set xmin value, re-validate object.

        Args:
            value:
                Value to set.
        Raises:
            DataSetException:
                When validation fails.
        """
        self._xmin = value
        self.validate()

    @xmax.setter
    def xmax(self, value):
        """Set xmax value, re-validate object.

        Args:
            value:
                Value to set.
        Raises:
            DataSetException:
                When validation fails.
        """
        self._xmax = value
        self.validate()

    @ymin.setter
    def ymin(self, value):
        """Set ymin value, re-validate object.

        Args:
            value:
                Value to set.
        Raises:
            DataSetException:
                When validation fails.
        """
        self._ymin = value
        self.validate()

    @ymax.setter
    def ymax(self, value):
        """Set ymax value, re-validate object.

        Args:
            value:
                Value to set.
        Raises:
            DataSetException:
                When validation fails.
        """
        self._ymax = value
        self.validate()

    @dx.setter
    def dx(self, value):
        """Set dx value, re-validate object.

        Args:
            value:
                Value to set.
        Raises:
            DataSetException:
                When validation fails.
        """
        self._dx = value
        self.validate()

    @dy.setter
    def dy(self, value):
        """Set dy value, re-validate object.

        Args:
            value:
                Value to set.
        Raises:
            DataSetException:
                When validation fails.
        """
        self._dy = value
        self.validate()

    @ny.setter
    def ny(self, value):
        """Set ny value, re-validate object.

        Args:
            value:
                Value to set.
        Raises:
            DataSetException:
                When validation fails.
        """
        self._ny = value
        self.validate()

    @nx.setter
    def nx(self, value):
        """Set nx value, re-validate object.

        Args:
            value:
                Value to set.
        Raises:
            DataSetException:
                When validation fails.
        """
        self._nx = value
        self.validate()

    @nodata.setter
    def nodata(self, value):
        """Set nodata value.
        Args:
            value:
                Value to set.
        """
        self._nodata = value

    def getDeltas(self):
        # handle cases where we're crossing the meridian from the eastern hemisphere
        # to the western
        if self._xmin > self._xmax:
            txmax = self._xmax + 360.0
        else:
            txmax = self._xmax
        # try calculating xmax from xmin, dx, and nx
        xmax = self._xmin + self._dx * (self._nx - 1)
        # dxmax = np.abs(xmax - txmax)
        dxmax = np.abs(float(xmax) / txmax - 1.0)

        # try calculating dx from bounds and nx
        dx = np.abs((txmax - self._xmin) / (self._nx - 1))
        # ddx = np.abs((dx - self._dx))
        ddx = np.abs(float(dx) / self._dx - 1.0)

        # try calculating ymax from ymin, dy, and ny
        ymax = self._ymin + self._dy * (self._ny - 1)
        # dymax = np.abs(ymax - self._ymax)
        dymax = np.abs(float(ymax) / self._ymax - 1.0)

        # try calculating dx from bounds and nx
        dy = np.abs((self._ymax - self._ymin) / (self._ny - 1))
        # ddy = np.abs(dy - self._dy)
        ddy = np.abs(float(dy) / self._dy - 1.0)

        return (dxmax, ddx, dymax, ddy)

    def validate(self, adjust):
        # dxmax,ddx,dymax,ddy = self.getDeltas()

        if adjust == "bounds":
            self._xmax = self._xmin + self._dx * (self._nx - 1)
            self._ymin = self._ymax - self._dy * (self._ny - 1)
        elif adjust == "res":
            self._dx = (self._xmax - self._xmin) / (self._nx - 1)
            if self._dx < 0:
                if self._xmin > self._xmax:
                    self._dx = ((self._xmax + 360) - self._xmin) / (self._nx - 1)
            self._dy = (self._ymax - self._ymin) / (self._ny - 1)
        else:
            raise DataSetException('Unsupported adjust option "%s"' % adjust)
        if (self._xmax - self._xmin) < 359:  # TODO - think about this more!
            if self._xmax > 180:
                self._xmax -= 360
            if self._xmin < -180:
                self._xmin += 360
        # if self._xmin == 180.0 and self._xmax < 0:
        #     self._xmin = -180.0


def get_affine(src):
    if rasterio.__version__ < "1.0.0":
        affine = src.affine
    else:
        affine = src.transform
    return affine


def geodict_from_src(src):
    """Return a geodict from a dataset object.

    Args:
        src (DatasetReader):
            Open rasterio DatasetReader object.
    Returns:
        GeoDict: GeoDict describing the DatasetReader object.
    """
    affine = get_affine(src)
    nx = src.width
    ny = src.height
    return geodict_from_affine(affine, ny, nx)


def geodict_from_affine(affine, ny, nx):
    """**********

    Args:
        affine ():
            **********
        ny():
            Number of rows.
        nx():
            Number of columns.
    Returns:
        gd:
            ************
    """
    geodict = {}
    geodict["dx"] = affine.a
    geodict["dy"] = -1 * affine.e
    geodict["xmin"] = affine.xoff + geodict["dx"] / 2.0
    geodict["ymax"] = affine.yoff - geodict["dy"] / 2.0
    geodict["ny"] = ny
    geodict["nx"] = nx
    geodict["xmax"] = geodict["xmin"] + (geodict["nx"] - 1) * geodict["dx"]
    geodict["ymin"] = geodict["ymax"] - (geodict["ny"] - 1) * geodict["dy"]

    gd = GeoDict(geodict)
    return gd


def affine_from_geodict(geodict):
    """**************

    Args:
        geodict ():
            *****
    """
    xoff = geodict.xmin - geodict.dx / 2.0
    yoff = geodict.ymax + geodict.dy / 2.0
    xres = geodict.dx
    yres = -1 * geodict.dy
    src = Affine.translation(xoff, yoff) * Affine.scale(xres, yres)
    return src


def get_longitude_intersection(xmin, xmax, fxmin, fxmax):
    """**************

    Args:
        xmin():
            *****
        xmax():
            *****
        fxmin():
            *****
        fxmax():
            *****
    """
    # if we're straddling the 180 meridian, adjust the xmax value so that our
    # new range of longitudes is 0-360. We'll adjust back at the end
    if fxmin > fxmax:
        fxmax += 360
    if xmin > xmax:
        xmax += 360

    # get the intersected bounds
    # detect if either geodict spans the globe
    if (fxmax - fxmin) >= GLOBE_SPAN:
        txmin = xmin
        txmax = xmax
    elif (xmax - xmin) >= GLOBE_SPAN:
        txmin = fxmin
        txmax = fxmax
    else:  # neither geodict spans the globe
        txmin = max(fxmin, xmin)
        txmax = min(fxmax, xmax)

    if txmin > txmax:
        raise Exception("No longitude intersection")
    return txmin, txmax
