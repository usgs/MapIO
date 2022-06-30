#!/usr/bin/env python

import abc
import numpy as np

# third party imports
from .dataset import DataSet
from .geodict import GeoDict


class Grid(DataSet):
    """
    An abstract class to represent lat/lon gridded datasets. Grids are
    assumed to be pixel-registered - that is, grid coordinates
    represent the value at the *center* of the cells.
    """

    @abc.abstractmethod  # should be a classmethod when instantiated
    def getFileGeoDict(filename):
        """
        Abstract method to return the bounding box, resolution, and shape of a file in
        whatever Grid format.

        Args:
            filename:
                The path to the filename of whatever grid format this is being
                implemented in.
        Returns:
            A geodict specifying the bounding box, resolution, and shape of the data in
            a file.
        """
        raise NotImplementedError

    @abc.abstractmethod  # should be a classmethod when instantiated
    def getBoundsWithin(filename, geodict):
        """
        Abstract method to return a geodict for this file that is guaranteed to be
        inside the input geodict defined, without resampling.

        Args:
            filename:
                The name of the file whose resolution/extent should be used.
            geodict:
                The geodict which is used as the base for finding the bounds for this
                file guaranteed to be inside of this geodict.
        Raises:
            NotImplementedError:
                Always in base class
        """
        raise NotImplementedError

    @classmethod
    def _getPadding(cls, geodict, paddict, padvalue):
        # get pad left columns - go outside specified bounds if not exact edge
        pxmin, pxmax, pymin, pymax = (
            paddict.xmin,
            paddict.xmax,
            paddict.ymin,
            paddict.ymax,
        )
        gxmin, gxmax, gymin, gymax = (
            geodict.xmin,
            geodict.xmax,
            geodict.ymin,
            geodict.ymax,
        )
        dx, dy = (geodict.dx, geodict.dy)
        ny, nx = (geodict.ny, geodict.nx)

        padleftcols = int(np.ceil((gxmin - pxmin) / dx))
        padrightcols = int(np.ceil((pxmax - gxmax) / dx))
        padbottomrows = int(np.ceil((gymin - pymin) / dy))
        padtoprows = int(np.ceil((pymax - gymax) / dy))

        # if any of these are negative, set them to zero
        if padleftcols < 0:
            padleftcols = 0
        if padrightcols < 0:
            padrightcols = 0
        if padbottomrows < 0:
            padbottomrows = 0
        if padtoprows < 0:
            padtoprows = 0

        leftpad = np.ones((ny, padleftcols)) * padvalue
        rightpad = np.ones((ny, padrightcols)) * padvalue
        nx += padrightcols + padleftcols
        bottompad = np.ones((padbottomrows, nx)) * padvalue
        toppad = np.ones((padtoprows, nx)) * padvalue

        # now figure out what the new bounds are
        outdict = {}
        outdict["nx"] = int(nx)
        outdict["ny"] = int(ny + bottompad.shape[0] + toppad.shape[0])

        outdict["xmin"] = gxmin - (padleftcols) * dx
        outdict["xmax"] = gxmax + (padrightcols) * dx
        outdict["ymin"] = gymin - (padbottomrows) * dy
        outdict["ymax"] = gymax + (padtoprows) * dy
        outdict["dx"] = dx
        outdict["dy"] = dy

        gd = GeoDict(outdict)
        return (leftpad, rightpad, bottompad, toppad, gd)

    @classmethod
    def checkGeoDict(cls, geodict):
        reqfields = set(["xmin", "xmax", "ymin", "ymax", "dx", "dy", "ny", "nx"])
        if not reqfields.issubset(set(geodict.keys())):
            return False
        return True

    @abc.abstractmethod
    def blockmean(self, geodict):
        """
        Abstract method to calculate average values for cells of larger size than the
        current grid.

        Args:
            geodict:
                Geodict that defines the new coarser grid.
        """
        raise NotImplementedError

    @abc.abstractmethod  # should be a classmethod when instantiated
    def loadFromCloud(cls, cloud, geodict):
        """
        Create a grid from a Cloud instance (scattered XY data).

        Args:
            cloud:
                A Cloud instance containing scattered XY data.
            geodict:
                A geodict object where ny/nx are optional (will be calculated from
                bounds/cell dimensions)
        Returns:
          An instance of a Grid object.
        """
        raise NotImplementedError

    @staticmethod
    def getLatLonMesh(geodict):
        if geodict.xmax < geodict.xmin:
            lons = np.linspace(geodict.xmin, geodict.xmax + 360, num=geodict.nx)
        else:
            lons = np.linspace(geodict.xmin, geodict.xmax, num=geodict.nx)
        lats = np.linspace(geodict.ymin, geodict.ymax, num=geodict.ny)
        lon, lat = np.meshgrid(lons, lats)
        return (lat, lon)

    @abc.abstractmethod
    def getGeoDict(self):
        """
        Return a reference to the geodict inside the Grid

        Returns:
            A reference to a dictionary (see constructor).
        """
        raise NotImplementedError("getGeoDict method not implemented in base class")

    @abc.abstractmethod
    def getLatLon(self, row, col):
        """Return geographic coordinates (lat/lon decimal degrees) for given data row
        and column.

        Args:
            row:
                Row dimension index into internal data array.
            col:
                Column dimension index into internal data array.
        Returns:
            Tuple of latitude and longitude.
        """
        raise NotImplementedError("getLatLon method not implemented in base class")

    @abc.abstractmethod
    def getRowCol(self, lat, lon, returnFloat=False):
        """Return data row and column from given geographic coordinates
        (lat/lon decimal degrees).

        Args:
            lat:
                Input latitude.
            lon:
                Input longitude.
            returnFloat:
                Boolean indicating whether floating point row/col coordinates
                should be returned.
        Returns:
            Tuple of row and column.
        """
        raise NotImplementedError("getRowCol method not implemented in base class")
