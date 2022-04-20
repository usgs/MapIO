#!/usr/bin/env python

# python 3 compatibility
from __future__ import print_function

# stdlib imports
import abc


class DataSetException(Exception):
    """
    Class to represent errors in the DataSet class.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class DataSetWarning(Warning):
    """
    Class to represent warnings in the DataSet class.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class DataSet(object):
    # This should be a @classmethod in subclasses
    @abc.abstractmethod
    def load(filename, bounds=None, resample=False, padValue=None):
        """
        Load data into a Grid subclass.  Parameters below are suggested for subclasses.

        Args:
            filename:
                File where data is stored
            bounds:
                Optional tuple of (lonmin,lonmax,latmin,latmax) to subset data from
                file.
            resample:
                If subsetting data, True indicates that *exact* bounds are desired and
                data should be resampled to fit.
            padValue:
                If asking for data outside bounds of grid, any value not None means
                fill in those cells with padValue. None means don't pad the grid at all.
        Raises:
            NotImplementedError:
                Always for this base class.
        """
        raise NotImplementedError("load method not implemented in base class")

    # TODO: Figure out how format-specific attributes will be handled
    # (ShakeMap, for example)
    # This should be a @classmethod in subclasses
    @abc.abstractmethod
    def save(self, filename):  # would we ever want to save a subset of the data?
        """
        Save the data contained in the grid to a format specific file.
        Other attributes may be required for format specific files.

        Args:
            filename:
                Where file containing data should be written.
        """
        raise NotImplementedError("Save method not implemented in base class")

    @abc.abstractmethod
    def getData(self, getCopy=False):
        """
        Return a reference to or copy of the data inside the Grid

        Args:
            getCopy:
                True indicates that the user wants a copy of the data,
                not a reference to it.
        Returns:
            A reference to or copy of a numpy array of data.
        """
        raise NotImplementedError("getData method not implemented in base class")

    @abc.abstractmethod
    def setData(self, data):
        """
        Modify the data inside the Grid.

        Args:
            data:
                numpy array of desired data.
        """
        raise NotImplementedError("setData method not implemented in base class")

    @abc.abstractmethod
    def getBounds(self):
        """
        Return the lon/lat range of the data.

        Returns:
           Tuple of (lonmin,lonmax,latmin,latmax)
        """
        raise NotImplementedError("getBounds method not implemented in base class")

    @abc.abstractmethod
    def trim(self, geodict, resample=False, method="linear"):
        """
        Trim data to a smaller set of bounds, resampling if requested.
        If not resampling, data will be trimmed to smallest grid boundary possible.

        Args:
            geodict:
                GeoDict object used to specify subset bounds and resolution
                (if resample is selected)
            resample:
                Boolean indicating whether the data should be resampled to *exactly*
                match input bounds.
            method:
                If resampling, method used, one of
                ('linear','nearest','cubic','quintic')
        """
        raise NotImplementedError("trim method not implemented in base class")

    @abc.abstractmethod
    def getValue(
        self, lat, lon, method="nearest", default=None
    ):  # return nearest neighbor value
        """Return numpy array at given latitude and longitude (using given resampling method).

        Arg:
            lat:
                Latitude (in decimal degrees) of desired data value.
        Arg:
            lon:
                Longitude (in decimal degrees) of desired data value.
            method:
                Interpolation method, one of ('nearest','linear','cubic','quintic')
            default:
                Default value to return when lat/lon is outside of grid bounds.
        Returns:
           Value at input latitude,longitude position.
        """
        raise NotImplementedError("getValue method not implemented in base class")

    @abc.abstractmethod
    def interpolateToGrid(self, geodict, method="linear"):
        """
        Given a geodict specifying a grid extent and resolution, resample current data
        set to match.

        Args:
            geodict:
                geodict object from a grid whose extents are inside the extent of this
                grid.
            method:
                Optional interpolation method - ['linear', 'cubic','quintic','nearest']
        Returns:
            Interpolated grid.
        Raises:
            DataSetException:
                If the Grid object upon which this function is being called is not
                completely contained by the grid to which this Grid is being resampled.
            DataSetException:
                If the resulting interpolated grid shape does not match input geodict.

        This function modifies the internal griddata and geodict object variables.
        """
        raise NotImplementedError(
            "interpolateToGrid method not implemented in base class"
        )
