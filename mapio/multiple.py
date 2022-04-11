# python 3 compatibility
from __future__ import print_function

from .gridbase import Grid
from .dataset import DataSetException
from .grid2d import Grid2D
import abc
from collections import OrderedDict


class MultiGrid(Grid):
    def __init__(self, layers, descriptions=None):
        """
        Construct a semi-abstract MultiGrid object, which can contain many 2D layers of
        gridded data, all at the same resolution and with the same extent.

        Args:
            layers (OrderedDict):
                OrderedDict of Grid2D objects.
            descriptions (list):
                list of layer descriptions, or None
        Raises:
            DataSetException:
                When:
                    - length of descriptions (when not None) does not match length of
                    layers
                    - input layers is not an OrderedDict
        """
        if not isinstance(layers, OrderedDict):
            raise DataSetException("Input layers must be of type OrderedDict.")
        if descriptions is None:
            descriptions = ["" for ll in layers.keys()]
        if len(descriptions) != len(layers):
            raise DataSetException(
                "List of descriptions does not match length of layers."
            )

        lnames = list(layers.keys())
        self._layers = OrderedDict()
        self._descriptions = OrderedDict()
        for i in range(0, len(lnames)):
            layername = lnames[i]
            layer = layers[layername]
            desc = descriptions[i]
            geodict = layer.getGeoDict()
            self._layers[layername] = layer
            self._descriptions[layername] = desc
        self._geodict = geodict

    @abc.abstractmethod
    def save(self, filename):
        """
        Save layers of data to a file

        Args:
            filename (str):
                File to save data to.
        """
        raise NotImplementedError("save method not implemented in MultiGrid")

    # subclassed implementation should be a @classmethod
    @abc.abstractmethod
    def load(self, filename):
        """
        Load layers of data from a file.

        Args:
            filename (str):
                File to load data from.
        """
        raise NotImplementedError("save method not implemented in MultiGrid")

    def setLayer(self, name, data, desc=None):
        """
        Add a 2D layer of data to a MultiGrid object.

        Args:
            name (str):
                String which will be used to retrieve the data.
            data (array):
                2D numpy array of data
            desc (:obj:`str`, optional):
                Optional text description of layer
        Raises:
            DataSetException:
                If the data layer dimensions don't match the geodict.
        """
        nr, nc = data.shape
        if nr != self._geodict.ny or nc != self._geodict.nx:
            raise DataSetException(
                "Data layer dimensions don't match those already in the grid"
            )
        self._layers[name] = Grid2D(data, self._geodict.copy())
        self._descriptions[name] = desc

    def getLayer(self, name):
        """
        Retrieve the 2D associated with a layer name.

        Args:
            name (str):
                Name of data layer.
        Returns:
            Grid2D object.
        Raises:
            DataSetException:
                When name is not found in list of layer names.
        """
        if name not in self._layers.keys():
            raise DataSetException('Layer "%s" not in list of layers.' % name)
        return self._layers[name]

    def getData(self):
        """
        Return the OrderedDict of data layers contained in MultiGrid.

        Returns:
            OrderedDict of Grid2D objects.
        """
        return self._layers

    def setData(self, layers, descriptions=None):
        """
        Return the OrderedDict of data layers contained in MultiGrid.

        Args:
            layers (OrderedDict):
                OrderedDict of Grid2D objects.
        """
        self._layers = layers
        layernames = layers.keys()
        self._geodict = layers[layernames[0]].getGeoDict().copy()

    def getGeoDict(self):
        """
        Return the geodict object which defines the extent and resolution of all the
        grids.

        Returns:
            geodict dictionary (see constructor)
        """
        return self._geodict

    def getBounds(self):
        """
        Return the lat/lon range of the data.

        Returns:
            Tuple of (lonmin,lonmax,latmin,latmax)
        """
        return (
            self._geodict.xmin,
            self._geodict.xmax,
            self._geodict.ymin,
            self._geodict.ymax,
        )

    def trim(self, geodict, resample=False, method="linear"):
        """
        Trim all data layers to a smaller set of bounds, resampling if requested.  If
        not resampling, data will be trimmed to smallest grid boundary possible.

        Args:
            geodict (geodict):
                GeoDict used to specify subset bounds and resolution (if resample is
                selected)
            resample: (bool)
                Boolean indicating whether the data should be resampled to *exactly*
                match input bounds.
            method (:obj:`str`, optional):
                If resampling, method used, one of ('linear','nearest','cubic','quintic'
                )
        """
        for (layername, layer) in self._layers.items():
            layer.trim(geodict, resample=resample, method=method)
        self._geodict = layer.getGeoDict().copy()

    def getLayerNames(self):
        """
        Return the list of layer names contained in the MultiGrid.

        Returns:
            List of layer names.
        """
        return self._layers.keys()

    def getValue(self, lat, lon, layername, method="nearest", default=None):
        """Return numpy array at given latitude and longitude (using nearest neighbor).

        Args:
            lat (float):
                Latitude (in decimal degrees) of desired data value.
            lon (float):
                Longitude (in decimal degrees) of desired data value.
            layername (str):
                Name of layer from which to retrieve data.
            method (:obj:`str`, optional):
                Interpolation method, one of ('nearest','linear','cubic','quintic')
            default (:obj:`float`, optional):
                Default value to return when lat/lon is outside of grid bounds.
        Return:
           Value at input latitude,longitude position.
        """
        return self._layers[layername].getValue(
            lat, lon, method=method, default=default
        )

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
        layernames = self._layers.keys()
        return self._layers[layernames[0]].getLatLon(row, col)

    def getRowCol(self, lat, lon, returnFloat=False):
        """Return data row and column from given geographic coordinates (lat/lon
        decimal degrees).

        Args:
            lat (float):
                Input latitude.
            lon (float):
                Input longitude.
            returnFloat (bool):
                Boolean indicating whether floating point row/col coordinates should be
                returned.
        Returns:
            Tuple of row and column.
        """
        layernames = self._layers.keys()
        return self._layers[layernames[0]].getRowCol(lat, lon, returnFloat=returnFloat)

    def subdivide(self, finerdict, cellFill="max"):
        """Subdivide the cells of the host grid into finer-resolution cells.

        Args:
            finerdict (geodict):
                GeoDict object defining a grid with a finer resolution than the host
                grid.
            cellFill (:obj:`str`, optional):
                String defining how to fill cells that span more than one host grid
                cell.
                Choices are:
                    'max': Choose maximum value of host grid cells.
                    'min': Choose minimum value of host grid cells.
                    'mean': Choose mean value of host grid cells.
        Returns:
            MultiGrid instance with host grid values subdivided onto finer grid.
        Raises:
            DataSetException:
                When finerdict is not a) finer resolution or b) does not intersect.x or
                cellFill is not valid.
        """
        layers = OrderedDict()
        for (layername, layer) in self._layers.items():
            layers[layername] = layer.subdivide(finerdict, cellFill=cellFill)
        return MultiGrid(layers)

    def interpolateToGrid(self, geodict, method="linear"):
        """
        Given a geodict specifying another grid extent and resolution, resample all
        grids to match.

        Arggs:
            geodict (geodict):
                geodict dictionary from another grid whose extents are inside the
                extent of this grid.
            method (:obj:`str`, optional):
                Optional interpolation method - ['linear', 'cubic','quintic','nearest']
        Raises:
            DataSetException:
                If the Grid object upon which this function is being called is not
                completely contained by the grid to which this Grid is being resampled.
            DataSetException:
                If the resulting interpolated grid shape does not match input geodict.

        This function modifies the internal griddata and geodict object variables.
        """
        layers = OrderedDict()
        for (layername, layer) in self._layers.items():
            # layer.interpolateToGrid(geodict,method=method)
            layers[layername] = layer.interpolateToGrid(geodict, method=method)
        # self._geodict = layer.getGeoDict().copy()
        return MultiGrid(layers)
