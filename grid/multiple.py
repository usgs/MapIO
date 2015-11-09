from gridbase import Grid
from dataset import DataSetException
from grid2d import Grid2D
import abc
from collections import OrderedDict

class MultiGrid(Grid):
    reqfields = set(['xmin','xmax','ymin','ymax','xdim','ydim','ncols','nrows'])
    def __init__(self,layers,descriptions=None):
        """
        Construct a semi-abstract MultiGrid object, which can contain many 2D layers of gridded data, all at the 
        same resolution and with the same extent.

        :param layers: 
           OrderedDict of Grid2D objects.
        :param descriptions:
           list of layer descriptions, or None
        :raises DataSetException:
          When:
           - input layer geodicts does not contain required keys
           - input layer geodicts do not match each other.
           - length of descriptions (when not None) does not match length of layers
           - input layers is not an OrderedDict
        """
        if not isinstance(layers,OrderedDict):
            raise DataSetException('Input layers must be of type OrderedDict.')
        if descriptions is None:
            descriptions = ['' for l in layers.keys()]
        if len(descriptions) != len(layers):
            raise DataSetException('List of descriptions does not match length of layers.')

        lnames = layers.keys()
        for i in range(0,len(lnames)):
            layername = lnames[i]
            layer = layers[layername]
            desc = descriptions[i]
            geodict = layer.getGeoDict()
            if not set(geodict.keys()).issuperset(self.reqfields):
                missing = self.reqfields - set(geodict.keys())
                raise DataSetException('Missing required keys "%s"' % str(missing))
            self._layers[layername] = layer
            self._descriptions[layername] = desc

    @abc.abstractmethod
    def save(self,filename):
        """
        Save layers of data to a file
        
        :param filename:
          File to save data to.
        """
        raise NotImplementedError("save method not implemented in MultiGrid")

    #subclassed implementation should be a @classmethod
    @abc.abstractmethod
    def load(self,filename):
        """
        Load layers of data from a file
        :param filename:
          File to load data from.
        """
        raise NotImplementedError("save method not implemented in MultiGrid")
        
    def setLayer(self,name,data,desc=None):
        """
        Add a 2D layer of data to a MultiGrid object.

        :param name: 
          String which will be used to retrieve the data.
        :param data:
          2D numpy array of data
        :param desc:
          Optional text description of layer
        :raises DataSetException:
          If the data layer dimensions don't match the geodict.
        """
        nr,nc = data.shape
        if nr != self._geodict['nrows'] or nc != self._geodict['ncols']:
            raise DataSetException("Data layer dimensions don't match those already in the grid")
        self._layers[name] = Grid2D(data,self._geodict.copy())
        self._descriptions[name] = desc

    def getLayer(self,name):
        """
        Retrieve the 2D associated with a layer name.

        :param name:
          Name of data layer.
        :returns:
          Grid2D object.
        :raises DataSetException:
          When name is not found in list of layer names.
        """
        if name not in self._layers.keys():
            raise DataSetException('Layer "%s" not in list of layers.' % name)
        return self._layers[name]

    def getData(self):
        """
        Return the OrderedDict of data layers contained in MultiGrid.

        :returns:
          OrderedDict of Grid2D objects.
        """
        return self._layers

    def setData(self,layers,descriptions=None):
        """
        Return the OrderedDict of data layers contained in MultiGrid.

        :param layers:
          OrderedDict of Grid2D objects.
        """
        self._layers = layers
        layernames = layers.keys()
        self._geodict = layers[layernames[0]].getGeoDict().copy()

    def getGeoDict(self):
        """
        Return the geodict object which defines the extent and resolution of all the grids.
        
        :returns:
          geodict dictionary (see constructor)
        """
        return self._geodict

    def getBounds(self):
        """
        Return the lat/lon range of the data.
        
        :returns:
          Tuple of (lonmin,lonmax,latmin,latmax)
        """
        return (self._geodict['xmin'],self._geodict['xmax'],self._geodict['ymin'],self._geodict['ymax'])

    def trim(self,bounds,resample=False,method='linear'):
        """
        Trim all data layers to a smaller set of bounds, resampling if requested.  If not resampling,
        data will be trimmed to smallest grid boundary possible.
        
        :param bounds:
           Tuple of (lonmin,lonmax,latmin,latmax)
        :param resample:
           Boolean indicating whether the data should be resampled to *exactly* match input bounds.
        :param method:
           If resampling, method used, one of ('linear','nearest','cubic','quintic')
        """
        for layername,layer in self._layers.iteritems():
            layer.trim(bounds,resample=resample,method=method)
        self._geodict = layer.getGeoDict().copy()

    def getLayerNames(self):
        """
        Return the list of layer names contained in the MultiGrid.
        :returns:
          List of layer names.
        """
        return self._layers.keys()
        
    def getValue(self,lat,lon,layername,method='nearest',default=None):
        """Return numpy array at given latitude and longitude (using nearest neighbor).
        
        :param lat: 
           Latitude (in decimal degrees) of desired data value.
        :param lon: 
           Longitude (in decimal degrees) of desired data value.
        :param layername:
          Name of layer from which to retrieve data.
        :param method:
           Interpolation method, one of ('nearest','linear','cubic','quintic')
        :param default:
           Default value to return when lat/lon is outside of grid bounds.
        :return: 
           Value at input latitude,longitude position.
        """
        return self._layers[layername].getValue(lat,lon,method=method,default=default)

    def getLatLon(self,row,col):
        """Return geographic coordinates (lat/lon decimal degrees) for given data row and column.
        
        :param row: 
           Row dimension index into internal data array.
        :param col: 
           Column dimension index into internal data array.
        :returns: 
           Tuple of latitude and longitude.
        """
        layernames = self._layers.keys()
        return self._layers[layernames[0]].getLatLon(row,col)

    def getRowCol(self,lat,lon,returnFloat=False):
        """Return data row and column from given geographic coordinates (lat/lon decimal degrees).
        
        :param lat: 
           Input latitude.
        :param lon: 
           Input longitude.
        :param returnFloat: 
           Boolean indicating whether floating point row/col coordinates should be returned.
        :returns: 
           Tuple of row and column.
        """
        layernames = self._layers.keys()
        return self._layers[layernames[0]].getRowCol(lat,lon,returnFloat=returnFloat)

    def interpolateToGrid(self,geodict,method='linear'):
        """
        Given a geodict specifying another grid extent and resolution, resample all grids to match.
        
        :param geodict: 
            geodict dictionary from another grid whose extents are inside the extent of this grid.
        :keyword method: 
            Optional interpolation method - ['linear', 'cubic','quintic','nearest']
        :raises DataSetException: 
           If the Grid object upon which this function is being called is not completely contained by the grid to which this Grid is being resampled.
        :raises DataSetException: 
           If the resulting interpolated grid shape does not match input geodict.

        This function modifies the internal griddata and geodict object variables.
        """
        for layername,layer in self._layers.iteritems():
            layer.interpolateToGrid(geodict,method=method)
        self._geodict = layer.getGeoDict().copy()

    
