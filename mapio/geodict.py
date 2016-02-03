#!/usr/bin/env python

import numpy as np
from .dataset import DataSetException

class GeoDict(object):
    EPS = 1e-12
    REQ_KEYS = ['xmin','xmax','ymin','ymax','xdim','ydim','nrows','ncols']
    def __init__(self,geodict,preserve=None):
        """
        An object which represents the spatial information for a grid and is guaranteed to be self-consistent.

        :param geodict:
          A dictionary containing the following fields:
             - xmin Longitude minimum (decimal degrees) (Center of upper left cell)
             - xmax Longitude maximum (decimal degrees) (Center of upper right cell)
             - ymin Longitude minimum (decimal degrees) (Center of lower left cell)
             - ymax Longitude maximum (decimal degrees) (Center of lower right cell)
             - xdim Cell width (decimal degrees)
             - ydim Cell height (decimal degrees)
             - nrows Number of rows of input data (must match input data dimensions)
             - ncols Number of columns of input data (must match input data dimensions).
        :param preserve:
            String (one of None,'dims','shape','corner')
              None: All input parameters are assumed to be self-consistent, an exception will be raised if they are not.
              'dims': xdim/ydim and boundaries are assumed to be correct, number of rows and columns will be adjusted if required.
              'shape': rows/cols and boundaries are assumed to be correct, xdim and ydim will be adjusted if required.
              'corner': xmin,ymax,xdim,ydim,nrows,ncols are assumed to be correct, xmax and ymin will be adjusted if required.
        :raises DataSetException:
          When preserve is set to None, and any parameters are not self-consistent.
          When preserve is set to 'dims', and parameters are not self-consistent after adjusting number of rows/cols.
          When preserve is set to 'shape', and parameters are not self-consistent after adjusting xdim/ydim.
        """
        for key in self.REQ_KEYS:
            if key not in geodict.keys():
                raise DataSetException('Missing required key "%s" from input geodict dictionary' % key)
        
        self._xmin = geodict['xmin']
        self._xmax = geodict['xmax']
        self._ymin = geodict['ymin']
        self._ymax = geodict['ymax']
        self._xdim = geodict['xdim']
        self._ydim = geodict['ydim']
        self._nrows = geodict['nrows']
        self._ncols = geodict['ncols']
        self.preserve = preserve
        self.validate()

    def asDict(self):
        """Return GeoDict object in dictionary representation.
        :returns:
          Dictionary containing the same fields as found in constructor.
        """
        tdict = {}
        tdict['xmin'] = self._xmin
        tdict['xmax'] = self._xmax
        tdict['ymin'] = self._ymin
        tdict['ymax'] = self._ymax
        tdict['xdim'] = self._xdim
        tdict['ydim'] = self._ydim
        tdict['nrows'] = self._nrows
        tdict['ncols'] = self._ncols
        return tdict
        
        
    def __repr__(self):
        rfmt = '''Bounds: (%.4f,%.4f,%.4f,%.4f)\nDims: (%.4f,%.4f)\nShape: (%i,%i)'''
        rtpl = (self._xmin,self._xmax,self._ymin,self._ymax,self._xdim,self._ydim,self._nrows,self._ncols)
        return rfmt % rtpl
        
    def copy(self):
        geodict = {'xmin':self._xmin,
                   'xmax':self._xmax,
                   'ymin':self._ymin,
                   'ymax':self._ymax,
                   'xdim':self._xdim,
                   'ydim':self._ydim,
                   'nrows':self._nrows,
                   'ncols':self._ncols}
        return GeoDict(geodict)
        
    def __eq__(self,other):
        """Check for equality between one GeoDict object and another.

        :param other:
          Another GeoDict object.
        :returns:
          True when all GeoDict parameters are no more different than 1e-12, False otherwise.
        """
        if np.abs(self._xmin-other._xmin) > self.EPS:
            return False
        if np.abs(self._ymin-other.ymin) > self.EPS:
            return False
        if np.abs(self._xmax-other.xmax) > self.EPS:
            return False
        if np.abs(self._ymax-other.ymax) > self.EPS:
            return False
        if np.abs(self._xdim-other.xdim) > self.EPS:
            return False
        if np.abs(self._ydim-other.ydim) > self.EPS:
            return False
        if np.abs(self._nrows-other.nrows) > self.EPS:
            return False
        if np.abs(self._ncols-other.ncols) > self.EPS:
            return False
        return True

    def getLatLon(self,row,col):
        """Return geographic coordinates (lat/lon decimal degrees) for given data row and column.
        
        :param row: 
           Row dimension index into internal data array.
        :param col: 
           Column dimension index into internal data array.
        :returns: 
           Tuple of latitude and longitude.
        """
        ulx = self._xmin
        uly = self._ymax
        xdim = self._xdim
        ydim = self._ydim
        lon = ulx + col*xdim
        lat = uly - row*ydim
        return (lat,lon)

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
        ulx = self._xmin
        uly = self._ymax
        xdim = self._xdim
        ydim = self._ydim
        #check to see if we're in a scenario where the grid crosses the meridian
        if self._xmax < ulx and lon < ulx:
            lon += 360
        col = (lon-ulx)/xdim
        row = (uly-lat)/ydim
        if returnFloat:
            return (row,col)
        
        return (np.floor(row).astype(int),np.floor(col).astype(int))

    #define setter and getter methods for all of the geodict parameters
    @property
    def xmin(self):
        """Get xmin value.
        :returns:
          xmin value.
        """
        return self._xmin

    @property
    def xmax(self):
        """Get xmin value.
        :returns:
          xmin value.
        """
        return self._xmax

    @property
    def ymin(self):
        """Get xmax value.
        :returns:
          xmax value.
        """
        return self._ymin

    @property
    def ymax(self):
        """Get xmax value.
        :returns:
          xmax value.
        """
        return self._ymax

    @property
    def xdim(self):
        """Get xdim value.
        :returns:
          xdim value.
        """
        return self._xdim

    @property
    def ydim(self):
        """Get ydim value.
        :returns:
          ydim value.
        """
        return self._ydim

    @property
    def nrows(self):
        """Get nrows value.
        :returns:
          nrows value.
        """
        return self._nrows

    @property
    def ncols(self):
        """Get ncols value.
        :returns:
          ncols value.
        """
        return self._ncols

    @xmin.setter
    def xmin(self, value):
        """Set xmin value, re-validate object.
        :param value:
          Value to set.
        :raises DataSetException:
          When validation fails.
        """
        self._xmin = value
        self.validate()

    @xmax.setter
    def xmax(self, value):
        """Set xmax value, re-validate object.
        :param value:
          Value to set.
        :raises DataSetException:
          When validation fails.
        """
        self._xmax = value
        self.validate()

    @ymin.setter
    def ymin(self, value):
        """Set ymin value, re-validate object.
        :param value:
          Value to set.
        :raises DataSetException:
          When validation fails.
        """
        self._ymin = value
        self.validate()

    @ymax.setter
    def ymax(self, value):
        """Set ymax value, re-validate object.
        :param value:
          Value to set.
        :raises DataSetException:
          When validation fails.
        """
        self._ymax = value
        self.validate()

    @xdim.setter
    def xdim(self, value):
        """Set xdim value, re-validate object.
        :param value:
          Value to set.
        :raises DataSetException:
          When validation fails.
        """
        self._xdim = value
        self.validate()

    @ydim.setter
    def ydim(self, value):
        """Set ydim value, re-validate object.
        :param value:
          Value to set.
        :raises DataSetException:
          When validation fails.
        """
        self._ydim = value
        self.validate()

    @nrows.setter
    def nrows(self, value):
        """Set nrows value, re-validate object.
        :param value:
          Value to set.
        :raises DataSetException:
          When validation fails.
        """
        self._nrows = value
        self.validate()

    @ncols.setter
    def ncols(self, value):
        """Set ncols value, re-validate object.
        :param value:
          Value to set.
        :raises DataSetException:
          When validation fails.
        """
        self._ncols = value
        self.validate()

    def getDeltas(self):
        #handle cases where we're crossing the meridian from the eastern hemisphere to the western
        if self._xmin > self._xmax:
            txmax = self._xmax + 360
        else:
            txmax = self._xmax
        #try calculating xmax from xmin, xdim, and ncols
        xmax = self._xmin + self._xdim*(self._ncols-1)
        dxmax = np.abs(xmax - txmax)

        #try calculating xdim from bounds and ncols
        xdim = np.abs((txmax - self._xmin)/(self._ncols-1))
        dxdim = np.abs((xdim - self._xdim))

        #try calculating ymax from ymin, ydim, and nrows
        ymax = self._ymin + self._ydim*(self._nrows-1)
        dymax = np.abs(ymax - self._ymax)

        #try calculating xdim from bounds and ncols
        ydim = np.abs((self._ymax - self._ymin)/(self._nrows-1))
        dydim = np.abs(ydim - self._ydim)

        return (dxmax,dxdim,dymax,dydim)
        
    def validate(self):
        dxmax,dxdim,dymax,dydim = self.getDeltas()

        if self.preserve is None:
            if dxmax > self.EPS:
                raise DataSetException('GeoDict X resolution is not consistent with bounds and number of columns')
            if dxdim > self.EPS:
                raise DataSetException('GeoDict X resolution is not consistent with bounds and number of columns')
            if dymax > self.EPS:
                raise DataSetException('GeoDict Y resolution is not consistent with bounds and number of rows')
            if dydim > self.EPS:
                raise DataSetException('GeoDict Y resolution is not consistent with bounds and number of rows')
        elif self.preserve == 'dims':
            #sacrifice rows/cols (shape) in favor of preserving the dimensions and the bounds
            if dxdim > self.EPS:
                if self._xmin > self._xmax:
                    txmax = self._xmax + 360
                else:
                    txmax = self._xmax
                ncols = int(np.round(((txmax - self._xmin)/self._xdim)+1))
                xmax = self._xmin + self._xdim*(ncols-1)
                dxmax = np.abs(xmax - txmax)
                if dxmax > self.EPS:
                    raise DataSetException('Could not preserve bounds when changing shape in X dimension.')
                self._ncols = ncols
            if dydim > self.EPS:
                nrows = int(np.round(((self._ymax - self._ymin)/self._ydim)+1))
                ymax = self._ymin + self._ydim*(nrows-1)
                dymax = np.abs(ymax - self._ymax)
                if dymax > self.EPS:
                    raise DataSetException('Could not preserve bounds when changing shape in Y dimension.')
                self._nrows = nrows
        elif self.preserve == 'shape':
            #sacrifice dimensions in favor of rows/cols (shape) and boundaries
            if dxdim > self.EPS:
                xdim = ((self._xmax - self._xmin)/(self._ncols-1))
                xmax = self._xmin + xdim*(self._ncols-1)
                dxmax = np.abs(xmax - self._xmax)
                if dxmax > self.EPS:
                    raise DataSetException('Could not preserve bounds when changing shape in X dimension.')
                self._xdim = xdim
            if dydim > self.EPS:
                ydim = ((self._ymax - self._ymin)/(self._nrows-1))
                ymax = self._ymin + ydim*(self._nrows-1)
                dymax = np.abs(ymax - self._ymax)
                if dymax > self.EPS:
                    raise DataSetException('Could not preserve bounds when changing shape in Y dimension.')
                self._ydim = ydim
        elif self.preserve == 'corner':
            #assume that:
            #xmin/ymax are ok
            #xdim/ydim are ok
            #nrows/ncols are ok
            #xmax/ymin are NOT ok, recalculate them
            xmax = self._xmin + self._xdim*(self._ncols-1)
            dxmax = np.abs(xmax - self._xmax)
            if dxmax > self.EPS:
                self._xmax = xmax

            ymin = self._ymax - self._ydim*(self._nrows-1)
            dymin = np.abs(ymin - self._ymin)
            if dymin > self.EPS:
                self._ymin = ymin
        else:
            raise DataSetException('Unsupported preserve option "%s"' % preserve)

