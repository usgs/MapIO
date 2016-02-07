#!/usr/bin/env python

import numpy as np
from .dataset import DataSetException

class GeoDict(object):
    EPS = 1e-12
    REQ_KEYS = ['xmin','xmax','ymin','ymax','dx','dy','ny','nx']
    def __init__(self,geodict,adjust=None):
        """
        An object which represents the spatial information for a grid and is guaranteed to be self-consistent.

        :param geodict:
          A dictionary containing the following fields:
             - xmin Longitude minimum (decimal degrees) (Center of upper left cell)
             - xmax Longitude maximum (decimal degrees) (Center of upper right cell)
             - ymin Longitude minimum (decimal degrees) (Center of lower left cell)
             - ymax Longitude maximum (decimal degrees) (Center of lower right cell)
             - dx Cell width (decimal degrees)
             - dy Cell height (decimal degrees)
             - ny Number of rows of input data (must match input data dimensions)
             - nx Number of columns of input data (must match input data dimensions).
        :param adjust:
            String (one of None,'bounds','res')
              None: All input parameters are assumed to be self-consistent, an exception will be raised if they are not.
              'bounds': dx/dy, nx/ny, xmin/ymax are assumed to be correct, xmax/ymin will be recalculated.
              'res': nx/ny, xmin/ymax, xmax/ymin and assumed to be correct, dx/dy will be recalculated.
        :raises DataSetException:
          When adjust is set to None, and any parameters are not self-consistent.
        """
        for key in self.REQ_KEYS:
            if key not in geodict.keys():
                raise DataSetException('Missing required key "%s" from input geodict dictionary' % key)
        
        self._xmin = geodict['xmin']
        self._xmax = geodict['xmax']
        self._ymin = geodict['ymin']
        self._ymax = geodict['ymax']
        self._dx = geodict['dx']
        self._dy = geodict['dy']
        self._ny = geodict['ny']
        self._nx = geodict['nx']
        self.validate(adjust=adjust)

    @classmethod
    def createDictFromBox(cls,xmin,xmax,ymin,ymax,dx,dy,inside=False):
        if xmin > xmax:
            txmax = xmax + 360
        else:
            txmax = xmax
        if inside:
            nx = np.floor(((txmax-xmin)/dx)+1)
            ny = np.floor(((ymax-ymin)/dy)+1)
            xmax2 = xmin + (nx-1)*dx
            ymin2 = ymax - (ny-1)*dx
        else:
            nx = np.ceil(((txmax-xmin)/dx)+1)
            ny = np.ceil(((ymax-ymin)/dy)+1)
        xmax2 = xmin + (nx-1)*dx
        ymin2 = ymax - (ny-1)*dx
        return cls({'xmin':xmin,'xmax':xmax2,
                    'ymin':ymin2,'ymax':ymax,
                    'dx':dx,'dy':dy,
                    'nx':nx,'ny':ny})

    @classmethod
    def createDictFromCenter(cls,cx,cy,dx,dy,xspan,yspan):
        xmin = cx - xspan/2.0
        xmax = cx + xspan/2.0
        ymin = cy - yspan/2.0
        ymax = cy + yspan/2.0
        return cls.createDictFromBox(xmin,xmax,ymin,ymax,dx,dy)

    def getBoundsWithin(self,geodict):
        fxmin,fxmax,fymin,fymax = (self.xmin,self.xmax,self.ymin,self.ymax)
        xmin,xmax,ymin,ymax = (geodict.xmin,geodict.xmax,geodict.ymin,geodict.ymax)
        fdx,fdy = (self.dx,self.dy)

        #find the nearest cell grid to xmin that is greater than xmin
        fleftcol = np.ceil((xmin-fxmin)/fdx)
        frightcol = np.floor((xmax-fxmin)/fdx)
        
        ftoprow = np.ceil((fymax-ymax)/fdy)
        fbottomrow = np.floor((fymax-ymin)/fdy)

        #these should all be on the host grid
        newxmin = fxmin + fleftcol*fdx
        newxmax = fxmin + frightcol*fdx
        newymin = fymax - fbottomrow*fdx
        newymax = fymax - ftoprow*fdx

        nx = round((newxmax-newxmin)/fdx + 1)
        ny = round((newymax-newymin)/fdy + 1)

        outgeodict = GeoDict({'xmin':newxmin,'xmax':newxmax,
                              'ymin':newymin,'ymax':newymax,
                              'dx':fdx,'dy':fdy,
                              'ny':ny,'nx':nx})
        return outgeodict
            
        
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
        tdict['dx'] = self._dx
        tdict['dy'] = self._dy
        tdict['ny'] = self._ny
        tdict['nx'] = self._nx
        return tdict
        
        
    def __repr__(self):
        rfmt = '''Bounds: (%.4f,%.4f,%.4f,%.4f)\nDims: (%.4f,%.4f)\nShape: (%i,%i)'''
        rtpl = (self._xmin,self._xmax,self._ymin,self._ymax,self._dx,self._dy,self._ny,self._nx)
        return rfmt % rtpl
        
    def copy(self):
        geodict = {'xmin':self._xmin,
                   'xmax':self._xmax,
                   'ymin':self._ymin,
                   'ymax':self._ymax,
                   'dx':self._dx,
                   'dy':self._dy,
                   'ny':self._ny,
                   'nx':self._nx}
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
        if np.abs(self._dx-other.dx) > self.EPS:
            return False
        if np.abs(self._dy-other.dy) > self.EPS:
            return False
        if np.abs(self._ny-other.ny) > self.EPS:
            return False
        if np.abs(self._nx-other.nx) > self.EPS:
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
        dx = self._dx
        dy = self._dy
        lon = ulx + col*dx
        lat = uly - row*dy
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
        dx = self._dx
        dy = self._dy
        #check to see if we're in a scenario where the grid crosses the meridian
        if self._xmax < ulx and lon < ulx:
            lon += 360
        col = (lon-ulx)/dx
        row = (uly-lat)/dy
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
    def dx(self):
        """Get dx value.
        :returns:
          dx value.
        """
        return self._dx

    @property
    def dy(self):
        """Get dy value.
        :returns:
          dy value.
        """
        return self._dy

    @property
    def ny(self):
        """Get ny value.
        :returns:
          ny value.
        """
        return self._ny

    @property
    def nx(self):
        """Get nx value.
        :returns:
          nx value.
        """
        return self._nx

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

    @dx.setter
    def dx(self, value):
        """Set dx value, re-validate object.
        :param value:
          Value to set.
        :raises DataSetException:
          When validation fails.
        """
        self._dx = value
        self.validate()

    @dy.setter
    def dy(self, value):
        """Set dy value, re-validate object.
        :param value:
          Value to set.
        :raises DataSetException:
          When validation fails.
        """
        self._dy = value
        self.validate()

    @ny.setter
    def ny(self, value):
        """Set ny value, re-validate object.
        :param value:
          Value to set.
        :raises DataSetException:
          When validation fails.
        """
        self._ny = value
        self.validate()

    @nx.setter
    def nx(self, value):
        """Set nx value, re-validate object.
        :param value:
          Value to set.
        :raises DataSetException:
          When validation fails.
        """
        self._nx = value
        self.validate()

    def getDeltas(self):
        #handle cases where we're crossing the meridian from the eastern hemisphere to the western
        if self._xmin > self._xmax:
            txmax = self._xmax + 360.0
        else:
            txmax = self._xmax
        #try calculating xmax from xmin, dx, and nx
        xmax = self._xmin + self._dx*(self._nx-1)
        #dxmax = np.abs(xmax - txmax)
        dxmax = np.abs(float(xmax)/txmax - 1.0)

        #try calculating dx from bounds and nx
        dx = np.abs((txmax - self._xmin)/(self._nx-1))
        #ddx = np.abs((dx - self._dx))
        ddx = np.abs(float(dx)/self._dx - 1.0)

        #try calculating ymax from ymin, dy, and ny
        ymax = self._ymin + self._dy*(self._ny-1)
        #dymax = np.abs(ymax - self._ymax)
        dymax = np.abs(float(ymax)/self._ymax - 1.0)

        #try calculating dx from bounds and nx
        dy = np.abs((self._ymax - self._ymin)/(self._ny-1))
        #ddy = np.abs(dy - self._dy)
        ddy = np.abs(float(dy)/self._dy - 1.0)

        return (dxmax,ddx,dymax,ddy)
        
    def validate(self,adjust=None):
        dxmax,ddx,dymax,ddy = self.getDeltas()

        if adjust is None:
            if dxmax > self.EPS:
                raise DataSetException('GeoDict X resolution is not consistent with bounds and number of columns')
            if ddx > self.EPS:
                raise DataSetException('GeoDict X resolution is not consistent with bounds and number of columns')
            if dymax > self.EPS:
                raise DataSetException('GeoDict Y resolution is not consistent with bounds and number of rows')
            if ddy > self.EPS:
                raise DataSetException('GeoDict Y resolution is not consistent with bounds and number of rows')
        elif adjust == 'bounds':
            if self._xmin > self._xmax:
                txmax = self._xmax + 360
            else:
                txmax = self._xmax
            self._xmax = self._xmin + self._dx*(self._nx-1)
            self._ymin = self._ymax - self._dy*(self._ny-1)
        elif adjust == 'res':
            self._dx = ((self._xmax - self._xmin)/(self._nx-1))
            self._dy = ((self._ymax - self._ymin)/(self._ny-1))
        else:
            raise DataSetException('Unsupported adjust option "%s"' % adjust)
        if self._xmax > 180:
            self._xmax -= 360

