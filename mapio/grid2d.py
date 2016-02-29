#!/usr/bin/env python

#python 3 compatibility
from __future__ import print_function

#stdlib imports
import abc
import textwrap
import glob
import os
import sys

#third party imports
from .gridbase import Grid
from .dataset import DataSetException
from .geodict import GeoDict

import numpy as np
from scipy import interpolate
import shapely
from affine import Affine
from rasterio import features
from shapely.geometry import MultiPoint,Polygon,mapping

class Grid2D(Grid):
    reqfields = set(['xmin','xmax','ymin','ymax','dx','dy','nx','ny'])
    def __init__(self,data=None,geodict=None):
        """
        Construct a Grid object.
        
        :param data: 
            A 2D numpy array (can be None).
        :param geodict: 
            A GeoDict Object (or None) containing the following fields:
        :returns:
            A Grid2D object.
        :raises DataSetException:
           When data is not 2D or the number of rows and columns do not match the geodict.
        """
        if data is not None and geodict is not None:
            #complain if data is not 2D (squeeze 1d dimensions out)
            dims = data.shape
            if len(dims) != 2:
                raise DataSetException('Grid data must be 2D.  Input data has shape of %s' % str(data.shape))
            ny,nx = dims
            if ny != geodict.ny or nx != geodict.nx:
                raise DataSetException('Input geodict does not match shape of input data.')
            self._geodict = geodict.copy()
            self._data = data.copy()
        else:
            self._data = None
            self._geodict = None

    def __repr__(self):
        """
        String representation of a Grid2D object.
        :returns:
          String containing description of Grid2D object.
        """
        xmin,xmax,ymin,ymax = (self._geodict.xmin,self._geodict.xmax,
                               self._geodict.ymin,self._geodict.ymax)
        ny,nx = self._data.shape
        dx,dy = (self._geodict.dx,self._geodict.dy)
        zmin = np.nanmin(self._data)
        zmax = np.nanmax(self._data)
        rstr = '''<%s Object:
        ny: %i
        nx: %i
        xmin: %.4f
        xmax: %.4f
        ymin: %.4f
        ymax: %.4f
        dx: %.4f
        dy: %.4f
        zmin: %.6f
        zmax: %.6f>''' % (self.__class__.__name__,ny,nx,xmin,xmax,ymin,ymax,dx,dy,zmin,zmax)
        parts = rstr.split('\n')
        newrstr = '\n'.join([p.strip() for p in parts])
        return textwrap.dedent(newrstr)

    @classmethod
    def _createSampleData(self,M,N):
        """Used for internal testing - create an NxN grid with lower left corner at 0.5,0.5, dx/dy = 1.0.
        :param M:
           Number of rows in output grid
        :param N:
           Number of columns in output grid
        :returns:
           GMTGrid object where data values are an NxN array of values from 0 to N-squared minus 1, and geodict
           lower left corner is at 0.5/0.5 and cell dimensions are 1.0.
        """
        data = np.arange(0,M*N).reshape(M,N)
        data = data.astype(np.int32) #arange gives int64 by default, not supported by netcdf3
        xvar = np.arange(0.5,0.5+N,1.0)
        yvar = np.arange(0.5,0.5+M,1.0)
        geodict = {'ny':M,
                   'nx':N,
                   'xmin':0.5,
                   'xmax':xvar[-1],
                   'ymin':0.5,
                   'ymax':yvar[-1],
                   'dx':1.0,
                   'dy':1.0}
        gd = GeoDict(geodict)
        return (data,gd)
    
    #This should be a @classmethod in subclasses
    @abc.abstractmethod
    def load(filename,bounds=None,resample=False,padValue=None):
        raise NotImplementedError('Load method not implemented in base class')

    @classmethod
    def copyFromGrid(cls,grid):
        """
        Copy constructor - can be used to create an instance of any Grid2D subclass from another.

        :param grid:
          Any Grid2D instance.
        :returns:
          A copy of the data in that input Grid2D instance.
        """
        if not isinstance(grid,Grid2D):
            raise DataSetException('Input to copyFromGrid must be an instance of a Grid2D object (inc. subclasses)')
        return cls(grid.getData(),grid.getGeoDict())

    #This should be a @classmethod in subclasses
    @abc.abstractmethod
    def save(self,filename): #would we ever want to save a subset of the data?
        raise NotImplementedError('Save method not implemented in base class')
    
    @classmethod
    def _createSections(self,bounds,geodict,firstColumnDuplicated,isScanLine=False):
        """Given a grid that goes from -180 to 180 degrees, figure out the two pixel regions that up both sides of the subset.
        :param bounds:
           Tuple of (xmin,xmax,ymin,ymax)
        :param geodict:
           Geodict dictionary
        :param firstColumnDuplicated:
          Boolean indicating whether this is a global data set where the first and last columns are identical.
        :param isScanLine:
          Boolean indicating whether this array is in scan line order (pixel[0,0] is the geographic upper left).
        :returns:
          Two tuples of 4 elements each - (iulx,iuly,ilrx,ilry). The first tuple defines the pixel offsets for the left
          side of the subsetted region, and the second tuple defines the pixel offsets for the right side.
        """
        (bxmin,bxmax,bymin,bymax) = bounds
        ulx = geodict.xmin
        uly = geodict.ymax
        dx = geodict.dx
        dy = geodict.dy
        nx = geodict.nx
        ny = geodict.ny
        #section 1
        iulx1 = int(np.floor((bxmin - ulx)/dx))
        ilrx1 = int(nx)

        if not isScanLine:
            iuly1 = int(np.ceil((uly - bymax)/dy))
            ilry1 = int(np.floor((uly - bymin)/dy)) + 1
        else:
            ilry1 = int(np.ceil((uly - bymin)/dy))
            iuly1 = int(np.floor((uly - bymax)/dy)) + 1

        #section 2
        iulx2 = 0
        ilrx2 = int(np.ceil((bxmax - ulx)/dx)) + 1
        iuly2 = iuly1
        ilry2 = ilry1

        if firstColumnDuplicated:
            ilrx1 -= 1

        region1 = (iulx1,iuly1,ilrx1,ilry1)
        region2 = (iulx2,iuly2,ilrx2,ilry2)
        return(region1,region2)
    
    def getData(self):
        """Return a reference to the data inside the Grid.
        
        :returns: 
          A reference to a 2D numpy array.
        """
        return self._data #should we return a copy of the data?

    def setData(self,data):
        """Set the data inside the Grid.
        
        :param data: 
          A 2D numpy array.
        :raises:
          DataSetException if the number of rows and columns do not match the the internal GeoDict, or if the input
          is not a numpy array.
        """
        if not isinstance(data,np.ndarray):
            raise DataSetException('setData() input is not a numpy array.')

        if len(data.shape) != 2:
            raise DataSetException('setData() input array must have two dimensions.')

        m,n = data.shape
        if m != self._geodict.ny or n != self._geodict.nx:
            raise DataSetException('setData() input array must match rows and columns of existing data.')
        self._data = data

    def getGeoDict(self):
        """
        Return a reference to the geodict inside the Grid.
        :returns:
          A reference to a dictionary (see constructor).
        """
        return self._geodict.copy()

    def getBounds(self):
        """
        Return the lon/lat range of the data.
        
        :returns:
           Tuple of (lonmin,lonmax,latmin,latmax)
        """
        return (self._geodict.xmin,self._geodict.xmax,self._geodict.ymin,self._geodict.ymax)

    def cut(self,xmin,xmax,ymin,ymax):
        """Cut out a section of Grid and return it.

        :param xmin: Longitude coordinate of upper left pixel, must be aligned with Grid.
        :param xmax: Longitude coordinate of lower right pixel, must be aligned with Grid.
        :param ymin: Latitude coordinate of upper left pixel, must be aligned with Grid.
        :param ymax: Latitude coordinate of lower right pixel, must be aligned with Grid.
        """
        td = GeoDict.createDictFromBox(xmin,xmax,ymin,ymax,self._geodict.dx,self._geodict.dy,inside=True)
        if not td.isAligned(self._geodict):
            raise DataSetException('Input bounds must align with this grid.')
        if not self._geodict.contains(td):
            raise DataSetException('Input bounds must be completely contained by this grid.')
        uly,ulx = self._geodict.getRowCol(ymax,xmin)
        lry,lrx = self._geodict.getRowCol(ymin,xmax)
        data = self._data[uly:lry+1,ulx:lrx+1]
        grid = Grid2D(data,td)
        return grid
    
    # def trim(self,geodict,resample=False,method='linear'):
    #     """
    #     Trim data to a smaller set of bounds, resampling if requested.  If not resampling,
    #     data will be trimmed to largest grid boundary possible that fits inside input bounds.
        
    #     :param geodict:
    #        GeoDict used to specify subset bounds and resolution (if resample is selected)
    #     :param resample:
    #        Boolean indicating whether the data should be resampled to *exactly* match input bounds.
    #     :param method:
    #        If resampling, method used, one of ('linear','nearest','cubic','quintic')
    #     """
    #     xmin,xmax,ymin,ymax = (geodict.xmin,geodict.xmax,geodict.ymin,geodict.ymax)
    #     gxmin,gxmax,gymin,gymax = self.getBounds()
    #     fdx,fdy = (self._geodict.dx,self._geodict.dy)
    #     #if any of the input bounds are outside the bounds of the grid, cut off those edges
    #     xmin = max(xmin,gxmin)
    #     xmax = min(xmax,gxmax)
    #     ymin = max(ymin,gymin)
    #     ymax = min(ymax,gymax)
    #     if not resample:
    #         sampledict = self._geodict.getBoundsWithin(geodict)
    #         #sampledict = GeoDict.createDictFromBox(xmin,xmax,ymin,ymax,fdx,fdy,inside=True)
    #         iuly,iulx = self._geodict.getRowCol(sampledict.ymax,sampledict.xmin)
    #         ilry,ilrx = self._geodict.getRowCol(sampledict.ymin,sampledict.xmax)
    #         self._data = self._data[iuly:ilry+1,iulx:ilrx+1]
    #         self._geodict = sampledict
    #     else:
    #         self.interpolateToGrid(geodict,method=method)

    def getValue(self,lat,lon,method='nearest',default=None): #return nearest neighbor value
        """Return numpy array at given latitude and longitude (using nearest neighbor).
        
        :param lat: 
           Latitude (in decimal degrees) of desired data value.
        :param lon: 
           Longitude (in decimal degrees) of desired data value.
        :param method:
           Interpolation method, one of ('nearest','linear','cubic','quintic')
        :param default:
           Default value to return when lat/lon is outside of grid bounds.
        :return: 
           Value at input latitude,longitude position.
        """
        if method == 'nearest':
            row,col = self.getRowCol(lat,lon)
        else:
            row,col = self.getRowCol(lat,lon,returnFloat=True)
        ny,nx = self._data.shape
        outidx = np.where((row < 0) | (row > ny-1) | (col < 0) | (col > nx-1))[0]
        inidx = np.where((row >= 0) & (row <= ny-1) & (col >= 0) & (col <= nx-1))[0]
        value = np.ones_like(lat)
        if len(outidx):
            if default is None:
                msg = 'One of more of your lat/lon values is outside Grid boundaries: %s' % (str(self.getBounds()))
                raise DataSetException(msg)
            value[outidx] = default
        if method == 'nearest':
            value[inidx] = self._data[row[inidx],col[inidx]]
            return value
        else:
            raise NotImplementedError('getValue method "%s" not implemented yet' % method)

    def getLatLon(self,row,col):
        """Return geographic coordinates (lat/lon decimal degrees) for given data row and column.
        
        :param row: 
           Row dimension index into internal data array.
        :param col: 
           Column dimension index into internal data array.
        :returns: 
           Tuple of latitude and longitude.
        """
        return self._geodict.getLatLon(row,col)

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
        return self._geodict.getRowCol(lat,lon,returnFloat)

    def _getInterpCoords(self,geodict):
        #translate geographic coordinates to 2 1-D arrays of X and Y pixel coordinates
        #remember that pixel coordinates are (0,0) at the top left and increase going down and to the right
        #geographic coordinates are (xmin,ymin) at the bottom left and increase going up and to the right
        dims = self._data.shape
        ny1 = self._geodict.ny
        nx1 = self._geodict.nx

        #handle meridian crossing grids
        if self._geodict.xmin > self._geodict.xmax:
            xmax1 = self._geodict.xmax + 360
        else:
            xmax1 = self._geodict.xmax
        xmin1 = self._geodict.xmin
        
        ymin1 = self._geodict.ymin
        ymax1 = self._geodict.ymax
        dx1 = self._geodict.dx
        dy1 = self._geodict.dy
        
        #extract the geographic information about the grid we're sampling to
        ny = geodict.ny
        nx = geodict.nx

        #handle meridian crossing grids
        if geodict.xmin > geodict.xmax:
            xmax = geodict.xmax + 360
        else:
            xmax = geodict.xmax
        xmin = geodict.xmin

        ymin = geodict.ymin
        ymax = geodict.ymax
        dx = geodict.dx
        dy = geodict.dy

        #make sure that the grid we're resampling TO is completely contained by our current grid
        if xmin1 > xmin or xmax1 < xmax or ymin1 > ymin or ymax1 < ymax:
            raise DataSetException('Grid you are resampling TO is not completely contained by base grid.')
        
        gxi = np.linspace(xmin,xmax,num=nx)
        gyi = np.linspace(ymin,ymax,num=ny)
        
        #we need to handle the meridian crossing here...
        if xmin > xmax:
            xmax += 360
            xmin1 += 360

        xi = (gxi - xmin1)/dx1
        #yi = (gyi - ymin1)/dy1
        yi = np.array(sorted(((ymax1 - gyi)/dy1)))

        return (xi,yi)
    
    def interpolateToGrid(self,geodict,method='linear'):
        """
        Given a geodict specifying another grid extent and resolution, resample current grid to match.
        
        :param geodict: 
            geodict dictionary from another grid whose extents are inside the extent of this grid.
        :param method: 
            Optional interpolation method - ['linear', 'cubic','nearest']
        :raises DataSetException: 
           If the Grid object upon which this function is being called is not completely contained by the grid to which this Grid is being resampled.
        :raises DataSetException: 
           If the method is not one of ['nearest','linear','cubic']
           If the resulting interpolated grid shape does not match input geodict.
        This function modifies the internal griddata and geodict object variables.
        """
        if method not in ['linear', 'cubic','nearest']:
            raise DataSetException('Resampling method must be one of "linear", "cubic","nearest"')
        bounds = (geodict.xmin,geodict.xmax,geodict.ymin,geodict.ymax)
        dx,dy = (geodict.dx,geodict.dy)
        ny,nx = (geodict.ny,geodict.nx)
        xi,yi = self._getInterpCoords(geodict)

        #now using scipy interpolate functions
        baserows,basecols = self._geodict.ny,self._geodict.nx
        basex = np.arange(0,basecols) #base grid PIXEL coordinates
        basey = np.arange(0,baserows)
        if method in ['linear','cubic']:
            if not np.isnan(self._data).any():
                #at the time of this writing, interp2d does not support NaN values at all.
                f = interpolate.interp2d(basex,basey,self._data,kind=method)
                self._data = f(xi,yi)
            else:
                #is Nan values are present, use griddata (slower by ~2 orders of magnitude but supports NaN).
                xi,yi = np.meshgrid(xi,yi)
                newrows,newcols = xi.shape
                xi = xi.flatten()
                yi = yi.flatten()
                xnew = np.zeros((len(xi),2))
                xnew[:,0] = xi
                xnew[:,1] = yi
                basex,basey = np.meshgrid(basex,basey)
                basex = basex.flatten()
                basey = basey.flatten()
                xold = np.zeros((len(basex),2))
                xold[:,0] = basex
                xold[:,1] = basey
                self._data = interpolate.griddata(xold,self._data.flatten(),xnew,method=method)
                self._data = self._data.reshape((newrows,newcols))
        else:
            x,y = np.meshgrid(basex,basey)
            #in Python2, list doesn't do anything
            #in python3, it makes result of zip from iterator into list
            xy = list(zip(x.flatten(),y.flatten())) 
            f = interpolate.NearestNDInterpolator(xy,self._data.flatten())
            newrows = geodict.ny
            newcols = geodict.nx
            xi = np.tile(xi,(newrows,1))
            yi = np.tile(yi.reshape(newrows,1),(1,newcols))
            self._data = f(list(zip(xi.flatten(),yi.flatten())))
            self._data = self._data.reshape(xi.shape)
                                                  
            
        ny,nx = geodict.ny,geodict.nx
        dims = self._data.shape
        ny_new = dims[0]
        nx_new = dims[1]
        if ny_new != ny or nx_new != nx:
            msg = "Interpolation failed!  Results (%i,%i) don't match (%i,%i)!" % (ny_new,nx_new,ny,nx)
            raise DataSetException(msg)
        #now the extents and resolution of the two grids should be identical...
        gdict = {'ny':geodict.ny,
                 'nx':geodict.nx,
                 'xmin':geodict.xmin,
                 'xmax':geodict.xmax,
                 'ymin':geodict.ymin,
                 'ymax':geodict.ymax,
                 'dx':geodict.dx,
                 'dy':geodict.dy}
        self._geodict = GeoDict(gdict)

    @classmethod
    def rasterizeFromGeometry(cls,shapes,geodict,burnValue=1.0,fillValue=np.nan,
                              mustContainCenter=False,attribute=None):
        """
        Create a Grid2D object from vector shapes, where the presence of a shape 
        (point, line, polygon) inside a cell turns that cell "on".
        
        :param shapes:
          One of:
            - One shapely geometry object (Point, Polygon, etc.) or a sequence of such objects
            - One GeoJSON like object or sequence of such objects. (http://geojson.org/)
            - A tuple of (geometry,value) or sequence of (geometry,value).
        :param geodict:
          GeoDict object which defines the grid onto which the shape values should be "burned".
        :param burnValue:
          Optional value which will be used to set the value of the pixels if there is no 
          value in the geometry field.
        :param fillValue:
          Optional value which will be used to fill the cells not touched by any geometry.
        :param mustContainCenter:
          Optional boolean which indicates whether the geometry must touch
          the center of the cell or merely be inside the cell in order to set the value.
        :raises DataSetException:
          When geometry input is not a subclass of shapely.geometry.base.BaseGeometry.
        :returns:
          Grid2D object.
        This method is a thin wrapper around rasterio->features->rasterize(), documented here:
        https://github.com/mapbox/rasterio/blob/master/docs/features.rst

        which is itself a Python wrapper around the functionality found in gdal_rasterize, documented here:
        http://www.gdal.org/gdal_rasterize.html
        """
        #check the type of shapes
        #features.rasterize() documentation says this:
        #iterable of (geometry, value) pairs or iterable over
        #geometries. `geometry` can either be an object that implements
        #the geo interface or GeoJSON-like object.

        #create list of allowable types
        if sys.version_info.major == 2:
            types = (int,float,long)
        else:
            types = (int,float)
        
        #figure out whether this is a single shape or a sequence of shapes
        isGeoJSON = False
        isGeometry = False
        isSequence = False
        isTuple = False
        if hasattr(shapes, '__iter__'):
            if isinstance(shapes[0],tuple):
                isTuple = True
        isOk = False
        isShape = False
        if isinstance(shapes,shapely.geometry.base.BaseGeometry):
            isOk = True
            isShape = True
        elif len(shapes) and isinstance(shapes[0],shapely.geometry.base.BaseGeometry):
            isOk = True
            isShape = True
        elif isinstance(shapes,dict) and 'geometry' in shapes and 'properties' in shapes:
            isOk = True
        elif len(shapes) and isinstance(shapes[0],dict) and 'geometry' in shapes[0] and 'properties' in shapes[0]:
            isOk = True
        else:
            pass
        if not isOk:
            raise DataSetException('shapes must be a single shapely object or sequence of them, or single Geo-JSON like-object')

        if not isShape:
            shapes2 = []
            for shape in shapes:
                geometry = shape['geometry']
                props = shape['properties']
                if attribute is not None:
                    if not attribute in props:
                        raise DataSetException('Input shapes do not have attribute "%s".' % attribute)
                    value = props[attribute]
                    if not isinstance(value,types):
                        raise DataSetException('value from input shapes object is not a number')
                else:
                    value = burnValue
                shapes2.append((geometry,value))
            shapes = shapes2
        
                                   
        xmin,xmax,ymin,ymax = (geodict.xmin,geodict.xmax,geodict.ymin,geodict.ymax)
        dx,dy = (geodict.dx,geodict.dy)

        xvar = np.arange(xmin,xmax+(dx*0.1),dx)
        yvar = np.arange(ymin,ymax+(dy*0.1),dy)
        nx = len(xvar)
        ny = len(yvar)
        
        #the rasterize function assumes a pixel registered data set, where we are grid registered.  In order to make this work
        #we need to adjust the edges of our grid out by half a cell width in each direction.  
        txmin = xmin - dx/2.0
        tymax = ymax + dy/2.0
        
        outshape = (ny,nx)
        transform = Affine.from_gdal(txmin,dx,0.0,tymax,0.0,-dy)
        allTouched = not mustContainCenter
        img = features.rasterize(shapes,out_shape=outshape,fill=fillValue,transform=transform,all_touched=allTouched,default_value=burnValue)
        #geodict = GeoDict({'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax,'dx':dx,'dy':dy,'ny':ny,'nx':nx})
        return cls(img,geodict)
        
        
