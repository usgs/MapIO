#!/usr/bin/env python

#python 3 compatibility
from __future__ import print_function

#stdlib imports
import abc
import textwrap
import glob
import os

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



def testGeoJSON(obj):
    if hasattr(obj,'has_key') and obj.has_key('geometry') and obj.has_key('properties'):
        return True
    return False

class Grid2D(Grid):
    reqfields = set(['xmin','xmax','ymin','ymax','xdim','ydim','ncols','nrows'])
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
            nrows,ncols = dims
            if nrows != geodict.nrows or ncols != geodict.ncols:
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
        nrows,ncols = self._data.shape
        xdim,ydim = (self._geodict.xdim,self._geodict.ydim)
        zmin = np.nanmin(self._data)
        zmax = np.nanmax(self._data)
        rstr = '''<%s Object:
        nrows: %i
        ncols: %i
        xmin: %.4f
        xmax: %.4f
        ymin: %.4f
        ymax: %.4f
        xdim: %.4f
        ydim: %.4f
        zmin: %.6f
        zmax: %.6f>''' % (self.__class__.__name__,nrows,ncols,xmin,xmax,ymin,ymax,xdim,ydim,zmin,zmax)
        parts = rstr.split('\n')
        newrstr = '\n'.join([p.strip() for p in parts])
        return textwrap.dedent(newrstr)

    @classmethod
    def _createSampleData(self,M,N):
        """Used for internal testing - create an NxN grid with lower left corner at 0.5,0.5, xdim/ydim = 1.0.
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
        geodict = {'nrows':M,
                   'ncols':N,
                   'xmin':0.5,
                   'xmax':xvar[-1],
                   'ymin':0.5,
                   'ymax':yvar[-1],
                   'xdim':1.0,
                   'ydim':1.0}
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
        xdim = geodict.xdim
        ydim = geodict.ydim
        ncols = geodict.ncols
        nrows = geodict.nrows
        #section 1
        iulx1 = int(np.floor((bxmin - ulx)/xdim))
        ilrx1 = int(ncols)

        if not isScanLine:
            iuly1 = int(np.ceil((uly - bymax)/ydim))
            ilry1 = int(np.floor((uly - bymin)/ydim)) + 1
        else:
            ilry1 = int(np.ceil((uly - bymin)/ydim))
            iuly1 = int(np.floor((uly - bymax)/ydim)) + 1

        #section 2
        iulx2 = 0
        ilrx2 = int(np.ceil((bxmax - ulx)/xdim)) + 1
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
        if m != self._geodict.nrows or n != self._geodict.ncols:
            raise DataSetException('setData() input array must match rows and columns of existing data.')
        self._data = data

    def getGeoDict(self):
        """
        Return a reference to the geodict inside the Grid.
        :returns:
          A reference to a dictionary (see constructor).
        """
        return self._geodict #should we return a copy of the geodict?

    def getBounds(self):
        """
        Return the lon/lat range of the data.
        
        :returns:
           Tuple of (lonmin,lonmax,latmin,latmax)
        """
        return (self._geodict.xmin,self._geodict.xmax,self._geodict.ymin,self._geodict.ymax)

    def trim(self,geodict,resample=False,method='linear',preserve='dims'):
        """
        Trim data to a smaller set of bounds, resampling if requested.  If not resampling,
        data will be trimmed to largest grid boundary possible.
        
        :param geodict:
           GeoDict used to specify subset bounds and resolution (if resample is selected)
        :param resample:
           Boolean indicating whether the data should be resampled to *exactly* match input bounds.
        :param method:
           If resampling, method used, one of ('linear','nearest','cubic','quintic')
        :param preserve:
            String (one of 'dims','shape') indicating whether xdim/ydim of input geodict should be preserved or nrows/ncols.
        """
        xmin,xmax,ymin,ymax = (geodict.xmin,geodict.xmax,geodict.ymin,geodict.ymax)
        gxmin,gxmax,gymin,gymax = self.getBounds()
        #if any of the input bounds are outside the bounds of the grid, cut off those edges
        xmin = max(xmin,gxmin)
        xmax = min(xmax,gxmax)
        ymin = max(ymin,gymin)
        ymax = min(ymax,gymax)
        if not resample:
            uly,ulx = self.getRowCol(ymax,xmin,returnFloat=True)
            lry,lrx = self.getRowCol(ymin,xmax,returnFloat=True)
            uly = int(np.floor(uly)) #these are in pixel space!
            ulx = int(np.floor(ulx))
            lrx = int(np.ceil(lrx))
            lry = int(np.ceil(lry))
            self._data = self._data[uly:lry+1,ulx:lrx+1]
            newymax,newxmin = self.getLatLon(uly,ulx)
            newymin,newxmax = self.getLatLon(lry,lrx)
            newdict = {}
            newdict['xmin'] = newxmin
            newdict['xmax'] = newxmax
            newdict['ymin'] = newymin
            newdict['ymax'] = newymax
            newdict['xdim'] = self._geodict.xdim
            newdict['ydim'] = self._geodict.ydim
            newdict['nrows'],newdict['ncols'] = self._data.shape
            self._geodict = GeoDict(newdict,preserve=preserve)
        else:
            self.interpolateToGrid(geodict,method=method,preserve=preserve)

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
        nrows,ncols = self._data.shape
        if (row < 0).any() or (row > nrows-1).any() or (col < 0).any() or (col > ncols-1).any():
            if default is None:
                msg = 'One of more of your lat/lon values is outside Grid boundaries: %s' % (str(self.getRange()))
                raise DataSetException(msg)
            value = np.ones_like(lat)*default
            return value
        if method == 'nearest':
            return self._data[row,col]
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
        nrows1 = self._geodict.nrows
        ncols1 = self._geodict.ncols

        #handle meridian crossing grids
        if self._geodict.xmin > self._geodict.xmax:
            xmax1 = self._geodict.xmax + 360
        else:
            xmax1 = self._geodict.xmax
        xmin1 = self._geodict.xmin
        
        ymin1 = self._geodict.ymin
        ymax1 = self._geodict.ymax
        xdim1 = self._geodict.xdim
        ydim1 = self._geodict.ydim
        
        #extract the geographic information about the grid we're sampling to
        nrows = geodict.nrows
        ncols = geodict.ncols

        #handle meridian crossing grids
        if geodict.xmin > geodict.xmax:
            xmax = geodict.xmax + 360
        else:
            xmax = geodict.xmax
        xmin = geodict.xmin

        ymin = geodict.ymin
        ymax = geodict.ymax
        xdim = geodict.xdim
        ydim = geodict.ydim

        #make sure that the grid we're resampling TO is completely contained by our current grid
        if xmin1 > xmin or xmax1 < xmax or ymin1 > ymin or ymax1 < ymax:
            raise DataSetException('Grid you are resampling TO is not completely contained by base grid.')
        
        gxi = np.linspace(xmin,xmax,num=ncols)
        gyi = np.linspace(ymin,ymax,num=nrows)
        
        #we need to handle the meridian crossing here...
        if xmin > xmax:
            xmax += 360
            xmin1 += 360

        xi = (gxi - xmin1)/xdim1
        #yi = (gyi - ymin1)/ydim1
        yi = np.array(sorted(((ymax1 - gyi)/ydim1)))

        return (xi,yi)
    
    def interpolateToGrid(self,geodict,method='linear',preserve='dims'):
        """
        Given a geodict specifying another grid extent and resolution, resample current grid to match.
        
        :param geodict: 
            geodict dictionary from another grid whose extents are inside the extent of this grid.
        :param method: 
            Optional interpolation method - ['linear', 'cubic','nearest']
        :param preserve:
            String (one of 'dims','shape') indicating whether xdim/ydim of input geodict should be preserved or nrows/ncols.
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
        xdim,ydim = (geodict.xdim,geodict.ydim)
        nrows,ncols = (geodict.nrows,geodict.ncols)
        xi,yi = self._getInterpCoords(geodict)

        #now using scipy interpolate functions
        baserows,basecols = self._geodict.nrows,self._geodict.ncols
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
            newrows = geodict.nrows
            newcols = geodict.ncols
            xi = np.tile(xi,(newrows,1))
            yi = np.tile(yi.reshape(newrows,1),(1,newcols))
            self._data = f(list(zip(xi.flatten(),yi.flatten())))
            self._data = self._data.reshape(xi.shape)
                                                  
            
        nrows,ncols = geodict.nrows,geodict.ncols
        dims = self._data.shape
        nrows_new = dims[0]
        ncols_new = dims[1]
        if nrows_new != nrows or ncols_new != ncols:
            msg = "Interpolation failed!  Results (%i,%i) don't match (%i,%i)!" % (nrows_new,ncols_new,nrows,ncols)
            raise DataSetException(msg)
        #now the extents and resolution of the two grids should be identical...
        gdict = {'nrows':geodict.nrows,
                 'ncols':geodict.ncols,
                 'xmin':geodict.xmin,
                 'xmax':geodict.xmax,
                 'ymin':geodict.ymin,
                 'ymax':geodict.ymax,
                 'xdim':geodict.xdim,
                 'ydim':geodict.ydim}
        self._geodict = GeoDict(gdict)

    @classmethod
    def rasterizeFromGeometry(cls,shapes,samplegeodict,burnValue=1.0,fillValue=np.nan,allTouched=True,attribute=None):
        """
        Create a Grid2D object from vector shapes, where the presence of a shape (point, line, polygon) inside a cell turns that cell "on".
        :param shapes:
          One of:
            - One shapely geometry object (Point, Polygon, etc.) or a sequence of such objects
            - One GeoJSON like object or sequence of such objects. (http://geojson.org/)
            - A tuple of (geometry,value) or sequence of (geometry,value).
        :param samplegeodict:
          GeoDict with at least xmin,xmax,ymin,ymax,xdim,ydim values set.
        :param burnValue:
          Optional value which will be used to set the value of the pixels if there is no value in the geometry field.
        :param fillValue:
          Optional value which will be used to fill the cells not touched by any geometry.
        :param allTouched:
          Optional boolean which indicates whether the geometry must touch the center of the cell or merely be inside the cell in order to set the value.
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
        elif isinstance(shapes,dict) and shapes.has_key('geometry') and shapes.has_key('properties'):
            isOk = True
        elif len(shapes) and isinstance(shapes[0],dict) and shapes[0].has_key('geometry') and shapes[0].has_key('properties'):
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
                    if not props.has_key(attribute):
                        raise DataSetException('Input shapes do not have attribute "%s".' % attribute)
                    value = props[attribute]
                    if not isinstance(value (int,float,long)):
                        raise DataSetException('value from input shapes object is not a number')
                else:
                    value = burnValue
                shapes2.append((geometry,value))
            shapes = shapes2
        
                                   
        xmin,xmax,ymin,ymax = (samplegeodict.xmin,samplegeodict.xmax,samplegeodict.ymin,samplegeodict.ymax)
        xdim,ydim = (samplegeodict.xdim,samplegeodict.ydim)

        xvar = np.arange(xmin,xmax+(xdim*0.1),xdim)
        yvar = np.arange(ymin,ymax+(ydim*0.1),ydim)
        ncols = len(xvar)
        nrows = len(yvar)
        
        #the rasterize function assumes a pixel registered data set, where we are grid registered.  In order to make this work
        #we need to adjust the edges of our grid out by half a cell width in each direction.  
        txmin = xmin - xdim/2.0
        tymax = ymax + ydim/2.0
        
        outshape = (nrows,ncols)
        transform = Affine.from_gdal(txmin,xdim,0.0,tymax,0.0,-ydim)
        img = features.rasterize(shapes,out_shape=outshape,fill=fillValue,transform=transform,all_touched=allTouched,default_value=burnValue)
        geodict = GeoDict({'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax,'xdim':xdim,'ydim':ydim,'nrows':nrows,'ncols':ncols})
        return cls(img,geodict)
        
        
