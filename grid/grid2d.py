#!/usr/bin/env python

#stdlib imports
import abc
import textwrap
import glob
import os

#third party imports
from gridbase import Grid
from dataset import DataSetException
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
    """
    A partially abstract class to represent 2D lat/lon gridded datasets. Some basic methods
    are implemented here, enough so that all functions of working with data (aside from loading and saving)
    can be used with this class.  Grids are assumed to be pixel-registered - that is, grid coordinates
    represent the value at the *center* of the cells.
    """
    reqfields = set(['xmin','xmax','ymin','ymax','xdim','ydim','ncols','nrows'])
    def __init__(self,data=None,geodict=None):
        """
        Construct a Grid object.
        
        :param data: 
            A 2D numpy array (can be None)
        :param geodict: 
            A dictionary (or None) containing the following fields:
             - xmin Longitude minimum (decimal degrees) (Center of upper left cell)
             - xmax Longitude maximum (decimal degrees) (Center of upper right cell)
             - ymin Longitude minimum (decimal degrees) (Center of lower left cell)
             - ymax Longitude maximum (decimal degrees) (Center of lower right cell)
             - xdim Cell width (decimal degrees)
             - ydim Cell height (decimal degrees)
             - nrows Number of rows of input data (must match input data dimensions)
             - ncols Number of columns of input data (must match input data dimensions)
        :returns:
            A Grid object.  Internal representation of geodict input will have nrows/ncols fields added.
        :raises DataSetException:
           When data is not 2D or the number of rows and columns do not match the geodict.
        """
        if data is not None and geodict is not None:
            #complain if data is not 2D (squeeze 1d dimensions out)
            dims = data.shape
            if len(dims) != 2:
                raise DataSetException('Grid data must be 2D.  Input data has shape of %s' % str(data.shape))
            nrows,ncols = dims
            if nrows != geodict['nrows'] or ncols != geodict['ncols']:
                raise DataSetException('Input geodict does not match shape of input data.')
            #complain if geodict does not have all required fields
            if not set(geodict.keys()).issuperset(self.reqfields):
                raise DataSetException('Missing some of required fields in geodict.')
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
        xmin,xmax,ymin,ymax = (self._geodict['xmin'],self._geodict['xmax'],
                               self._geodict['ymin'],self._geodict['ymax'])
        nrows,ncols = self._data.shape
        xdim,ydim = (self._geodict['xdim'],self._geodict['ydim'])
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
        """Used for internal testing - create an NxN grid with lower left corner at 0.5,0.5, xdim/ydim = 1.0
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
        return (data,geodict)
    
    #This should be a @classmethod in subclasses
    @abc.abstractmethod
    def load(filename,bounds=None,resample=False,padValue=None):
        raise NotImplementedError('Load method not implemented in base class')

    @classmethod
    def copyFromGrid(cls,grid):
        if not isinstance(grid,Grid2D):
            raise DataSetException('Input to copyFromGrid must be an instance of a Grid2D object (inc. subclasses)')
        cls(grid.getData(),grid.getGeoDict())

    
    #This should be a @classmethod in subclasses
    @abc.abstractmethod
    def save(self,filename): #would we ever want to save a subset of the data?
        raise NotImplementedError('Save method not implemented in base class')
    
    @classmethod
    def _createSections(self,bounds,geodict,firstColumnDuplicated,isScanLine=False):
        """Given a grid that goes from -180 to 180 degrees, figure out the two pixel regions that up both sides of the subset
        :param bounds:
           Tuple of (xmin,xmax,ymin,ymax)
        :param geodict:
           Geodict dictionary
        :param firstColumnDuplicated:
          Boolean indicating whether this is a global data set where the first and last columns are identical
        :param isScanLine:
          Boolean indicating whether this array is in scan line order (pixel[0,0] is the geographic upper left).
        :returns:
          Two tuples of 4 elements each - (iulx,iuly,ilrx,ilry). The first tuple defines the pixel offsets for the left
          side of the subsetted region, and the second tuple defines the pixel offsets for the right side.
        """
        (bxmin,bxmax,bymin,bymax) = bounds
        ulx = geodict['xmin']
        uly = geodict['ymax']
        xdim = geodict['xdim']
        ydim = geodict['ydim']
        ncols = geodict['ncols']
        nrows = geodict['nrows']
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
        """
        Return a reference to the data inside the Grid
        :returns:
          A reference to a 2D numpy array.
        """
        return self._data #should we return a copy of the data?

    def getGeoDict(self):
        """
        Return a reference to the geodict inside the Grid
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
        return (self._geodict['xmin'],self._geodict['xmax'],self._geodict['ymin'],self._geodict['ymax'])

    def trim(self,bounds,resample=False,method='linear'):
        """
        Trim data to a smaller set of bounds, resampling if requested.  If not resampling,
        data will be trimmed to largest grid boundary possible.
        
        :param bounds:
           Tuple of (lonmin,lonmax,latmin,latmax)
        :param resample:
           Boolean indicating whether the data should be resampled to *exactly* match input bounds.
        :param method:
           If resampling, method used, one of ('linear','nearest','cubic','quintic')
        """
        xmin,xmax,ymin,ymax = bounds
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
            self._geodict['xmin'] = newxmin
            self._geodict['xmax'] = newxmax
            self._geodict['ymin'] = newymin
            self._geodict['ymax'] = newymax
            self._geodict['nrows'],self._geodict['ncols'] = self._data.shape
        else:
            xdim = self._geodict['xdim']
            ydim = self._geodict['ydim']
            indict = {'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax,'xdim':xdim,'ydim':ydim}
            ncols = len(np.arange(xmin,xmax+xdim,xdim))
            nrows = len(np.arange(ymin,ymax+ydim,ydim))
            indict['nrows'] = nrows
            indict['ncols'] = ncols
            self.interpolateToGrid(indict,method=method)

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
        ulx = self._geodict['xmin']
        uly = self._geodict['ymax']
        xdim = self._geodict['xdim']
        ydim = self._geodict['ydim']
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
        ulx = self._geodict['xmin']
        uly = self._geodict['ymax']
        xdim = self._geodict['xdim']
        ydim = self._geodict['ydim']
        #check to see if we're in a scenario where the grid crosses the meridian
        if self._geodict['xmax'] < ulx and lon < ulx:
            lon += 360
        col = (lon-ulx)/xdim
        row = (uly-lat)/ydim
        if returnFloat:
            return (row,col)
        
        return (np.floor(row).astype(int),np.floor(col).astype(int))

    def _getInterpCoords(self,geodict):
        #translate geographic coordinates to 2 1-D arrays of X and Y pixel coordinates
        #remember that pixel coordinates are (0,0) at the top left and increase going down and to the right
        #geographic coordinates are (xmin,ymin) at the bottom left and increase going up and to the right
        dims = self._data.shape
        nrows1 = self._geodict['nrows']
        ncols1 = self._geodict['ncols']
        xmin1 = self._geodict['xmin']
        xmax1 = self._geodict['xmax']
        ymin1 = self._geodict['ymin']
        ymax1 = self._geodict['ymax']
        xdim1 = self._geodict['xdim']
        ydim1 = self._geodict['ydim']
        
        #extract the geographic information about the grid we're sampling to
        nrows = geodict['nrows']
        ncols = geodict['ncols']
        xmin = geodict['xmin']
        xmax = geodict['xmax']
        ymin = geodict['ymin']
        ymax = geodict['ymax']
        xdim = geodict['xdim']
        ydim = geodict['ydim']

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
        yi = (gyi - ymin1)/ydim1

        return (xi,yi)
    
    def interpolateToGrid(self,geodict,method='linear'):
        """
        Given a geodict specifying another grid extent and resolution, resample current grid to match.
        
        :param geodict: 
            geodict dictionary from another grid whose extents are inside the extent of this grid.
        :keyword method: 
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
        geodict = super(Grid2D,self).fillGeoDict(geodict)
        xi,yi = self._getInterpCoords(geodict)

        #now using scipy interpolate functions
        baserows,basecols = self._geodict['nrows'],self._geodict['ncols']
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
            f = interpolate.NearestNDInterpolator(zip(x.flatten(),y.flatten()),self._data.flatten())
            newrows = geodict['nrows']
            newcols = geodict['ncols']
            xi = np.tile(xi,(newrows,1))
            yi = np.tile(yi.reshape(newrows,1),(1,newcols))
            self._data = f(zip(xi.flatten(),yi.flatten()))
            self._data = self._data.reshape(xi.shape)
                                                  
            
        nrows,ncols = geodict['nrows'],geodict['ncols']
        dims = self._data.shape
        nrows_new = dims[0]
        ncols_new = dims[1]
        if nrows_new != nrows or ncols_new != ncols:
            msg = "Interpolation failed!  Results (%i,%i) don't match (%i,%i)!" % (nrows_new,ncols_new,nrows,ncols)
            raise DataSetException(msg)
        #now the extents and resolution of the two grids should be identical...
        self._geodict['nrows'] = geodict['nrows']
        self._geodict['ncols'] = geodict['ncols']
        self._geodict['xmin'] = geodict['xmin']
        self._geodict['xmax'] = geodict['xmax']
        self._geodict['ymin'] = geodict['ymin']
        self._geodict['ymax'] = geodict['ymax']
        self._geodict['xdim'] = geodict['xdim']
        self._geodict['ydim'] = geodict['ydim']

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
        
                                   
        xmin,xmax,ymin,ymax = (samplegeodict['xmin'],samplegeodict['xmax'],samplegeodict['ymin'],samplegeodict['ymax'])
        xdim,ydim = (samplegeodict['xdim'],samplegeodict['ydim'])

        xvar = np.arange(xmin,xmax+xdim,xdim)
        yvar = np.arange(ymin,ymax+ydim,ydim)
        ncols = len(xvar)
        nrows = len(yvar)
        
        #the rasterize function assumes a pixel registered data set, where we are grid registered.  In order to make this work
        #we need to adjust the edges of our grid out by half a cell width in each direction.  
        txmin = xmin - xdim/2.0
        tymax = ymax + ydim/2.0
        
        outshape = (nrows,ncols)
        transform = Affine.from_gdal(txmin,xdim,0.0,tymax,0.0,-ydim)
        img = features.rasterize(shapes,out_shape=outshape,fill=fillValue,transform=transform,all_touched=allTouched,default_value=burnValue)
        geodict = {'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax,'xdim':xdim,'ydim':ydim,'nrows':nrows,'ncols':ncols}
        return cls(img,geodict)
        
def _test_basics():
    geodict = {'xmin':0.5,'xmax':3.5,'ymin':0.5,'ymax':3.5,'xdim':1.0,'ydim':1.0,'nrows':4,'ncols':4}
    data = np.arange(0,16).reshape(4,4)
    grid = Grid2D(data,geodict)
    print 'Testing basic Grid2D functionality (retrieving data, lat/lon to pixel coordinates, etc...'
    np.testing.assert_almost_equal(grid.getData(),data)

    assert grid.getGeoDict() == geodict

    assert grid.getBounds() == (geodict['xmin'],geodict['xmax'],geodict['ymin'],geodict['ymax'])
    
    lat,lon = grid.getLatLon(0,0)

    assert lat == 3.5 and lon == 0.5
        
    row,col = grid.getRowCol(lat,lon)

    assert row == 0 and col == 0
    
    value = grid.getValue(lat,lon)

    assert value == 0
    
    frow,fcol = grid.getRowCol(1.0,3.0,returnFloat=True)

    assert frow == 2.5 and fcol == 2.5
    
    irow,icol = grid.getRowCol(1.0,3.0,returnFloat=False)

    assert irow == 2 and icol == 2
    print 'Passed basic Grid2D functionality (retrieving data, lat/lon to pixel coordinates, etc...'
    
def _test_resample():
    geodict = {'xmin':0.5,'xmax':4.5,'ymin':0.5,'ymax':4.5,'xdim':1.0,'ydim':1.0,'nrows':5,'ncols':5}
    data = np.arange(0,25).reshape(5,5)

    print 'Testing data trimming without resampling...'
    grid = Grid2D(data,geodict)
    bounds = (2.0,3.0,2.0,3.0)
    grid.trim(bounds,resample=False)
    output = np.array([[6,7,8],[11,12,13],[16,17,18]])
    np.testing.assert_almost_equal(grid.getData(),output)
    print 'Passed data trimming without resampling...'

    print 'Testing data trimming with resampling...'
    grid = Grid2D(data,geodict)
    grid.trim(bounds,resample=True)
    output = np.array([[9.0,10.0],[14.0,15.0]])
    np.testing.assert_almost_equal(grid.getData(),output)
    print 'Passed data trimming with resampling...'

def _test_interpolate():
    geodict = {'xmin':0.5,'xmax':4.5,'ymin':0.5,'ymax':4.5,'xdim':1.0,'ydim':1.0,'nrows':5,'ncols':5}
    data = np.arange(0,25).reshape(5,5)
    
    for method in ['nearest','linear','cubic']:
        print 'Testing interpolate with method "%s"...' % method
        grid = Grid2D(data,geodict)
        sampledict = {'xmin':2.0,'xmax':3.0,'ymin':2.0,'ymax':3.0,'xdim':1.0,'ydim':1.0}
        grid.interpolateToGrid(sampledict,method=method)
        if method == 'nearest':
            output = np.array([[6.0,7.0],[11.0,17.0]])
        elif method == 'linear':
            output = np.array([[9.0,10.0],[14.0,15.0]])
        elif method == 'cubic':
            output = np.array([[9.0,10.0],[14.0,15.0]])
        else:
            pass
        np.testing.assert_almost_equal(grid.getData(),output)
        print 'Passed interpolate with method "%s".' % method

def _test_rasterize():
    samplegeodict = {'xmin':0.5,'xmax':3.5,'ymin':0.5,'ymax':3.5,'xdim':1.0,'ydim':1.0}
    print 'Testing rasterizeFromGeometry() trying to get binary output...'
    points = MultiPoint([(0.25,3.5,5.0),
                         (1.75,3.75,6.0),
                         (1.0,2.5,10.0),
                         (3.25,2.5,17.0),
                         (1.5,1.5,1.0),
                         (3.25,0.5,86.0)])
    
    grid = Grid2D.rasterizeFromGeometry(points,samplegeodict,burnValue=1.0,fillValue=0.0)
    output = np.array([[1.0,1.0,0.0,0.0],
                       [0.0,1.0,0.0,1.0],
                       [0.0,1.0,0.0,0.0],
                       [0.0,0.0,0.0,1.0]])
    np.testing.assert_almost_equal(grid.getData(),output)
    print 'Passed rasterizeFromGeometry() trying to get binary output.'

    try:
        print 'Testing rasterizeFromGeometry() burning in values from a polygon sequence...'
        #Define two simple polygons and assign them to shapes
        poly1 = [(0.25,3.75),(1.25,3.25),(1.25,2.25)]
        poly2 = [(2.25,3.75),(3.25,3.75),(3.75,2.75),(3.75,1.50),(3.25,0.75),(2.25,2.25)]
        shape1 = {'properties':{'value':5},'geometry':mapping(Polygon(poly1))}
        shape2 = {'properties':{'value':7},'geometry':mapping(Polygon(poly2))}
        shapes = [shape1,shape2]
        
        grid = Grid2D.rasterizeFromGeometry(shapes,samplegeodict,fillValue=0,attribute='value')
        output = np.array([[5,5,7,7],
                           [5,5,7,7],
                           [0,0,7,7],
                           [0,0,0,7]])
        np.testing.assert_almost_equal(grid.getData(),output)
        print 'Testing rasterizeFromGeometry() burning in values from a polygon shapefile...'
    except:
        shpfiles = glob.glob('test.*')
        for shpfile in shpfiles:
            os.remove(shpfile)
    
    
if __name__ == '__main__':
    _test_rasterize()
    _test_basics()
    _test_resample()
    _test_interpolate()
    
        
