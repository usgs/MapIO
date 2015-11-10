#!/usr/bin/env python

import abc
import numpy as np

#third party imports
from dataset import DataSet

class Grid(DataSet):
    """
    An abstract class to represent lat/lon gridded datasets. Grids are
    assumed to be pixel-registered - that is, grid coordinates
    represent the value at the *center* of the cells.
    """
    @abc.abstractmethod #should be a classmethod when instantiated
    def getFileGeoDict(filename):
        """
        Abstract method to return the bounding box, resolution, and shape of a file in whatever Grid format.
        :param filename:
           The path to the filename of whatever grid format this is being implemented in.
        :returns:
          A geodict specifying the bounding box, resolution, and shape of the data in a file.
        """
        raise NotImplementedError

    @abc.abstractmethod #should be a classmethod when instantiated
    def getBoundsWithin(filename,geodict):
        """
        Abstract method to return a geodict for this file that is guaranteed to be inside the input geodict defined, without resampling.
        :param filename:
           The name of the file whose resolution/extent should be used.
        :param geodict:
           The geodict which is used as the base for finding the bounds for this file guaranteed to be inside of this geodict.
        :raises NotImplementedError:
          Always in base class
        """
        raise NotImplementedError
    
    @classmethod
    def _getPadding(cls,geodict,padbounds,padvalue):
        xmin,xmax,ymin,ymax = padbounds
        gxmin,gxmax,gymin,gymax = (geodict['xmin'],geodict['xmax'],geodict['ymin'],geodict['ymax'])
        xdim = geodict['xdim']
        ydim = geodict['ydim']
        nrows,ncols = (geodict['nrows'],geodict['ncols'])
        padleftcols = int((gxmin - xmin)/xdim)
        padrightcols = int((xmax - gxmax)/xdim)
        padbottomrows = int((gymin - ymin)/ydim)
        padtoprows = int((ymax - gymax)/ydim)

        padleftcols = np.ceil((gxmin - xmin)/xdim)
        padrightcols = np.ceil((xmax - gxmax)/xdim)
        padbottomrows = np.ceil((gymin - ymin)/ydim)
        padtoprows = np.ceil((ymax - gymax)/ydim)

        #if any of these are negative, set them to zero
        if padleftcols < 0:
            padleftcols = 0
        if padrightcols < 0:
            padrightcols = 0
        if padbottomrows < 0:
            padbottomrows = 0
        if padtoprows < 0:
            padtoprows = 0

        leftpad = np.ones((nrows,padleftcols))*padvalue
        rightpad = np.ones((nrows,padrightcols))*padvalue
        ncols += padrightcols + padleftcols
        bottompad = np.ones((padbottomrows,ncols))*padvalue
        toppad = np.ones((padtoprows,ncols))*padvalue

        #now figure out what the new bounds are
        geodict['xmin'] = gxmin - padleftcols*xdim
        geodict['xmax'] = gxmax + padrightcols*xdim
        geodict['ymin'] = gymin - padbottomrows*ydim
        geodict['ymax'] = gymax + padtoprows*ydim
        geodict['ncols'] = geodict['ncols'] + leftpad.shape[1] + rightpad.shape[1]
        geodict['nrows'] = geodict['nrows'] + bottompad.shape[0] + toppad.shape[0]
        return (leftpad,rightpad,bottompad,toppad,geodict)
    
    @classmethod 
    def checkGeoDict(cls,geodict):
        reqfields = set(['xmin','xmax','ymin','ymax','xdim','ydim','nrows','ncols'])
        if not reqfields.issubset(set(geodict.keys())):
            return False
        return True

    @classmethod
    def fixGeoDict(cls,bounds,xdim,ydim,nrows,ncols,preserve='dims'):
        xmin,xmax,ymin,ymax = bounds
        mcross = False
        if xmin > xmax:
            xmax += 360
            mcross = True
        if preserve == 'dims':
            ncols = int((xmax-xmin)/xdim)
            xmax = xmin + ncols*xdim
            xvar = np.arange(xmin,xmax+(xdim*0.1),xdim)
            ncols = len(xvar)
            xdiff = np.abs(xmax - xvar[-1]) #xmax is not guaranteed to be exactly the same as what we just calculated...
            if mcross:
                xmax -= 360

            nrows = int((ymax-ymin)/ydim)
            ymax = ymin + nrows*ydim
            yvar = np.arange(ymin,ymax+(ydim*0.1),ydim)
            nrows = len(yvar)
            ydiff = np.abs(ymax - yvar[-1]) #ymax is not guaranteed to be exactly the same as what we just calculated...

        elif preserve == 'shape': #preserve rows and columns
            xvar,xdim = np.linspace(xmin,xmax,num=ncols,retstep=True)
            yvar,ydim = np.linspace(ymin,ymax,num=nrows,retstep=True)
            xmin = xvar[0]
            xmax = xvar[-1]
            ymin = yvar[0]
            ymax = yvar[-1]
        else:
            raise Exception('%s not supported' % preserve)

        geodict = {'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax,'xdim':xdim,'ydim':ydim,'nrows':nrows,'ncols':ncols}
        return geodict
    
    @abc.abstractmethod
    def blockmean(self,geodict):
        """
        Abstract method to calculate average values for cells of larger size than the current grid.
        :param geodict:
          Geodict that defines the new coarser grid.
        """
        raise NotImplementedError

    @abc.abstractmethod #should be a classmethod when instantiated
    def loadFromCloud(cls,cloud,geodict):
        """
        Create a grid from a Cloud instance (scattered XY data).
        :param cloud:
          A Cloud instance containing scattered XY data.
        :param geodict:
          A geodict object where nrows/ncols are optional (will be calculated from bounds/cell dimensions)
        :returns:
          An instance of a Grid object.
        """
        raise NotImplementedError
    
    @staticmethod
    def getLatLonMesh(geodict):
        lons = np.linspace(geodict['xmin'],geodict['xmax'],num=geodict['ncols'])
        lats = np.linspace(geodict['ymin'],geodict['ymax'],num=geodict['nrows'])
        lon,lat = np.meshgrid(lons,lats)
        return (lat,lon)
    
    @abc.abstractmethod
    def getGeoDict(self):
        """
        Return a reference to the geodict inside the Grid
        
        :returns:
          A reference to a dictionary (see constructor).
        """
        raise NotImplementedError('getGeoDict method not implemented in base class')

    @abc.abstractmethod
    def getLatLon(self,row,col):
        """Return geographic coordinates (lat/lon decimal degrees) for given data row and column.
        
        :param row: 
           Row dimension index into internal data array.
        :param col: 
           Column dimension index into internal data array.
        :returns: 
           Tuple of latitude and longitude.
        """
        raise NotImplementedError('getLatLon method not implemented in base class')

    @abc.abstractmethod
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
        raise NotImplementedError('getRowCol method not implemented in base class')



    
    
    
        
