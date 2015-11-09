#!/usr/bin/env python

#stdlib imports
import os.path
import sys
from collections import OrderedDict
import warnings

#third party imports
import rasterio
import numpy as np
from grid2d import Grid2D
from dataset import DataSetException,DataSetWarning

class GDALGrid(Grid2D):
    def __init__(self,data,geodict):
        """Construct a GMTGrid object.
        :param data:
           2D numpy data array (must match geodict spec)
        :param geodict:
           Dictionary specifying the spatial extent,resolution and shape of the data.
        :returns:
           A GMTGrid object.
        :raises DataSetException:
          When data and geodict dimensions do not match. 
        """
        m,n = data.shape
        if m != geodict['nrows'] or n != geodict['ncols']:
            raise DataSetException('Input geodict does not match shape of input data.')
        self._data = data
        self._geodict = geodict

    @classmethod
    def getFileGeoDict(cls,filename):
        """Get the spatial extent, resolution, and shape of grid inside ESRI grid file.
        :param filename:
           File name of ESRI grid file.
        :returns:
           - GeoDict specifying spatial extent, resolution, and shape of grid inside ESRI grid file.
           - xvar array specifying X coordinates of data columns
           - yvar array specifying Y coordinates of data rows
        :raises DataSetException:
          When the file contains a grid with more than one band.
        """
        geodict = {}
        with rasterio.drivers():
            with rasterio.open(filename) as src:
                aff = src.affine
                geodict['xdim'] = aff.a
                geodict['ydim'] = -1*aff.e
                geodict['xmin'] = aff.xoff + geodict['xdim']/2.0
                geodict['ymax'] = aff.yoff - geodict['ydim']/2.0
                                
                shp = src.shape
                if len(shp) > 2:
                    raise DataSetException('Cannot support grids with more than one band')
                geodict['nrows'] = src.height
                geodict['ncols'] = src.width
                geodict['xmax'] = geodict['xmin'] + (geodict['ncols']-1)*geodict['xdim']
                geodict['ymin'] = geodict['ymax'] - (geodict['nrows']-1)*geodict['ydim']
        xvar = np.arange(geodict['xmin'],geodict['xmax']+geodict['xdim'],geodict['xdim'])
        yvar = np.arange(geodict['ymin'],geodict['ymax']+geodict['ydim'],geodict['ydim'])
        return (geodict,xvar,yvar)

    @classmethod
    def getBoundsWithin(cls,filename,geodict):
        """
        Return a geodict for a file that is guaranteed to be contained by an input geodict.
        :param filename:
          Path to an ESRI grid file.
        :param geodict:
          Geodict defining a spatial extent that we want to envelope the returned geodict.
        :returns:
          A geodict that is guaranteed to be contained by input geodict.
        """
        fgeodict,xvar,yvar = cls.getFileGeoDict(filename)
        fxmin,fxmax,fymin,fymax = (fgeodict['xmin'],fgeodict['xmax'],fgeodict['ymin'],fgeodict['ymax'])
        xmin,xmax,ymin,ymax = (geodict['xmin'],geodict['xmax'],geodict['ymin'],geodict['ymax'])
        fxdim,fydim = (fgeodict['xdim'],fgeodict['ydim'])
        
        ulcol = int(np.ceil((xmin - fxmin)/fxdim))+1
        ulrow = int(np.floor((ymax - fymin)/fydim))-1
        lrcol = int(np.floor((xmax - fxmin)/fxdim))-1
        lrrow = int(np.ceil((ymin-fymin)/fydim))+1

        newxmin = fxmin + ulcol*fxdim
        newxmax = fxmin + lrcol*fxdim
        newymax = fymin + ulrow*fydim
        newymin = fymin + lrrow*fydim

        outgeodict = {'xmin':newxmin,'xmax':newxmax,'ymin':newymin,'ymax':newymax,'xdim':fxdim,'ydim':fydim}
        return outgeodict
    
    @classmethod
    def _subsetRegions(self,src,bounds,fgeodict,xvar,yvar,firstColumnDuplicated):
        """Internal method used to do subsampling of data for all three GMT formats.
        :param zvar:
          A numpy array-like thing (CDF/HDF variable, or actual numpy array)
        :param bounds:
          Tuple with (xmin,xmax,ymin,ymax) for subsetting.
        :param fgeodict:
          Geo dictionary with the file information.
        :param xvar:
          Numpy array specifying X coordinates of data columns
        :param yvar:
          Numpy array specifying Y coordinates of data rows
        :param firstColumnDuplicated:
          Boolean - is this a file where the last column of data is the same as the first (for grids that span entire globe).
        :returns:
          Tuple of (data,geodict) (subsetted data and geodict describing that data).
        """
        txmin,txmax,tymin,tymax = bounds
        #we're not doing anything fancy with the data here, just cutting out what we need
        xmin = max(fgeodict['xmin'],txmin)
        xmax = min(fgeodict['xmax'],txmax)
        ymin = max(fgeodict['ymin'],tymin)
        ymax = min(fgeodict['ymax'],tymax)
        #these are the bounds of the whole file
        gxmin = fgeodict['xmin']
        gxmax = fgeodict['xmax']
        gymin = fgeodict['ymin']
        gymax = fgeodict['ymax']
        xdim = fgeodict['xdim']
        ydim = fgeodict['ydim']
        gnrows = fgeodict['nrows']
        gncols = fgeodict['ncols']
        if xmin == gxmin and xmax == gxmax and ymin == gymin and ymax == gymax:
            data = src.read()
            data = np.squeeze(data)
            if firstColumnDuplicated:
                data = data[:,0:-1]
                geodict['xmax'] -= geodict['xdim']
        else:
            if xmin > xmax:
                #cut user's request into two regions - one from the minimum to the
                #meridian, then another from the meridian to the maximum.
                (region1,region2) = self._createSections((xmin,xmax,ymin,ymax),fgeodict,firstColumnDuplicated)
                (iulx1,iuly1,ilrx1,ilry1) = region1
                (iulx2,iuly2,ilrx2,ilry2) = region2
                window1 = ((iuly1,ilry1),(iulx1,ilrx1))
                window2 = ((iuly2,ilry2),(iulx2,ilrx2))
                section1 = src.read(1,window=window1)
                section2 = src.read(1,window=window2)
                data = np.hstack((section1,section2))
                outrows,outcols = data.shape
                xmin = (gxmin + iulx1*xdim)
                ymax = gymax - iuly1*ydim
                xmax = gxmin + (ilrx2-1)*xdim
                ymin = gymin + (gnrows-ilry1)*ydim
                fgeodict['xmin'] = xmin
                fgeodict['xmax'] = xmax + 360
                fgeodict['ymin'] = ymin
                fgeodict['ymax'] = ymax
                fgeodict['nrows'],fgeodict['ncols'] = data.shape
            else:
                #get the highest index of a positive difference btw xmin and xvar
                #use that as an index to get the xmin on a grid cell
                ixmin = np.where((xmin-xvar) >= 0)[0].max()
                ixmax = np.where((xmax-xvar) <= 0)[0].min()

                iymin = int((gymax - ymax)/ydim)
                iymax = int((gymax - ymin)/ydim)
                
                # iymax = int(np.ceil((gnrows-1) - ((ymin-gymin)/ydim)))
                # iymin = int(np.floor((gnrows-1) - ((ymax-gymin)/ydim)))
               
                fgeodict['xmin'] = xvar[ixmin].copy()
                fgeodict['xmax'] = xvar[ixmax].copy()
                fgeodict['ymin'] = gymax - iymax*ydim
                fgeodict['ymax'] = gymax - iymin*xdim
                window = ((iymin,iymax+1),(ixmin,ixmax+1))
                data = src.read(1,window=window)
                data = np.squeeze(data)
                fgeodict['nrows'],fgeodict['ncols'] = data.shape
            
        
        return (data,fgeodict)

    @classmethod
    def readGDAL(cls,filename,bounds=None,firstColumnDuplicated=False):
        geodict,xvar,yvar = cls.getFileGeoDict(filename)
        data = None
        with rasterio.drivers():
            with rasterio.open(filename) as src:
                if bounds is None:
                    data = src.read()
                    data = np.squeeze(data)
                    if firstColumnDuplicated:
                        data = data[:,0:-1]
                        geodict['xmax'] -= geodict['xdim']
                else:
                    data,geodict = cls._subsetRegions(src,bounds,geodict,xvar,yvar,firstColumnDuplicated)
        #Put NaN's back in where nodata value was
        nodata = src.get_nodatavals()[0]
        if (data==nodata).any():
            data[data == nodata] = np.nan
        return (data,geodict)
                

    def _getHeader(self):
        hdr = {}
        if sys.byteorder == 'little':
            hdr['BYTEORDER'] = 'LSBFIRST'
        else:
            hdr['BYTEORDER'] = 'MSBFIRST'
        hdr['LAYOUT'] = 'BIL'
        hdr['NROWS'],hdr['NCOLS'] = self._data.shape
        hdr['NBANDS'] = 1
        if self._data.dtype == np.uint8:
            hdr['NBITS'] = 8
            hdr['PIXELTYPE'] = 'UNSIGNEDINT'
        elif self._data.dtype == np.int8:
            hdr['NBITS'] = 8
            hdr['PIXELTYPE'] = 'SIGNEDINT'
        elif self._data.dtype == np.uint16:
            hdr['NBITS'] = 16
            hdr['PIXELTYPE'] = 'UNSIGNEDINT'
        elif self._data.dtype == np.int16:
            hdr['NBITS'] = 16
            hdr['PIXELTYPE'] = 'SIGNEDINT'
        elif self._data.dtype == np.uint32:
            hdr['NBITS'] = 32
            hdr['PIXELTYPE'] = 'UNSIGNEDINT'
        elif self._data.dtype == np.int32:
            hdr['NBITS'] = 32
            hdr['PIXELTYPE'] = 'SIGNEDINT'
        elif self._data.dtype == np.float32:
            hdr['NBITS'] = 32
            hdr['PIXELTYPE'] = 'FLOAT'
        elif self._data.dtype == np.float64:
            hdr['NBITS'] = 32
            hdr['PIXELTYPE'] = 'FLOAT'
        else:
            raise DataSetException('Data type "%s" not supported.' % str(self._data.dtype))
        hdr['BANDROWBYTES'] = hdr['NCOLS']*(hdr['NBITS']/8)
        hdr['TOTALROWBYTES'] = hdr['NCOLS']*(hdr['NBITS']/8)
        hdr['ULXMAP'] = self._geodict['xmin']
        hdr['ULYMAP'] = self._geodict['ymax']
        hdr['XDIM'] = self._geodict['xdim']
        hdr['YDIM'] = self._geodict['ydim']
        #try to have a nice readable NODATA value in the header file
        zmin = np.nanmin(self._data)
        zmax = np.nanmax(self._data)
        if self._data.dtype in [np.int8,np.int16,np.int32]:
            nodata = np.array([-1*int('9'*i) for i in range(3,20)])
            if zmin > nodata[-1]:
                NODATA = nodata[np.where(nodata < zmin)[0][0]]
            else: #otherwise just pick an arbitrary value smaller than our smallest
                NODATA = zmin - 1
        else:
            nodata = np.array([int('9'*i) for i in range(3,20)])
            if zmin < nodata[-1]:
                NODATA = nodata[np.where(nodata > zmin)[0][0]]
            else: #otherwise just pick an arbitrary value smaller than our smallest
                NODATA = zmax + 1
        hdr['NODATA'] = NODATA
        keys = ['BYTEORDER','LAYOUT','NROWS','NCOLS','NBANDS','NBITS','BANDROWBYTES','TOTALROWBYTES','PIXELTYPE',
                'ULXMAP','ULYMAP','XDIM','YDIM','NODATA']
        hdr2 = OrderedDict()
        for key in keys:
            hdr2[key] = hdr[key]
        return hdr2
    
    def save(self,filename,format='EHdr'):
        #http://webhelp.esri.com/arcgisdesktop/9.3/index.cfm?TopicName=BIL,_BIP,_and_BSQ_raster_files
        #http://resources.esri.com/help/9.3/arcgisdesktop/com/gp_toolref/conversion_tools/float_to_raster_conversion_.htm
        supported = ['EHdr']
        if format not in supported:
            raise DataSetException('Only "%s" file formats supported for saving' % str(supported))
        hdr = self._getHeader()
        data = self._data #create a reference to the data - this may be overridden by a downcasted version for doubles
        if self._data.dtype == np.float32:
            data = self._data.astype(np.float32) #so we can find/reset nan values without screwing up original data
            data[np.isnan(data)] = hdr['NODATA']
        elif self._data.dtype == np.float64:
            data = self._data.astype(np.float32)
            data[np.isnan(data)] = hdr['NODATA']
            warnings.warn(DataSetWarning('Down-casting double precision floating point to single precision'))

        data.tofile(filename)
        #write out the header file
        basefile,ext = os.path.splitext(filename)
        hdrfile = basefile+'.hdr'
        f = open(hdrfile,'wt')
        for key,value in hdr.iteritems():
            value = hdr[key]
            f.write('%s  %s\n' % (key,str(value)))
        f.close()
            
    
    @classmethod
    def load(cls,filename,samplegeodict=None,resample=False,method='linear',doPadding=False,padValue=np.nan):
        """Create a GDALGrid object from a (possibly subsetted, resampled, or padded) GDAL-compliant grid file.
        :param filename:
          Name of input file.
        :param samplegeodict:
          GeoDict used to specify subset bounds and resolution (if resample is selected)
        :param resample:
          Boolean used to indicate whether grid should be resampled from the file based on samplegeodict.
        :param method:
          If resample=True, resampling method to use ('nearest','linear','cubic','quintic')
        :param doPadding:
          Boolean used to indicate whether, if samplegeodict is outside bounds of grid, to pad values around the edges.
        :param padValue:
          Value to fill in around the edges if doPadding=True.
        :returns:
          GDALgrid instance (possibly subsetted, padded, or resampled)
        :raises DataSetException:
          * When sample bounds are outside (or too close to outside) the bounds of the grid and doPadding=False.
          * When the input file type is not recognized.
        """
        data = None
        geodict = None
        bounds = None
        samplebounds = None
        firstColumnDuplicated = False
        if samplegeodict is not None:
            bounds = (samplegeodict['xmin'],samplegeodict['xmax'],samplegeodict['ymin'],samplegeodict['ymax'])
            samplebounds = bounds
            #if the user wants resampling, we can't just read the bounds they asked for, but instead
            #go outside those bounds.  if they asked for padding and the input bounds exceed the bounds
            #of the file, then we can pad.  If they *didn't* ask for padding and input exceeds, raise exception.
            if resample:
                PADFACTOR = 2 #how many cells will we buffer out for resampling?
                filegeodict,xvar,yvar = cls.getFileGeoDict(filename)
                xdim = filegeodict['xdim']
                ydim = filegeodict['ydim']
                fbounds = (filegeodict['xmin'],filegeodict['xmax'],filegeodict['ymin'],filegeodict['ymax'])
                hasMeridianWrap = False
                if fbounds[0] == fbounds[1]-360:
                    firstColumnDuplicated = True
                if firstColumnDuplicated or np.abs(fbounds[0]-(fbounds[1]-360)) == xdim:
                    hasMeridianWrap = True
                isOutside = False
                #make a bounding box that is PADFACTOR number of rows/cols greater than what the user asked for
                rbounds = [bounds[0]-xdim*PADFACTOR,bounds[1]+xdim*PADFACTOR,bounds[2]-ydim*PADFACTOR,bounds[3]+ydim*PADFACTOR]
                #compare that bounding box to the file bounding box
                if not hasMeridianWrap:
                    if fbounds[0] > rbounds[0] or fbounds[1] < rbounds[1] or fbounds[2] > rbounds[2] or fbounds[3] < rbounds[3]:
                        isOutside = True
                else:
                    if fbounds[2] > rbounds[2] or fbounds[3] < rbounds[3]:
                        isOutside = True
                if isOutside:
                    if doPadding==False:
                        raise DataSetException('Cannot resample data given input bounds, unless doPadding is set to True.')
                    else:
                        samplebounds = rbounds
                else:
                    samplebounds = rbounds

        data,geodict = cls.readGDAL(filename,samplebounds,firstColumnDuplicated=firstColumnDuplicated)
        if doPadding:
            #up to this point, all we've done is either read in the whole file or cut out (along existing
            #boundaries) the section of data we want.  Now we do padding as necessary.
            #_getPadding is a class method inherited from Grid (our grandparent)
            leftpad,rightpad,bottompad,toppad,geodict = super(Grid2D,cls)._getPadding(geodict,samplebounds,padValue)
            data = np.hstack((leftpad,data))
            data = np.hstack((data,rightpad))
            data = np.vstack((toppad,data))
            data = np.vstack((data,bottompad))
        #if the user asks to resample, take the (possibly cut and padded) data set, and resample
        #it using the Grid2D super class
        if resample:
            grid = Grid2D(data,geodict)
            if samplegeodict['xmin'] > samplegeodict['xmax']:
                samplegeodict['xmax'] += 360
            grid.interpolateToGrid(samplegeodict,method=method)
            data = grid.getData()
            geodict = grid.getGeoDict()
        return cls(data,geodict)


def createSample(M,N):
    data = np.arange(0,M*N).reshape(M,N)
    data = data.astype(np.int32) #arange gives int64 by default, not supported by netcdf3
    xvar = np.arange(0.5,0.5+N,1.0)
    yvar = np.arange(0.5,0.5+M,1.0)
    geodict = {'nrows':N,
               'ncols':N,
               'xmin':0.5,
               'xmax':xvar[-1],
               'ymin':0.5,
               'ymax':yvar[-1],
               'xdim':1.0,
               'ydim':1.0}
    gmtgrid = GDALGrid(data,geodict)
    return gmtgrid

def _format_test():
    try:
        for dtype in [np.uint8,np.uint16,np.uint32,np.int8,np.int16,np.int32,np.float32,np.float64]:
            print 'Testing saving/loading of data with type %s...' % str(dtype)
            data = np.arange(0,16).reshape(4,4).astype(dtype)
            if dtype in [np.float32,np.float64]:
                data[1,1] = np.nan
            geodict = {'xmin':0.5,'xmax':3.5,'ymin':0.5,'ymax':3.5,'xdim':1.0,'ydim':1.0,'nrows':4,'ncols':4}
            gdalgrid = GDALGrid(data,geodict)
            gdalgrid.save('test.bil')
            gdalgrid2 = GDALGrid.load('test.bil')
            np.testing.assert_almost_equal(gdalgrid2.getData(),gdalgrid.getData())
            print 'Passed saving/loading of data with type %s...' % str(dtype)

    except Exception,obj:
        print 'Failed tests with message: "%s"' % str(obj)
    os.remove('test.bil')

def _pad_test():
    try:
        print 'Test padding data with null values...'
        data,geodict = Grid2D._createSampleData(4,4)
        gdalgrid = GDALGrid(data,geodict)
        gdalgrid.save('test.bil')

        newdict = {'xmin':-0.5,'xmax':4.5,'ymin':-0.5,'ymax':4.5,'xdim':1.0,'ydim':1.0}
        gdalgrid2 = GDALGrid.load('test.bil',samplegeodict=newdict,doPadding=True)
        output = np.array([[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                           [np.nan,0.0,1.0,2.0,3.0,np.nan],
                           [np.nan,4.0,5.0,6.0,7.0,np.nan],
                           [np.nan,8.0,9.0,10.0,11.0,np.nan],
                           [np.nan,12.0,13.0,14.0,15.0,np.nan],
                           [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]])
        np.testing.assert_almost_equal(gdalgrid2._data,output)
        print 'Passed padding data null values.'
    except AssertionError,error:
        print 'Failed padding test:\n %s' % error
    os.remove('test.bil')

def _subset_test():
    try:
        print 'Testing subsetting of non-square grid...'
        data,geodict = Grid2D._createSampleData(6,4)
        gdalgrid = GDALGrid(data,geodict)
        gdalgrid.save('test.bil')
        newdict = {'xmin':1.5,'xmax':2.5,'ymin':1.5,'ymax':3.5,'xdim':1.0,'ydim':1.0}
        gdalgrid3 = GDALGrid.load('test.bil',samplegeodict=newdict)
        output = np.array([[9,10],
                           [13,14],
                           [17,18]])
        np.testing.assert_almost_equal(gdalgrid3._data,output)
        print 'Passed subsetting of non-square grid.'
        
    except AssertionError,error:
        print 'Failed subset test:\n %s' % error

    os.remove('test.bil')

def _resample_test():
    try:
        print 'Test resampling data without padding...'
        data,geodict = Grid2D._createSampleData(9,7)
        gdalgrid = GDALGrid(data,geodict)
        gdalgrid.save('test.bil')
        
        newdict = {'xmin':3.0,'xmax':4.0,'ymin':3.0,'ymax':4.0,'xdim':1.0,'ydim':1.0}
        newdict = Grid2D.fillGeoDict(newdict)
        gdalgrid3 = GDALGrid.load('test.bil',samplegeodict=newdict,resample=True)
        output = np.array([[34,35],
                           [41,42]])
        np.testing.assert_almost_equal(gdalgrid3._data,output)
        print 'Passed resampling data without padding...'
        
        print 'Test resampling data with padding...'
        data,geodict = Grid2D._createSampleData(4,4)
        gdalgrid = GDALGrid(data,geodict)
        gdalgrid.save('test.bil')
        newdict = {'xmin':0.0,'xmax':4.0,'ymin':0.0,'ymax':4.0,'xdim':1.0,'ydim':1.0}
        newdict = Grid2D.fillGeoDict(newdict)
        gdalgrid3 = GDALGrid.load('test.bil',samplegeodict=newdict,resample=True,doPadding=True)
        output = np.array([[np.nan,np.nan,np.nan,np.nan,np.nan],
                           [np.nan,2.5,3.5,4.5,np.nan],
                           [np.nan,6.5,7.5,8.5,np.nan],
                           [np.nan,10.5,11.5,12.5,np.nan],
                           [np.nan,np.nan,np.nan,np.nan,np.nan]])
        np.testing.assert_almost_equal(gdalgrid3._data,output)
        print 'Passed resampling data with padding...'
    except AssertionError,error:
        print 'Failed resample test:\n %s' % error

    os.remove('test.bil')

def _meridian_test():
    try:
        print 'Testing resampling of global grid where sample crosses 180/-180 meridian...'
        data = np.arange(0,84).astype(np.int32).reshape(7,12)
        geodict = {'xmin':-180.0,'xmax':150.0,'ymin':-90.0,'ymax':90.0,'xdim':30,'ydim':30,'nrows':7,'ncols':12}
        gdalgrid = GDALGrid(data,geodict)
        gdalgrid.save('test.bil')

        sampledict = {'xmin':105,'xmax':-105,'ymin':-15.0,'ymax':15.0,'xdim':30.0,'ydim':30.0,'nrows':2,'ncols':5}
        gdalgrid5 = GDALGrid.load('test.bil',samplegeodict=sampledict,resample=True,doPadding=True)

        output = np.array([[ 39.5,40.5,35.5,30.5,31.5,32.5],
                           [ 51.5,52.5,47.5,42.5,43.5,44.5]])
        #output = np.random.rand(2,6) #this will fail assertion test
        np.testing.assert_almost_equal(gdalgrid5._data,output)
        print 'Passed resampling of global grid where sample crosses 180/-180 meridian...'

        print 'Testing resampling of global grid where sample crosses 180/-180 meridian and first column is duplicated by last...'
        data = np.arange(0,84).astype(np.int32).reshape(7,12)
        data = np.hstack((data,data[:,0].reshape(7,1)))
        geodict = {'xmin':-180.0,'xmax':180.0,'ymin':-90.0,'ymax':90.0,'xdim':30,'ydim':30,'nrows':7,'ncols':13}
        gdalgrid = GDALGrid(data,geodict)
        gdalgrid.save('test.bil')

        sampledict = {'xmin':105,'xmax':-105,'ymin':-15.0,'ymax':15.0,'xdim':30.0,'ydim':30.0,'nrows':2,'ncols':5}
        gdalgrid5 = GDALGrid.load('test.bil',samplegeodict=sampledict,resample=True,doPadding=True)

        output = np.array([[ 39.5,40.5,35.5,30.5,31.5,32.5],
                           [ 51.5,52.5,47.5,42.5,43.5,44.5]])
        #output = np.random.rand(2,6) #this will fail assertion test
        np.testing.assert_almost_equal(gdalgrid5._data,output)
        print 'Passed resampling of global grid where sample crosses 180/-180 meridian and first column is duplicated by last...'
        
    except AssertionError,error:
        print 'Failed meridian test:\n %s' % error
    os.remove('test.bil')
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        gdalfile = sys.argv[1]
        sampledict = None
        if len(sys.argv) == 6:
            xmin = float(sys.argv[2])
            xmax = float(sys.argv[3])
            ymin = float(sys.argv[4])
            ymax = float(sys.argv[5])
            sampledict = {'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax}
            grid = GDALGrid.load(gdalfile,samplegeodict=sampledict)
    else:
        _format_test()
        _pad_test()
        _subset_test()
        _resample_test()
        _meridian_test()
        

