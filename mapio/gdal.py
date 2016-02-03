#!/usr/bin/env python

#python 3 compatibility
from __future__ import print_function

#stdlib imports
import os.path
import sys
from collections import OrderedDict
import warnings

#third party imports
import rasterio
import numpy as np
from .grid2d import Grid2D
from .dataset import DataSetException,DataSetWarning
from .geodict import GeoDict

class GDALGrid(Grid2D):
    def __init__(self,data,geodict):
        """Construct a GMTGrid object.
        :param data:
           2D numpy data array (must match geodict spec)
        :param geodict:
           GeoDict Object specifying the spatial extent,resolution and shape of the data.
        :returns:
           A GMTGrid object.
        :raises DataSetException:
          When data and geodict dimensions do not match. 
        """
        m,n = data.shape
        if m != geodict.nrows or n != geodict.ncols:
            raise DataSetException('Input geodict does not match shape of input data.')
        self._data = data
        self._geodict = geodict

    @classmethod
    def getFileGeoDict(cls,filename):
        """Get the spatial extent, resolution, and shape of grid inside ESRI grid file.
        :param filename:
           File name of ESRI grid file.
        :returns:
           - GeoDict object specifying spatial extent, resolution, and shape of grid inside ESRI grid file.
           - xvar array specifying X coordinates of data columns
           - yvar array specifying Y coordinates of data rows
        :raises DataSetException:
          When the file contains a grid with more than one band.
          When the file geodict is internally inconsistent.
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

                gd = GeoDict(geodict)
        xvar = np.arange(gd.xmin,gd.xmax+gd.xdim,gd.xdim)
        yvar = np.arange(gd.ymin,gd.ymax+gd.ydim,gd.ydim)
        xvar2,xdim2 = np.linspace(gd.xmin,gd.xmax,num=gd.ncols,retstep=True)
        yvar2,ydim2 = np.linspace(gd.ymin,gd.ymax,num=gd.nrows,retstep=True)
        return (gd,xvar,yvar)

    @classmethod
    def getBoundsWithin(cls,filename,geodict):
        """
        Return a geodict object for a file that is guaranteed to be contained by an input geodict.
        :param filename:
          Path to an ESRI grid file.
        :param geodict:
          Geodict object defining a spatial extent that we want to envelope the returned geodict.
        :returns:
          A geodict that is guaranteed to be contained by input geodict.
          
        """
        fgeodict,xvar,yvar = cls.getFileGeoDict(filename)
        fxmin,fxmax,fymin,fymax = (fgeodict.xmin,fgeodict.xmax,fgeodict.ymin,fgeodict.ymax)
        xmin,xmax,ymin,ymax = (geodict.xmin,geodict.xmax,geodict.ymin,geodict.ymax)
        fxdim,fydim = (fgeodict.xdim,fgeodict.ydim)
        
        ulcol = int(np.ceil((xmin - fxmin)/fxdim))+1
        ulrow = int(np.floor((ymax - fymin)/fydim))-1
        lrcol = int(np.floor((xmax - fxmin)/fxdim))-1
        lrrow = int(np.ceil((ymin-fymin)/fydim))+1

        newxmin = fxmin + ulcol*fxdim
        newxmax = fxmin + lrcol*fxdim
        newymax = fymin + ulrow*fydim
        newymin = fymin + lrrow*fydim

        #outgeodict = {'xmin':newxmin,'xmax':newxmax,'ymin':newymin,'ymax':newymax,'xdim':fxdim,'ydim':fydim}
        eps = 1e-12
        ncols = int((newxmax-newxmin)/fxdim + eps) + 1
        nrows = int((newymax-newymin)/fydim + eps) + 1
        outgeodict = GeoDict({'xmin':newxmin,'xmax':newxmax,
                              'ymin':newymin,'ymax':newymax,
                              'xdim':fxdim,'ydim':fydim,
                              'nrows':nrows,'ncols':ncols},preserve='dims')
        return outgeodict
    
    @classmethod
    def _subsetRegions(self,src,sampledict,fgeodict,xvar,yvar,firstColumnDuplicated):
        """Internal method used to do subsampling of data for all three GMT formats.
        :param zvar:
          A numpy array-like thing (CDF/HDF variable, or actual numpy array)
        :param sampledict:
          GeoDict object with bounds and row/col information.
        :param fgeodict:
          GeoDict object with the file information.
        :param xvar:
          Numpy array specifying X coordinates of data columns
        :param yvar:
          Numpy array specifying Y coordinates of data rows
        :param firstColumnDuplicated:
          Boolean - is this a file where the last column of data is the same as the first (for grids that span entire globe).
        :returns:
          Tuple of (data,geodict) (subsetted data and geodict describing that data).
        """
        txmin,txmax,tymin,tymax = (sampledict.xmin,sampledict.xmax,sampledict.ymin,sampledict.ymax)
        trows,tcols = (sampledict.nrows,sampledict.ncols)
        #we're not doing anything fancy with the data here, just cutting out what we need
        xmin = max(fgeodict.xmin,txmin)
        xmax = min(fgeodict.xmax,txmax)
        ymin = max(fgeodict.ymin,tymin)
        ymax = min(fgeodict.ymax,tymax)
        
        #these are the bounds of the whole file
        gxmin = fgeodict.xmin
        gxmax = fgeodict.xmax
        gymin = fgeodict.ymin
        gymax = fgeodict.ymax
        xdim = fgeodict.xdim
        ydim = fgeodict.ydim
        gnrows = fgeodict.nrows
        gncols = fgeodict.ncols
        geodict = None
        if xmin == gxmin and xmax == gxmax and ymin == gymin and ymax == gymax:
            #just read the whole file
            tfdict = fgeodict.asDict()
            data = src.read()
            data = np.squeeze(data)
            if firstColumnDuplicated:
                data = data[:,0:-1]
                tfdict['xmax'] -= geodict.xdim
                tfdict['ncols'] -= 1
            geodict = GeoDict(tfdict)
        else:
            #what are the nearest grid coordinates to our desired bounds?
            #for example, if the grid starts at xmin = 0.5 and xmax = 6.5 with xdim=1.0, 
            #and the user wants txmin = 2.0 and txmax of 4.0, the pixel coordinates
            #(erring on the side of including more data), would be at txmin = 1.5 and 
            #txmax = 4.5.
            txmin2 = gxmin + xdim*np.floor((txmin - gxmin)/xdim)
            txmax2 = gxmin + xdim*np.ceil((txmax - gxmin)/xdim)
            tymin2 = gymin + ydim*np.floor((tymin - gymin)/ydim)
            tymax2 = gymin + ydim*np.ceil((tymax - gymin)/ydim)
            if xmin > xmax:
                #cut user's request into two regions - one from the minimum to the
                #meridian, then another from the meridian to the maximum.
                #new create sections algorithm
                #get section from the xmin to the 180 meridian
                iuly1,iulx1 = fgeodict.getRowCol(tymax2,txmin2)
                ilry1,ilrx1 = fgeodict.getRowCol(tymin2,fgeodict.xmax)
                #get section from the 180 meridian to xmax
                iuly2,iulx2 = fgeodict.getRowCol(tymax2,fgeodict.xmin)
                ilry2,ilrx2 = fgeodict.getRowCol(tymin2,txmax2)

                if firstColumnDuplicated:
                    ilrx1 -= 1

                tnrows = (ilry1 - iuly1)+1
                tncols = (ilrx1 - iulx1)+1 + (ilrx2 - iulx2)+1
                
                #(region1,region2) = self._createSections((xmin,xmax,ymin,ymax),fgeodict,firstColumnDuplicated)
                #(iulx1,iuly1,ilrx1,ilry1) = region1
                #(iulx2,iuly2,ilrx2,ilry2) = region2
                window1 = ((iuly1,ilry1+1),(iulx1,ilrx1+1))
                window2 = ((iuly2,ilry2+1),(iulx2,ilrx2+1))
                section1 = src.read(1,window=window1)
                section2 = src.read(1,window=window2)
                data = np.hstack((section1,section2))
                tfdict = {}
                newymax,newxmin = fgeodict.getLatLon(iuly1,iulx1)
                newymin,newxmax = fgeodict.getLatLon(ilry2,ilrx2)
                tfdict['xmin'] = newxmin
                tfdict['xmax'] = newxmax
                tfdict['ymin'] = newymin
                tfdict['ymax'] = newymax
                tfdict['xdim'] = xdim
                tfdict['ydim'] = ydim
                tfdict['nrows'],tfdict['ncols'] = data.shape
                geodict = GeoDict(tfdict)
            else:
                iuly,iulx = fgeodict.getRowCol(tymax2,txmin2)
                ilry,ilrx = fgeodict.getRowCol(tymin2,txmax2)
                tnrows = (ilry - iuly)+1
                tncols = (ilrx - iulx)+1

                window = ((iuly,ilry+1),(iulx,ilrx+1))
                tfdict = {}
                newymax,newxmin = fgeodict.getLatLon(iuly,iulx)
                newymin,newxmax = fgeodict.getLatLon(ilry,ilrx)
                tfdict['xmin'] = newxmin
                tfdict['xmax'] = newxmax
                tfdict['ymin'] = newymin
                tfdict['ymax'] = newymax
                tfdict['xdim'] = xdim
                tfdict['ydim'] = ydim
                #window = ((iymin,iymax+1),(ixmin,ixmax+1))
                data = src.read(1,window=window)
                data = np.squeeze(data)
                tfdict['nrows'],tfdict['ncols'] = data.shape
                geodict = GeoDict(tfdict)
            
        
        return (data,geodict)

    @classmethod
    def readGDAL(cls,filename,sampledict=None,firstColumnDuplicated=False):
        """
        Read an ESRI flt/bip/bil/bsq formatted file using rasterIO (GDAL Python wrapper).
        :param filename:
          Input ESRI formatted grid file.
        :param sampledict:
          GeoDict object containing bounds, x/y dims, nrows/ncols.
        :param firstColumnDuplicated:
          Indicate whether the last column is the same as the first column.
        :returns:
          A tuple of (data,geodict) where data is a 2D numpy array of all data found inside bounds, and 
          geodict gives the geo-referencing information for the data.
        """
        fgeodict,xvar,yvar = cls.getFileGeoDict(filename)
        data = None
        with rasterio.drivers():
            with rasterio.open(filename) as src:
                if sampledict is None:
                    data = src.read()
                    data = np.squeeze(data)
                    tgeodict = fgeodict.asDict()
                    if firstColumnDuplicated:
                        data = data[:,0:-1]
                        tgeodict['xmax'] -= geodict['xdim']
                        geodict = GeoDict(tgeodict,preserve='dims')
                else:
                    data,geodict = cls._subsetRegions(src,sampledict,fgeodict,xvar,yvar,firstColumnDuplicated)
        #Put NaN's back in where nodata value was
        nodata = src.get_nodatavals()[0]
        if nodata is not None and data.dtype in [np.float32,np.float64]: #NaNs only valid for floating point data
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
        hdr['ULXMAP'] = self._geodict.xmin
        hdr['ULYMAP'] = self._geodict.ymax
        hdr['XDIM'] = self._geodict.xdim
        hdr['YDIM'] = self._geodict.ydim
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
        """
        Save the data contained in this grid to a float or integer ESRI grid file.  Described here:
        http://webhelp.esri.com/arcgisdesktop/9.3/index.cfm?TopicName=BIL,_BIP,_and_BSQ_raster_files
        http://resources.esri.com/help/9.3/arcgisdesktop/com/gp_toolref/conversion_tools/float_to_raster_conversion_.htm.

        :param filename:
          String representing file to which data should be saved.
        :param format:
          Currently this code only supports the GDAL format 'EHdr' (see formats above.)  As rasterIO write support is expanded, this code should add functionality accordingly.
        :raises DataSetException:
          When format is not 'EHdr'.        
        """
        
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
        for (key,value) in hdr.items():
            value = hdr[key]
            f.write('%s  %s\n' % (key,str(value)))
        f.close()
            
    
    @classmethod
    def load(cls,filename,samplegeodict=None,preserve='dims',resample=False,method='linear',doPadding=False,padValue=np.nan):
        """Create a GDALGrid object from a (possibly subsetted, resampled, or padded) GDAL-compliant grid file.
        :param filename:
          Name of input file.
        :param samplegeodict:
          GeoDict object used to specify subset bounds and resolution (if resample is selected)
        :param preserve:
          String (one of 'dims','shape') indicating whether xdim/ydim of input geodict should be preserved or nrows/ncols.
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
        filegeodict,xvar,yvar = cls.getFileGeoDict(filename)
        #verify that if not resampling, the dimensions of the sampling geodict must match the file.
        if resample == False and samplegeodict is not None:
            dxdim = np.abs(filegeodict.xdim - samplegeodict.xdim)
            dydim = np.abs(filegeodict.ydim - samplegeodict.ydim)
            if dxdim > GeoDict.EPS or dxdim > GeoDict.EPS:
                raise DataSetException('File dimensions are different from sampledict dimensions.') 

        
        data = None
        geodict = None
        bounds = None
        sampledict = None
        firstColumnDuplicated = False
        if samplegeodict is not None:
            bounds = (samplegeodict.xmin,samplegeodict.xmax,samplegeodict.ymin,samplegeodict.ymax)
            samplebounds = bounds
            #if the user wants resampling, we can't just read the bounds they asked for, but instead
            #go outside those bounds.  if they asked for padding and the input bounds exceed the bounds
            #of the file, then we can pad.  If they *didn't* ask for padding and input exceeds, raise exception.
            if resample:
                PADFACTOR = 2 #how many cells will we buffer out for resampling?
                
                xdim = filegeodict.xdim
                ydim = filegeodict.ydim
                fbounds = (filegeodict.xmin,filegeodict.xmax,filegeodict.ymin,filegeodict.ymax)
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
                #does this dictionary need to be on the file boundaries?
                sdict = {'xmin':samplebounds[0],
                         'xmax':samplebounds[1],
                         'ymin':samplebounds[2],
                         'ymax':samplebounds[3],
                         'xdim':samplegeodict.xdim,
                         'ydim':samplegeodict.ydim,
                         'nrows':2,
                         'ncols':2}
                sampledict = GeoDict(sdict,preserve='dims')
            else:
                sampledict = samplegeodict
                         
                    
        data,geodict = cls.readGDAL(filename,sampledict,firstColumnDuplicated=firstColumnDuplicated)

        if doPadding:
            #up to this point, all we've done is either read in the whole file or cut out (along existing
            #boundaries) the section of data we want.  Now we do padding as necessary.
            #_getPadding is a class method inherited from Grid (our grandparent)
            #because we're in a class method, we have to do some gymnastics to call it.
            leftpad,rightpad,bottompad,toppad,geodict = super(Grid2D,cls)._getPadding(geodict,sampledict,padValue)
            data = np.hstack((leftpad,data))
            data = np.hstack((data,rightpad))
            data = np.vstack((toppad,data))
            data = np.vstack((data,bottompad))
        #if the user asks to resample, take the (possibly cut and padded) data set, and resample
        #it using the Grid2D super class
        if resample:
            grid = Grid2D(data,geodict)
            if samplegeodict.xmin > samplegeodict.xmax:
                samplegeodict.xmax += 360
            grid.interpolateToGrid(samplegeodict,method=method)
            data = grid.getData()
            geodict = grid.getGeoDict()
        return cls(data,geodict)


