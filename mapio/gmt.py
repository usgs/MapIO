#!/usr/bin/env python

#stdlib imports
import struct
import os.path
import sys

#third party imports
import numpy as np
from scipy.io import netcdf
from grid2d import Grid2D
from gridbase import Grid
from dataset import DataSetException
import h5py

NETCDF_TYPES = {'B':np.uint8,
                'b':np.int8,
                'h':np.int16,
                'i':np.int32,
                'f':np.float32,
                'd':np.float64}

INVERSE_NETCDF_TYPES = {'uint8':'B',
                        'int8':'b',
                        'int16':'h',
                        'int32':'i',
                        'float32':'f',
                        'float64':'d'}

def sub2ind(shape,subtpl):
    """
    Convert 2D subscripts into 1D index.
    @param shape: Tuple indicating size of 2D array.
    @param subtpl: Tuple of (possibly) numpy arrays of row,col values.
    @return: 1D array of indices.
    """
    if len(shape) != 2 or len(shape) != len(subtpl):
        raise IndexError, "Input size and subscripts must have length 2 and be equal in length"
    
    row,col = subtpl
    nrows,ncols = shape
    ind = ncols*row + col
    return ind

def indexArray(array,shp,i1,i2,j1,j2):
    if len(array.shape) == 1:
        nrows = i2-i1
        ncols = j2-j1
        if hasattr(array,'dtype'):
            data = np.zeros((nrows,ncols),dtype=array.dtype)
        else:
            typecode = array.typecode()
            dtype = NETCDF_TYPES[typecode]
            data = np.zeros((nrows,ncols),dtype=dtype)
        rowidx = np.arange(i1,i2)
        i = 0
        for row in rowidx:
            idx1 = sub2ind(shp,(row,j1))
            idx2 = sub2ind(shp,(row,j2))
            data[i,:] = array[idx1:idx2]
            i += 1
    else:
        data = array[i1:i2,j1:j2].copy()
    return data

def createSampleXRange(M,N,filename,bounds=None,xdim=None,ydim=None):
    if xdim is None:
        xdim = 1.0
    if ydim is None:
        ydim = 1.0
    if bounds is None:
        xmin = 0.5
        xmax = xmin + (N-1)*xdim
        ymin = 0.5
        ymax = ymin + (M-1)*ydim
    else:
        xmin,xmax,ymin,ymax = bounds
    data = np.arange(0,M*N).reshape(M,N).astype(np.int32)
    cdf = netcdf.netcdf_file(filename,'w')
    cdf.createDimension('side',2)
    cdf.createDimension('xysize',M*N)
    dim = cdf.createVariable('dimension','i',('side',))
    dim[:] = np.array([N,M])
    spacing = cdf.createVariable('spacing','i',('side',))
    spacing[:] = np.array([xdim,ydim])
    zrange = cdf.createVariable('z_range',INVERSE_NETCDF_TYPES[str(data.dtype)],('side',))
    zrange[:] = np.array([data.min(),data.max()])
    x_range = cdf.createVariable('x_range','d',('side',))
    x_range[:] = np.array([xmin,xmax])
    y_range = cdf.createVariable('y_range','d',('side',))
    y_range[:] = np.array([ymin,ymax])
    z = cdf.createVariable('z',INVERSE_NETCDF_TYPES[str(data.dtype)],('xysize',))
    z[:] = data.flatten()
    cdf.close()
    return data

def createSampleGrid(M,N):
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
    geodict = {'nrows':N,
               'ncols':N,
               'xmin':0.5,
               'xmax':xvar[-1],
               'ymin':0.5,
               'ymax':yvar[-1],
               'xdim':1.0,
               'ydim':1.0}
    gmtgrid = GMTGrid(data,geodict)
    return gmtgrid

class GMTGrid(Grid2D):
    """
    A class that implements a Grid2D object around GMT NetCDF/HDF/native gridded data sets.
    """
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
    def getFileType(cls,grdfile):
        """Get the GMT sub-format (netcdf, hdf, or GMT binary).
        :param grdfile:
          File name of suspected GMT grid file.
        :returns:
          One of 'netcdf' (NetCDF version 3), 'hdf' (NetCDF version 4), 'native' (so-called GMT native format), or 'unknown'.
        """
        f = open(grdfile,'rb')
        #check to see if it's HDF or CDF
        f.seek(1,0)
        hdfsig = ''.join(struct.unpack('ccc',f.read(3)))
        ftype = 'unknown'
        if hdfsig == 'HDF':
            ftype = 'hdf'
        else:
            f.seek(0,0)
            cdfsig = ''.join(struct.unpack('ccc',f.read(3)))
            if cdfsig == 'CDF':
                ftype = 'netcdf'
            else:
                f.seek(8,0)
                offset = struct.unpack('I',f.read(4))[0]
                if offset == 0 or offset == 1:
                    ftype = 'native'
                    
        f.close()
        return ftype

    @classmethod
    def getFileGeoDict(cls,filename):
        """Get the spatial extent, resolution, and shape of grid inside NetCDF file.
        :param filename:
           File name of NetCDF file.
        :returns:
          GeoDict specifying spatial extent, resolution, and shape of grid inside NetCDF file.
        :raises DataSetException:
          When the file is not detectable as one of the three flavors of GMT grids.
        """
        ftype = cls.getFileType(filename)
        if ftype == 'native':
            geodict,xvar,yvar,fmt,zscale,zoffset = cls.getNativeHeader(filename)
        elif ftype == 'netcdf':
            geodict,xvar,yvar = cls.getNetCDFHeader(filename)
        elif ftype == 'hdf':
            geodict,xvar,yvar = cls.getHDFHeader(filename)
        else:
            raise DataSetException('Unknown file type for file "%s".' % filename)
        return geodict

    @classmethod
    def getBoundsWithin(cls,filename,geodict):
        """
        Return a geodict for a file that is guaranteed to be contained by an input geodict.
        :param filename:
          Path to an GMT grid file.
        :param geodict:
          Geodict defining a spatial extent that we want to envelope the returned geodict.
        :returns:
          A geodict that is guaranteed to be contained by input geodict.
        """
        fgeodict = cls.getFileGeoDict(filename)
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
    def getNativeHeader(cls,fname,fmt=None):
        """Get the header information from a GMT native grid file.
        :param fname:
           File name of GMT native grid
        :param fmt:
          One of:
            - 'h' for 16 bit signed integer
            - 'i' for 32 bit signed integer
            - 'f' for 32 bit float
            - 'd' for 64 bit float
        :returns:
           - GeoDict specifying spatial extent, resolution, and shape of grid inside NetCDF file.
           - xvar array specifying X coordinates of data columns
           - yvar array specifying Y coordinates of data rows
           - fmt If input fmt is None, this will be a best guess as to the data format in the file.
           - zscale Data multiplier
           - zoffset Value to be added to data
        """
        f = open(fname,'rb')
        #Given that we can't automatically distinguish between 32 bit ints and 32 bit floats, we'll use
        #this value as a cutoff for "detecting" when a value read from disk as a float has an "unreasonably" 
        #high exponent value.  This is NOT guaranteed to work - use the fmt keyword if you want to be sure.
        MAX_FLOAT_EXP = 30
        fsize = os.path.getsize(fname)
        datalen = fsize-892
        f.seek(0,0)
        geodict = {}
        geodict['ncols'] = struct.unpack('I',f.read(4))[0]
        geodict['nrows'] = struct.unpack('I',f.read(4))[0]
        offset = struct.unpack('I',f.read(4))[0]
        geodict['xmin'] = struct.unpack('d',f.read(8))[0]
        geodict['xmax'] = struct.unpack('d',f.read(8))[0]
        geodict['ymin'] = struct.unpack('d',f.read(8))[0]
        geodict['ymax'] = struct.unpack('d',f.read(8))[0]
        zmin = struct.unpack('d',f.read(8))[0]
        zmax = struct.unpack('d',f.read(8))[0]
        geodict['xdim'] = struct.unpack('d',f.read(8))[0]
        geodict['ydim'] = struct.unpack('d',f.read(8))[0]
        zscale = struct.unpack('d',f.read(8))[0]
        zoffset = struct.unpack('d',f.read(8))[0]
        xunits = f.read(80).strip()
        yunits = f.read(80).strip()
        zunits = f.read(80).strip()
        title = f.read(80).strip()
        command = f.read(320).strip()
        remark = f.read(160).strip()
        npixels = geodict['nrows']*geodict['ncols']
        lenshort = npixels*2
        lenfloat = npixels*4
        lendouble = npixels*8
        if fmt is None:
            if datalen == lenshort:
                fmt = 'h'
            if datalen == lendouble:
                fmt = 'd'
            if datalen == lenfloat: #let's try to guess whether this is float or int
                fpos = f.tell()
                #read 1 byte, check to see if it's nan or 0 - if it is, then we definitely have a float
                dbytes = struct.unpack('f',f.read(4))[0]
                while dbytes == 0.0:
                    dbytes = struct.unpack('f',f.read(4))[0]
                f.seek(fpos) #go back to where we were
                if np.isnan(dbytes):
                    fmt = 'f'
                elif int(np.abs(np.log10(dbytes))) > MAX_FLOAT_EXP: #does this have a crazy large exponent?
                    fmt = 'i'
                else:
                    fmt = 'f'
        f.close()

        #We are going to represent all grids internally as grid-line registered
        #The difference between pixel and gridline-registered grids is depicted well here:
        #http://gmt.soest.hawaii.edu/doc/5.1.0/GMT_Docs.html#grid-registration-the-r-option
        if offset == 1:
            geodict['xmin'] += geodict['xdim']/2.0
            geodict['xmax'] -= geodict['xdim']/2.0
            geodict['ymin'] += geodict['ydim']/2.0
            geodict['ymax'] -= geodict['ydim']/2.0
        xvar = np.arange(geodict['xmin'],geodict['xmax']+geodict['xdim'])
        yvar = np.arange(geodict['ymin'],geodict['ymax']+geodict['ydim'])
        return (geodict,xvar,yvar,fmt,zscale,zoffset)

            
    @classmethod
    def readGMTNative(cls,fname,bounds=None,firstColumnDuplicated=False,fmt=None):
        """Read the data and geo-referencing information from a GMT native grid file, subsetting if requested.
        http://gmt.soest.hawaii.edu/doc/5.1.2/GMT_Docs.html#native-binary-grid-files
        :param fname:
          File name of GMT native grid
        :param bounds:
           Tuple of (xmin,xmax,ymin,ymax)
        :param firstColumnDuplicated:
           Boolean - is this a file where the last column of data is the same as the first (for grids that span entire globe).
        :param fmt: 
           Data width, one of: 
             - 'i' (16 bit signed integer)
             - 'l' (32 bit signed integer)
             - 'f' (32 bit float)
             - 'd' (64 bit float)
          Strictly speaking, this is only necessary when the data file is 32 bit float or 32 bit integer, as there is
          no *sure* way to tell from the header or data which data type is contained in the file.  If fmt is None, then
          the code will try to guess as best it can from the data whether it is integer or floating point data. Caveat emptor!
        :returns:
          Tuple of data (2D numpy array of data, possibly subsetted from file) and geodict (see above).
        :raises NotImplementedError:
          For any bounds not None (we'll get to it eventually!)
        """
        HDRLEN = 892
        geodict,xvar,yvar,fmt,zscale,zoffset = cls.getNativeHeader(fname,fmt)
        #for right now we're reading everything in then subsetting that.  Fix later with something
        #clever like memory mapping...
        sfmt = '%i%s' % (geodict['ncols']*geodict['nrows'],fmt)
        dwidths = {'h':2,'i':4,'f':4,'d':8}
        dwidth = dwidths[fmt]
        f = open(fname,'rb')
        f.seek(HDRLEN)
        dbytes = f.read(geodict['ncols']*geodict['nrows']*dwidth)
        if bounds is None:
            data = np.flipud(np.array(struct.unpack(sfmt,dbytes)))
            #data = np.array(data).reshape(geodict['nrows'],-1)
            data.shape = (geodict['nrows'],geodict['ncols'])
            if zscale != 1.0 or zoffset != 0.0:
                data = (data * zscale) + zoffset
            if firstColumnDuplicated:
                data = data[:,0:-1]
                geodict['xmax'] -= geodict['xdim']
        else:
            data = np.array(struct.unpack(sfmt,dbytes))
            data.shape = (geodict['nrows'],geodict['ncols'])
            data = np.fliplr(data)
            if zscale != 1.0 or zoffset != 0.0:
                data = (data * zscale) + zoffset
            if firstColumnDuplicated:
                data = data[:,0:-1]
                geodict['xmax'] -= geodict['xdim']
            data,geodict = cls._subsetRegions(data,bounds,geodict,xvar,yvar,firstColumnDuplicated)
        
        f.close()
        return (data,geodict)

    @classmethod
    def getNetCDFHeader(cls,filename):
        """Get the header information from a GMT NetCDF3 file.
        :param fname:
           File name of GMT NetCDF3 grid
        :returns:
           - GeoDict specifying spatial extent, resolution, and shape of grid inside NetCDF file.
           - xvar array specifying X coordinates of data columns
           - xvar array specifying Y coordinates of data rows
        """
        cdf = netcdf.netcdf_file(filename)
        geodict = {}
        xvarname = None
        registration = 'gridline'
        if hasattr(cdf,'node_offset') and getattr(cdf,'node_offset') == 1:
            registration = 'pixel'
        if 'x' in cdf.variables.keys():
            xvarname = 'x'
            yvarname = 'y'
        elif 'lon' in cdf.variables.keys():
            xvarname = 'lon'
            yvarname = 'lat'
        if xvarname is not None:
            xvar = cdf.variables[xvarname].data.copy()
            yvar = cdf.variables[yvarname].data.copy()
            geodict['ncols'] = len(xvar)
            geodict['nrows'] = len(yvar)
            geodict['xmin'] = xvar.min()
            geodict['xmax'] = xvar.max()
            geodict['ymin'] = yvar.min()
            geodict['ymax'] = yvar.max()
            newx = np.linspace(geodict['xmin'],geodict['xmax'],num=geodict['ncols'])
            newy = np.linspace(geodict['ymin'],geodict['ymax'],num=geodict['nrows'])
            geodict['xdim'] = newx[1]-newx[0]
            geodict['ydim'] = newy[1]-newy[0]
        elif 'x_range' in cdf.variables.keys():
            geodict['xmin'] = cdf.variables['x_range'].data[0]
            geodict['xmax'] = cdf.variables['x_range'].data[1]
            geodict['ymin'] = cdf.variables['y_range'].data[0]
            geodict['ymax'] = cdf.variables['y_range'].data[1]
            geodict['ncols'],geodict['nrows'] = cdf.variables['dimension'].data
            #geodict['xdim'],geodict['ydim'] = cdf.variables['spacing'].data
            xvar = np.linspace(geodict['xmin'],geodict['xmax'],num=geodict['ncols'])
            yvar = np.linspace(geodict['ymin'],geodict['ymax'],num=geodict['nrows'])
            geodict['xdim'] = xvar[1] - xvar[0]
            geodict['ydim'] = yvar[1] - yvar[0]
        else:
            raise DataSetException('No support for CDF data file with variables: %s' % str(cdf.variables.keys()))

        #We are going to represent all grids internally as grid-line registered
        #The difference between pixel and gridline-registered grids is depicted well here:
        #http://gmt.soest.hawaii.edu/doc/5.1.0/GMT_Docs.html#grid-registration-the-r-option
        if registration == 'pixel':
            geodict['xmin'] += geodict['xdim']/2.0
            geodict['xmax'] -= geodict['xdim']/2.0
            geodict['ymin'] += geodict['ydim']/2.0
            geodict['ymax'] -= geodict['ydim']/2.0
            
        return (geodict,xvar,yvar)

    @classmethod
    def _subsetRegions(self,zvar,bounds,fgeodict,xvar,yvar,firstColumnDuplicated):
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
        isScanLine = len(zvar.shape) == 1
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
            #data = np.flipud(zvar[:].copy())
            if isScanLine:
                data = indexArray(zvar,(gnrows,gncols),0,gnrows,0,gncols)
            else:
                data = np.flipud(indexArray(zvar,(gnrows,gncols),0,gnrows,0,gncols))
            if firstColumnDuplicated:
                data = data[:,0:-1]
                geodict['xmax'] -= geodict['xdim']
        else:
            if xmin > xmax:
                #cut user's request into two regions - one from the minimum to the
                #meridian, then another from the meridian to the maximum.
                (region1,region2) = self._createSections((xmin,xmax,ymin,ymax),fgeodict,firstColumnDuplicated,isScanLine=isScanLine)
                (iulx1,iuly1,ilrx1,ilry1) = region1
                (iulx2,iuly2,ilrx2,ilry2) = region2
                outcols1 = long(ilrx1-iulx1)
                outcols2 = long(ilrx2-iulx2)
                outcols = long(outcols1+outcols2)
                outrows = long(ilry1-iuly1)
                section1 = indexArray(zvar,(gnrows,gncols),iuly1,ilry1,iulx1,ilrx1)
                #section1 = zvar[iuly1:ilry1,iulx1:ilrx1].copy()
                #section2 = zvar[iuly2:ilry2,iulx2:ilrx2].copy()
                section2 = indexArray(zvar,(gnrows,gncols),iuly2,ilry2,iulx2,ilrx2)
                if isScanLine:
                    data = np.concatenate((section1,section2),axis=1)
                else:
                    data = np.flipud(np.concatenate((section1,section2),axis=1))
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

                if isScanLine:
                    iymin = int((gymax - ymax)/ydim)
                    iymax = int((gymax - ymin)/ydim)
                    fgeodict['ymax'] = gymax - iymin*ydim
                    fgeodict['ymin'] = gymax - iymax*ydim
                else:
                    iymin = np.where((ymin-yvar) >= 0)[0].max()
                    iymax = np.where((ymax-yvar) <= 0)[0].min()
                    fgeodict['ymin'] = yvar[iymin].copy()
                    fgeodict['ymax'] = yvar[iymax].copy()
                    
                fgeodict['xmin'] = xvar[ixmin].copy()
                fgeodict['xmax'] = xvar[ixmax].copy()
                
                if isScanLine:
                    data = indexArray(zvar,(gnrows,gncols),iymin,iymax+1,ixmin,ixmax+1)
                else:
                    data = np.flipud(indexArray(zvar,(gnrows,gncols),iymin,iymax+1,ixmin,ixmax+1))
                #data = np.flipud(zvar[iymin:iymax+1,ixmin:ixmax+1].copy())
                fgeodict['nrows'],fgeodict['ncols'] = data.shape

        return (data,fgeodict)
    
    @classmethod
    def readNetCDF(cls,filename,bounds=None,firstColumnDuplicated=False):
        """Read the data and geo-referencing information from a GMT NetCDF3 grid file, subsetting if requested.
        :param filename:
          File name of GMT NetCDF3 grid
        :param bounds:
           Tuple of (xmin,xmax,ymin,ymax)
        :returns:
          Tuple of data (2D numpy array of data, possibly subsetted from file) and geodict (see above).
        """
        geodict,xvar,yvar = cls.getNetCDFHeader(filename)
        cdf = netcdf.netcdf_file(filename)
        if bounds is None:
            nrows,ncols = (geodict['nrows'],geodict['ncols'])
            data = cdf.variables['z'].data.copy()
            shp = cdf.variables['z'].shape
            if len(shp) == 1: #sometimes the z array is flattened out, this should put it back
                data.shape = (nrows,ncols)
            if not cdf.variables.has_key('x_range'):
                data = np.flipud(data)
                
            if firstColumnDuplicated:
                data = data[:,0:-1]
                geodict['xmax'] -= geodict['xdim']
        else:
            data,geodict = cls._subsetRegions(cdf.variables['z'],bounds,geodict,xvar,yvar,firstColumnDuplicated)
        cdf.close()
        return (data,geodict)

    @classmethod
    def getHDFHeader(cls,hdffile):
        """Get the header information from a GMT NetCDF4 (HDF) file.
        :param fname:
           File name of GMT NetCDF4 grid
        :returns:
          GeoDict specifying spatial extent, resolution, and shape of grid inside NetCDF file.
        """
        geodict = {}
        f = h5py.File(hdffile,'r')
        registration = 'gridline'
        if f.get('node_offset') is not None and f.attrs['node_offset'][0] == 1:
            registration = 'pixel'
        if 'x' in f.keys():
            xvarname = 'x'
            yvarname = 'y'
        elif 'lon' in cdf.variables.keys():
            xvarname = 'lon'
            yvarname = 'lat'
        if xvarname is not None:
            xvar = f[xvarname][:]
            yvar = f[yvarname][:]
            geodict['nrows'] = len(yvar)
            geodict['ncols'] = len(xvar)
            geodict['xmin'] = xvar[0]
            geodict['xmax'] = xvar[-1]
            geodict['ymin'] = yvar[0]
            geodict['ymax'] = yvar[-1]
            newx = np.linspace(geodict['xmin'],geodict['xmax'],num=geodict['ncols'])
            newy = np.linspace(geodict['ymin'],geodict['ymax'],num=geodict['nrows'])
            geodict['xdim'] = newx[1]-newx[0]
            geodict['ydim'] = newy[1]-newy[0]
        else:
            geodict['xmin'] = f['x_range'][0]
            geodict['xmax'] = f['x_range'][1]
            geodict['ymin'] = f['y_range'][0]
            geodict['ymax'] = f['y_range'][1]
            geodict['ncols'],geodict['nrows'] = (f['dimension'][0],f['dimension'][1])
            xvar = np.linspace(geodict['xmin'],geodict['xmax'],num=ncols)
            yvar = np.linspace(geodict['ymin'],geodict['ymax'],num=nrows)
            geodict['xdim'] = xvar[1] - xvar[0]
            geodict['ydim'] = yvar[1] - yvar[0]

        #We are going to represent all grids internally as grid-line registered
        #The difference between pixel and gridline-registered grids is depicted well here:
        #http://gmt.soest.hawaii.edu/doc/5.1.0/GMT_Docs.html#grid-registration-the-r-option
        if registration == 'pixel':
            geodict['xmin'] += geodict['xdim']/2.0
            geodict['xmax'] -= geodict['xdim']/2.0
            geodict['ymin'] += geodict['ydim']/2.0
            geodict['ymax'] -= geodict['ydim']/2.0
        f.close()
        return (geodict,xvar,yvar)
        
    @classmethod
    def readHDF(cls,hdffile,bounds=None,firstColumnDuplicated=False):
        """Read the data and geo-referencing information from a GMT NetCDF4 (HDF) grid file, subsetting if requested.
        :param hdffile:
          File name of GMT NetCDF4 grid
        :param bounds:
           Tuple of (xmin,xmax,ymin,ymax)
        :returns:
          Tuple of data (2D numpy array of data, possibly subsetted from file) and geodict (see above).
        """
        #need a reproducible way of creating netcdf file in HDF format
        geodict,xvar,yvar = cls.getHDFHeader(hdffile)
        f = h5py.File(hdffile,'r')
        zvar = f['z']
        if bounds is None:
            data = np.flipud(zvar[:])
            if firstColumnDuplicated:
                data = data[:,0:-1]
                geodict['xmax'] -= geodict['xdim']
        else:
            data,geodict = cls._subsetRegions(f['z'],bounds,geodict,xvar,yvar,firstColumnDuplicated)
        f.close()
        return (data,geodict)

    def save(self,filename,format='netcdf'):
        """Save a GMTGrid object to a file.
        :param filename:
          Name of desired output file.
        :param format:
          One of 'netcdf','hdf' or 'native'.
        :raises DataSetException:
          When format not one of ('netcdf,'hdf','native')
        """
        if format not in ['netcdf','hdf','native']:
            raise DataSetException('Only NetCDF3, HDF (NetCDF4), and GMT native output are supported.')
        if format == 'netcdf':
            f = netcdf.NetCDFFile(filename,'w')
            m,n = self._data.shape
            xdim = f.createDimension('x',n)
            ydim = f.createDimension('y',m)
            x = f.createVariable('x',np.float64,('x'))
            y = f.createVariable('y',np.float64,('y'))
            x[:] = np.linspace(self._geodict['xmin'],self._geodict['xmax'],self._geodict['ncols'])
            y[:] = np.linspace(self._geodict['ymin'],self._geodict['ymax'],self._geodict['nrows'])
            z = f.createVariable('z',self._data.dtype,('y','x'))
            z[:] = np.flipud(self._data)
            f.close()
        elif format == 'hdf':
            #Create the file and the top-level attributes GMT wants to see
            f = h5py.File(filename,'w')
            f.attrs['Conventions'] = 'COARDS, CF-1.5'
            f.attrs['title'] = 'filename'
            f.attrs['history'] = 'Created with python GMTGrid.save(%s,format="hdf")' % filename
            f.attrs['GMT_version'] = 'NA'

            #Create the x array and the attributes of that GMT wants to see
            xvar = np.linspace(self._geodict['xmin'],self._geodict['xmax'],self._geodict['ncols'])
            x = f.create_dataset('x',data=xvar,shape=xvar.shape,dtype=str(xvar.dtype))
            x.attrs['CLASS'] = 'DIMENSION_SCALE'
            x.attrs['NAME'] = 'x'
            x.attrs['_Netcdf4Dimid'] = 0 #no idea what this is
            x.attrs['long_name'] = 'x'
            x.attrs['actual_range'] = np.array((xvar[0],xvar[-1]))

            #Create the x array and the attributes of that GMT wants to see
            yvar = np.linspace(self._geodict['ymin'],self._geodict['ymax'],self._geodict['nrows'])
            y = f.create_dataset('y',data=yvar,shape=yvar.shape,dtype=str(yvar.dtype))
            y.attrs['CLASS'] = 'DIMENSION_SCALE'
            y.attrs['NAME'] = 'y'
            y.attrs['_Netcdf4Dimid'] = 1 #no idea what this is
            y.attrs['long_name'] = 'y'
            y.attrs['actual_range'] = np.array((yvar[0],yvar[-1]))
            
            #create the z data set
            z = f.create_dataset('z',data=np.flipud(self._data),shape=self._data.shape,dtype=str(self._data.dtype))
            z.attrs['long_name'] = 'z'
            #zvar.attrs['_FillValue'] = array([ nan], dtype=float32)
            z.attrs['actual_range'] = np.array((np.nanmin(self._data),np.nanmax(self._data)))
            
            #close the hdf file
            f.close()
        elif format == 'native':
            f = open(filename,'w')
            f.write(struct.pack('I',self._geodict['ncols']))
            f.write(struct.pack('I',self._geodict['nrows']))
            f.write(struct.pack('I',0)) #gridline registration
            f.write(struct.pack('d',self._geodict['xmin']))
            f.write(struct.pack('d',self._geodict['xmax']))
            f.write(struct.pack('d',self._geodict['ymin']))
            f.write(struct.pack('d',self._geodict['ymax']))
            f.write(struct.pack('d',self._data.min()))
            f.write(struct.pack('d',self._data.max()))
            f.write(struct.pack('d',self._geodict['xdim']))
            f.write(struct.pack('d',self._geodict['ydim']))
            f.write(struct.pack('d',1.0)) #scale factor to multiply data by
            f.write(struct.pack('d',0.0)) #offfset to add to data
            f.write(struct.pack('80s','X units (probably degrees)'))
            f.write(struct.pack('80s','Y units (probably degrees)'))
            f.write(struct.pack('80s','Z units unknown'))
            f.write(struct.pack('80s','')) #title
            f.write(struct.pack('320s','Created with GMTGrid() class, a product of the NEIC.')) #command
            f.write(struct.pack('160s','')) #remark
            if self._data.dtype not in [np.int16,np.int32,np.float32,np.float64]:
                raise DataSetException('Data type of "%s" is not supported by the GMT native format.' % str(self._data.dtype))
            fpos1 = f.tell()
            newdata = np.fliplr(np.flipud(self._data[:])) #the left-right flip is necessary because of the way tofile() works
            newdata.tofile(f)
            fpos2 = f.tell()
            bytesout = fpos2 - fpos1
            f.close()
            
    
    @classmethod
    def load(cls,gmtfilename,samplegeodict=None,resample=False,method='linear',doPadding=False,padValue=np.nan):
        """Create a GMTGrid object from a (possibly subsetted, resampled, or padded) GMT grid file.
        :param gmtfilename:
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
          GMTgrid instance (possibly subsetted, padded, or resampled)
        :raises DataSetException:
          * When sample bounds are outside (or too close to outside) the bounds of the grid and doPadding=False.
          * When the input file type is not recognized.
        """
        ftype = cls.getFileType(gmtfilename)
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
                filegeodict = cls.getFileGeoDict(gmtfilename)
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
        
        if ftype == 'native':
            #we're dealing with a binary "native" GMT grid file
            data,geodict = cls.readGMTNative(gmtfilename,samplebounds,firstColumnDuplicated)
        elif ftype == 'netcdf':
            data,geodict = cls.readNetCDF(gmtfilename,samplebounds,firstColumnDuplicated)
        elif ftype == 'hdf':
            data,geodict = cls.readHDF(gmtfilename,samplebounds,firstColumnDuplicated)
        else:
            raise DataSetException('File type "%s" cannot be read.' % ftype)
        
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
            
        
class BinCDFArray(object):
    def __init__(self,array,nrows,ncols):
        self.array = array
        self.nrows = nrows
        self.ncols = ncols

    def __getitem__(self,*args):
        """Allows slicing of CDF data array in the same way as a numpy array."""
        if len(args) == 1 and isinstance(args[0][0],int):
            #user has passed in a tuple of row,col - they only want one value
            row = args[0][0]
            col = args[0][1]
            nrows = self.nrows
            ncols = self.ncols
            if row < 0 or row > nrows-1:
                raise Exception,"Row index out of bounds"
            if col < 0 or col > ncols-1:
                raise Exception,"Row index out of bounds"
            idx = ncols * row + col
            offset = 0
            return self.array[idx]

        if len(args) == 1 and isinstance(args[0][0],slice): #they want a non-scalar subset of the data
            nrows = self.nrows
            ncols = self.ncols
            #calculate offset to first data element
            key1 = args[0][0]
            key2 = args[0][1]
            rowstart = key1.start
            rowend = key1.stop
            rowstep = key1.step
            colstart = key2.start
            colend = key2.stop
            colstep = key2.step
            
            if rowstep is None:
                rowstep = 1
            if colstep is None:
                colstep = 1

            #error checking
            if rowstart < 0 or rowstart > nrows-1:
                raise Exception,"Row index out of bounds"
            if rowend < 0 or rowend > nrows:
                raise Exception,"Row index out of bounds"
            if colstart < 0 or colstart > ncols-1:
                raise Exception,"Col index out of bounds"
            if colend < 0 or colend > ncols:
                raise Exception,"Col index out of bounds"

            colcount = (colend-colstart)
            rowcount = (rowend-rowstart)
            outrows = np.ceil(rowcount/rowstep)
            outcols = np.ceil(colcount/colstep)
            data = np.zeros([outrows,outcols],dtype=self.dtype)
            outrow = 0
            for row in range(int(rowstart),int(rowend),int(rowstep)):
                #just go to the beginning of the row, we're going to read in the whole line
                idx = ncols*row 
                offset = self.dwidth*idx #beginning of row
                line = self.array[idx:idx+ncols]
                data[outrow,:] = line[colstart:colend:colstep]
                outrow = outrow+1
                
        else:
            raise Exception, "Unsupported __getitem__ input %s" % (str(key))
        return(data)

def _save_test():
    try:
        print 'Testing saving and loading to/from NetCDF3...'
        #make a sample data set
        gmtgrid = createSampleGrid(4,4)

        #save it as netcdf3
        gmtgrid.save('test.grd',format='netcdf')
        gmtgrid2 = GMTGrid.load('test.grd')
        np.testing.assert_almost_equal(gmtgrid._data,gmtgrid2._data)
        print 'Passed saving and loading to/from NetCDF3.'

        print 'Testing saving and loading to/from NetCDF4 (HDF)...'
        #save it as HDF
        gmtgrid.save('test.grd',format='hdf')
        gmtgrid3 = GMTGrid.load('test.grd')
        np.testing.assert_almost_equal(gmtgrid._data,gmtgrid3._data)
        print 'Passed saving and loading to/from NetCDF4 (HDF)...'

        print 'Testing saving and loading to/from GMT native)...'
        gmtgrid.save('test.grd',format='native')
        gmtgrid4 = GMTGrid.load('test.grd')
        np.testing.assert_almost_equal(gmtgrid._data,gmtgrid4._data)
        print 'Passed saving and loading to/from GMT native...'
    except AssertionError,error:
        print 'Failed padding test:\n %s' % error
    os.remove('test.grd')

def _pad_test():
    try:
        for fmt in ['netcdf','hdf','native']:
            print 'Test padding data with null values (format %s)...' % fmt
            gmtgrid = createSampleGrid(4,4)
            gmtgrid.save('test.grd',format=fmt)

            newdict = {'xmin':-0.5,'xmax':4.5,'ymin':-0.5,'ymax':4.5,'xdim':1.0,'ydim':1.0}
            gmtgrid2 = GMTGrid.load('test.grd',samplegeodict=newdict,doPadding=True)
            output = np.array([[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                               [np.nan,0.0,1.0,2.0,3.0,np.nan],
                               [np.nan,4.0,5.0,6.0,7.0,np.nan],
                               [np.nan,8.0,9.0,10.0,11.0,np.nan],
                               [np.nan,12.0,13.0,14.0,15.0,np.nan],
                               [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]])
            np.testing.assert_almost_equal(gmtgrid2._data,output)
            print 'Passed padding data with format %s.' % fmt
    except AssertionError,error:
        print 'Failed padding test:\n %s' % error
    os.remove('test.grd')

def _subset_test():
    try:
        for fmt in ['netcdf','hdf','native']:
            print 'Testing subsetting of non-square grid (format %s)...' % fmt
            data = np.arange(0,24).reshape(6,4).astype(np.int32)
            geodict = {'xmin':0.5,'xmax':3.5,'ymin':0.5,'ymax':5.5,'xdim':1.0,'ydim':1.0,'nrows':6,'ncols':4}
            gmtgrid = GMTGrid(data,geodict)
            gmtgrid.save('test.grd',format=fmt)
            newdict = {'xmin':1.5,'xmax':2.5,'ymin':1.5,'ymax':3.5,'xdim':1.0,'ydim':1.0}
            gmtgrid3 = GMTGrid.load('test.grd',samplegeodict=newdict)
            output = np.array([[9,10],
                               [13,14],
                               [17,18]])
            np.testing.assert_almost_equal(gmtgrid3._data,output)
            print 'Passed subsetting of non-square grid (format %s)...' % fmt
        
    except AssertionError,error:
        print 'Failed subset test:\n %s' % error

    os.remove('test.grd')

def _resample_test():
    try:
        for fmt in ['netcdf','hdf','native']:
            print 'Test resampling data without padding (format %s)...' % fmt
            data = np.arange(0,63).astype(np.int32).reshape(9,7)
            geodict = {'xmin':0.5,'xmax':6.5,'ymin':0.5,'ymax':8.5,'xdim':1.0,'ydim':1.0,'nrows':9,'ncols':7}
            gmtgrid = GMTGrid(data,geodict)
            gmtgrid.save('test.grd',format=fmt)

            newdict = {'xmin':3.0,'xmax':4.0,'ymin':3.0,'ymax':4.0,'xdim':1.0,'ydim':1.0}
            newdict = Grid.fillGeoDict(newdict)
            gmtgrid3 = GMTGrid.load('test.grd',samplegeodict=newdict,resample=True)
            output = np.array([[34,35],
                               [41,42]])
            np.testing.assert_almost_equal(gmtgrid3._data,output)
            print 'Passed resampling data without padding (format %s)...' % fmt

            print 'Test resampling data with padding (format %s)...' % fmt
            gmtgrid = createSampleGrid(4,4)
            gmtgrid.save('test.grd',format=fmt)
            newdict = {'xmin':0.0,'xmax':4.0,'ymin':0.0,'ymax':4.0,'xdim':1.0,'ydim':1.0}
            newdict = Grid.fillGeoDict(newdict)
            gmtgrid3 = GMTGrid.load('test.grd',samplegeodict=newdict,resample=True,doPadding=True)
            output = np.array([[np.nan,np.nan,np.nan,np.nan,np.nan],
                               [np.nan,2.5,3.5,4.5,np.nan],
                               [np.nan,6.5,7.5,8.5,np.nan],
                               [np.nan,10.5,11.5,12.5,np.nan],
                               [np.nan,np.nan,np.nan,np.nan,np.nan]])
            np.testing.assert_almost_equal(gmtgrid3._data,output)
            print 'Passed resampling data with padding (format %s)...' % fmt
    except AssertionError,error:
        print 'Failed resample test:\n %s' % error

    os.remove('test.grd')
    
def _meridian_test():
    try:
        for fmt in ['netcdf','hdf','native']:
            print 'Testing resampling of global grid where sample crosses 180/-180 meridian (format %s)...' % fmt
            data = np.arange(0,84).astype(np.int32).reshape(7,12)
            geodict = {'xmin':-180.0,'xmax':150.0,'ymin':-90.0,'ymax':90.0,'xdim':30,'ydim':30,'nrows':7,'ncols':12}
            gmtgrid = GMTGrid(data,geodict)
            gmtgrid.save('test.grd',format=fmt)

            sampledict = {'xmin':105,'xmax':-105,'ymin':-15.0,'ymax':15.0,'xdim':30.0,'ydim':30.0,'nrows':2,'ncols':5}
            gmtgrid5 = GMTGrid.load('test.grd',samplegeodict=sampledict,resample=True,doPadding=True)

            output = np.array([[ 39.5,40.5,35.5,30.5,31.5,32.5],
                               [ 51.5,52.5,47.5,42.5,43.5,44.5]])
            #output = np.random.rand(2,6) #this will fail assertion test
            np.testing.assert_almost_equal(gmtgrid5._data,output)
            print 'Passed resampling of global grid where sample crosses 180/-180 meridian (format %s)...' % fmt

            print 'Testing resampling of global grid where sample crosses 180/-180 meridian and first column is duplicated by last (format %s)...' % fmt
            data = np.arange(0,84).astype(np.int32).reshape(7,12)
            data = np.hstack((data,data[:,0].reshape(7,1)))
            geodict = {'xmin':-180.0,'xmax':180.0,'ymin':-90.0,'ymax':90.0,'xdim':30,'ydim':30,'nrows':7,'ncols':13}
            gmtgrid = GMTGrid(data,geodict)
            gmtgrid.save('test.grd')

            sampledict = {'xmin':105,'xmax':-105,'ymin':-15.0,'ymax':15.0,'xdim':30.0,'ydim':30.0,'nrows':2,'ncols':5}
            gmtgrid5 = GMTGrid.load('test.grd',samplegeodict=sampledict,resample=True,doPadding=True)

            output = np.array([[ 39.5,40.5,35.5,30.5,31.5,32.5],
                               [ 51.5,52.5,47.5,42.5,43.5,44.5]])
            #output = np.random.rand(2,6) #this will fail assertion test
            np.testing.assert_almost_equal(gmtgrid5._data,output)
            print 'Passed resampling of global grid where sample crosses 180/-180 meridian and first column is duplicated by last (format %s)...' % fmt
        
    except AssertionError,error:
        print 'Failed meridian test:\n %s' % error
    os.remove('test.grd')

def _index_test():
    data = np.arange(0,42).reshape(6,7)
    d2 = data.flatten()
    shp = data.shape
    res1 = data[1:3,1:3]
    res2 = indexArray(d2,shp,1,3,1,3)
    np.testing.assert_almost_equal(res1,res2)

def _xrange_test():
    #there is a type of GMT netcdf file where the data is in scanline order
    #we don't care enough to support this in the save() method, but we do need a test for it.  Sigh.
    try:
        print 'Testing loading whole x_range style grid...'
        data = createSampleXRange(6,4,'test.grd')
        gmtgrid = GMTGrid.load('test.grd')
        np.testing.assert_almost_equal(data,gmtgrid.getData())
        print 'Passed loading whole x_range style grid...'

        print 'Testing loading partial x_range style grid...'
        #test with subsetting
        newdict = {'xmin':1.5,'xmax':2.5,'ymin':1.5,'ymax':3.5,'xdim':1.0,'ydim':1.0}
        gmtgrid3 = GMTGrid.load('test.grd',samplegeodict=newdict)
        output = np.array([[9,10],
                           [13,14],
                           [17,18]])
        np.testing.assert_almost_equal(gmtgrid3._data,output)
        print 'Passed loading partial x_range style grid...'

        print 'Testing x_range style grid where we cross meridian...'
        data = createSampleXRange(7,12,'test.grd',(-180.,150.,-90.,90.),xdim=30.,ydim=30.)
        sampledict = {'xmin':105,'xmax':-105,'ymin':-15.0,'ymax':15.0,'xdim':30.0,'ydim':30.0,'nrows':2,'ncols':5}
        gmtgrid5 = GMTGrid.load('test.grd',samplegeodict=sampledict,resample=True,doPadding=True)
        print 'Testing x_range style grid where we cross meridian...'
        
    except AssertionError,error:
        print 'Failed an xrange test:\n %s' % error
    os.remove('test.grd')    

def _within_test():
    try:
        print 'Testing class method getBoundsWithin()...'
        gmtgrid = createSampleGrid(8,8)
        gmtgrid.save('test.grd',format='netcdf')
        sdict = {'xmin':2.7,'xmax':6.7,'ymin':2.7,'ymax':6.7}
        newdict = GMTGrid.getBoundsWithin('test.grd',sdict)
        testdict = {'xmin':4.5,'xmax':5.5,'ymin':4.5,'ymax':5.5,'xdim':1.0,'ydim':1.0}
        assert newdict == testdict
        gmtgrid2 = GMTGrid.load('test.grd',samplegeodict=newdict)
        output = np.array([[20,21],
                           [28,29]])
        np.testing.assert_almost_equal(gmtgrid2.getData(),output)
        print 'Passed class method getBoundsWithin()...'
    except AssertionError,error:
        print 'Failed test of getBoundsWithin():\n %s' % error
    os.remove('test.grd')
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        gmtfile = sys.argv[1]
        sampledict = None
        if len(sys.argv) == 6:
            xmin = float(sys.argv[2])
            xmax = float(sys.argv[3])
            ymin = float(sys.argv[4])
            ymax = float(sys.argv[5])
            sampledict = {'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax}
            grid = GMTGrid.load(gmtfile,samplegeodict=sampledict)
    else:
        _index_test()
        _save_test()
        _pad_test()
        _subset_test()
        _resample_test()
        _meridian_test()
        _xrange_test()
        _within_test()
