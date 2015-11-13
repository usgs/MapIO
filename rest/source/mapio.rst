mapio package
=============

mapio.grid2d module
--------------------

Grid subclass for manipulating 2D geospatial data grids.

Usage:

Creating a Grid2D object.

::

   nrows = 100
   ncols = 100
   data = np.random.rand(100,100)
   bounds = (-119.0,-117.0,32.0,34.0)
   xdim = -1
   ydim = -1
   geodict = Grid2D.fixGeoDict(bounds,xdim,ydim,nrows,ncols,preserve='shape')
   grid = Grid2D(data,geodict)

Interpolating a smaller area inside that grid.

::

   subbounds = (-118.7834,-117.0123,32.8754,33.7896)
   xdim,ydim = (geodict['xdim'],geodict['ydim'])
   nrows = ncols = -1
   subgeodict = Grid2D.fixGeoDict(bounds,xdim,ydim,nrows,ncols,preserve='dims')
   interpgrid = grid.interpolateToGrid(subgeodict)
     

This class is a semi-abstract implementation of the Grid superclass - that is, it implements a number
of useful functions for creating and manipulating 2D grid objects, but no methods for reading or 
writing those grids to disk in standard file formats.  For that, see:

 * GMTGrid
 * GDALGrid
 * ShakeGrid
 * MultiHazardGrid


.. autoclass:: mapio.gmt.GMTGrid
   :members:
   :inherited-members:

mapio.gdal module
--------------------

Grid2D subclass for reading,writing, and manipulating ESRI format grids.

Usage:

Getting the geo-referencing information about a grid

::

   fgeodict = GDALGrid.getFileGeoDict(gmtfile)
   bounds = [fgeodict['xmin'],fgeodict['xmax'],fgeodict['ymin'],fgeodict['ymax']]
   print 'The file spans from %.3f to %.3f in longitude, and %.3f to %.3f in latitude.' % bounds

Loading a subset of the data contained in the file

::

   bounds[0] = bounds[0] + (bounds[1]-bounds[0]/4.0)
   bounds[1] = bounds[1] - (bounds[1]-bounds[0]/4.0)
   bounds[2] = bounds[0] + (bounds[1]-bounds[0]/4.0)
   bounds[3] = bounds[1] - (bounds[1]-bounds[0]/4.0)
   xdim,ydim = (fgeodict['xdim'],fgeodict['ydim'])
   nrows = ncols = -1
   geodict = GDALGrid.fixGeoDict(bounds,xdim,ydim,nrows,ncols,preserve='dims')
   gmtgrid = GDALGrid.load(gmtfilename,samplegeodict=geodict)
     

This class supports reading and writing of ESRI floating point and integer formats (.flt, .bil/.bip/.bsq).


.. autoclass:: mapio.gmt.GMTGrid
   :members:
   :inherited-members:


mapio.gmt module
--------------------

Grid2D subclass for reading,writing, and manipulating GMT format grids.

Usage:

Getting the geo-referencing information about a grid

::

   fgeodict = GMTGrid.getFileGeoDict(gmtfile)
   bounds = [fgeodict['xmin'],fgeodict['xmax'],fgeodict['ymin'],fgeodict['ymax']]
   print 'The file spans from %.3f to %.3f in longitude, and %.3f to %.3f in latitude.' % bounds

Loading a subset of the data contained in the file

::

   bounds[0] = bounds[0] + (bounds[1]-bounds[0]/4.0)
   bounds[1] = bounds[1] - (bounds[1]-bounds[0]/4.0)
   bounds[2] = bounds[0] + (bounds[1]-bounds[0]/4.0)
   bounds[3] = bounds[1] - (bounds[1]-bounds[0]/4.0)
   xdim,ydim = (fgeodict['xdim'],fgeodict['ydim'])
   nrows = ncols = -1
   geodict = GMTGrid.fixGeoDict(bounds,xdim,ydim,nrows,ncols,preserve='dims')
   gmtgrid = GMTGrid.load(gmtfilename,samplegeodict=geodict)
     

This class supports reading and writing of all three GMT formats: NetCDF, HDF, and the GMT "native" format.


.. autoclass:: mapio.gmt.GMTGrid
   :members:
   :inherited-members:

mapio.multihaz module
--------------------

MultiGrid subclass for reading,writing, and manipulating HDF format multi-hazard layer grids.

Usage:

Getting the geo-referencing information about a grid

::

   fgeodict = MultiHazGrid.getFileGeoDict(gmtfile)
   bounds = [fgeodict['xmin'],fgeodict['xmax'],fgeodict['ymin'],fgeodict['ymax']]
   print 'The file spans from %.3f to %.3f in longitude, and %.3f to %.3f in latitude.' % bounds

Loading a subset of the data contained in the file

::

   bounds[0] = bounds[0] + (bounds[1]-bounds[0]/4.0)
   bounds[1] = bounds[1] - (bounds[1]-bounds[0]/4.0)
   bounds[2] = bounds[0] + (bounds[1]-bounds[0]/4.0)
   bounds[3] = bounds[1] - (bounds[1]-bounds[0]/4.0)
   xdim,ydim = (fgeodict['xdim'],fgeodict['ydim'])
   nrows = ncols = -1
   geodict = GMTGrid.fixGeoDict(bounds,xdim,ydim,nrows,ncols,preserve='dims')
   gmtgrid = GMTGrid.load(gmtfilename,samplegeodict=geodict)
     

This class supports reading and writing of all three GMT formats: NetCDF, HDF, and the GMT "native" format.


.. autoclass:: mapio.gmt.GMTGrid
   :members:
   :inherited-members:
