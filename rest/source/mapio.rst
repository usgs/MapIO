mapio package
=============

mapio.grid2d module
--------------------

A partially abstract subclass of Grid to represent 2D lat/lon gridded datasets. Some basic methods
are implemented here, enough so that all functions of working with data (aside from loading and saving)
can be used with this class.  Grids are assumed to be pixel-registered - that is, grid coordinates
represent the value at the *center* of the cells.

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


.. autoclass:: mapio.grid2d.Grid2D
   :members:
   :inherited-members:

mapio.gdal module
--------------------

Grid2D subclass for reading,writing, and manipulating ESRI format
grids.

 * `Integer Format (BIL/BIP/BSQ) Description <http://webhelp.esri.com/arcgisdesktop/9.3/index.cfm?TopicName=BIL,_BIP,_and_BSQ_raster_files>`_
 * `Float Format (FLT) Description <http://resources.esri.com/help/9.3/arcgisdesktop/com/gp_toolref/conversion_tools/float_to_raster_conversion_.htm>`_

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


.. autoclass:: mapio.gdal.GDALGrid
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

   fgeodict = MultiHazardGrid.getFileGeoDict(hazfile)
   bounds = [fgeodict['xmin'],fgeodict['xmax'],fgeodict['ymin'],fgeodict['ymax']]
   print 'The file spans from %.3f to %.3f in longitude, and %.3f to %.3f in latitude.' % bounds

Loading all data contained in the file (subsetting not supported at
this time), and retrieving various metadata fields:

::
   
   hazgrid = MultiHazardGrid.load(hazfile)
   origin = hazgrid.getOrigin()
   header = hazgrid.getHeader()
   metadata = hazgrid.getMetadata()
   geodict = hazgrid.getGeoDict()

Creating a multi-hazard grid and saving to a file

::
   
   lsprob = np.random.rand(100,100)
   lqprob = np.random.rand(100,100)
   bounds = (-118.7834,-117.0123,32.8754,33.7896)
   xdim,ydim = (0.02,0.02)
   nrows,ncols = (-1,-1)
   geodict =
   MultiHazardGrid.fixGeoDict(bounds,xdim,ydim,nrows,ncols,preserve='dims')
   layers = OrderedDict()
   layers['landslide'] = lsprob
   layers['liquefaction'] = lqprob
   origin = {'id':'us2015abcd',
             'source':'us',
             'time':datetime(2015,11,15,1,1,1),
             'lat':32.1654,
             'lon':-118.6543,
             'depth':23.1,
             'magnitude':5.8}
   header = {'type':'gfe',
             'version':1,
             'process_time':datetime(2015,11,15,1,1,1),
             'code_version':'1.0',
             'originator':'us',
             'product_id':'us2015abcd',
             'map_status':'RELEASED',
             'event_type':'ACTUAL'}
   metadata = {'creator':{'name':'Jane Smith','position':'Secondary Hazard Technician'}}
   hazgrid = MultiHazardGrid(layers,geodict,origin,header,metadata)
     

This class supports reading and writing of all three GMT formats: NetCDF, HDF, and the GMT "native" format.


.. autoclass:: mapio.multihaz.MultiHazardGrid
   :members:
   :inherited-members:

mapio.shake module
--------------------

MultiGrid subclass for reading,writing, and manipulating XML format multi-hazard ShakeMap grids.

Usage:

Getting the geo-referencing information about a grid

::

   fgeodict = ShakeGrid.getFileGeoDict(hazfile)
   bounds = [fgeodict['xmin'],fgeodict['xmax'],fgeodict['ymin'],fgeodict['ymax']]
   print 'The file spans from %.3f to %.3f in longitude, and %.3f to %.3f in latitude.' % bounds

Loading all data contained in the file (subsetting not supported at
this time), and retrieving various metadata fields:

::
   
   shakegrid = ShakeGrid.load(hazfile)
   geodict = shakegrid.getGeoDict()
   pgagrid = shakegrid.getLayer('pga')

Creating a ShakeGrid and saving to a file

::
   
   pga = np.random.rand(100,100)
   pgv = np.random.rand(100,100)
   mmi = np.random.rand(100,100)
   bounds = (-118.7834,-117.0123,32.8754,33.7896)
   xdim,ydim = (0.02,0.02)
   nrows,ncols = (-1,-1)
   geodict = ShakeGrid.fixGeoDict(bounds,xdim,ydim,nrows,ncols,preserve='dims')
   layers = OrderedDict()
   layers['pga'] = pga
   layers['pgv'] = pgv
   layers['mmi'] = mmi
   eventdict = {'event_id':'us2015abcd',
                'event_network':'us',
                'event_timestamp':datetime(2015,11,15,1,1,1),
                'lat':32.1654,
                'lon':-118.6543,
                'depth':23.1,
                'magnitude':5.8}
   shakedict = {'event_id':'us2015abcd',
                'shakemap_id':'us2015abcd',
                'shakemap_version':1,
                'process_timestamp':datetime(2015,11,15,1,1,1),
                'code_version':'1.0',
                'shakemap_originator':'us',
                'map_status':'RELEASED',
                'shakemap_event_type':'ACTUAL'}
   uncertainty = {'pga':(0.0,0),
                  'mmi':(0.0,0),
                  'pgv':(0.0,0)}
   shakegrid = ShakeGrid(layers,geodict,eventdict,shakedict,uncertainty)
     

This class supports reading and writing of ShakeMaps in XML format.


.. autoclass:: mapio.shake.ShakeGrid
   :members:
   :inherited-members:
