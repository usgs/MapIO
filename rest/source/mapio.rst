mapio package
=============

mapio.geodict module
--------------------

A class to represent the spatial information about a Grid (see classes
below).  The required data about a grid that a GeoDict object contains
are:
  - xmin,xmax,ymin,ymax The bounds of the grid (where the
    left,right,top, and bottom grid lines are.)
  - dx,dy The resolution of the grid in the X and Y dimensions (width
    and height of cells).
  - nx,ny The number of columns and rows in the grid, respectively.

A GeoDict object can be created outside of the context of any of the
Grid classes (see below), and it has a number of useful methods.  Each
of the Grid classes contains a GeoDict object.

Usage:

Creating a GeoDict object - three different methods.

::
   from mapio.geodict import GeoDict

   ny = 100
   nx = 100
   data = np.random.rand(100,100)
   xmin,xmax,ymin,ymax = (-119.0,-117.0,32.0,34.0)
   dx = 0.1
   dy = 0.1
   #Using constructor
   geodict1 = GeoDict({'xmin':xmin,'xmax':xmax,
                      'ymin':ymin,'ymax':ymax,
                      'dx':dx,'dy':dy,
                      'nx':nx,'ny':ny},adjust='res')
   #Using box class method
   geodict2 = GeoDict.createDictFromBox(xmin,xmax,ymin,ymax,dx,dy)
   cx = xmin + (xmax-xmin)/2.0
   cy = ymin + (ymax-ymin)/2.0
   xspan = 2.0
   yspan = 2.0
   #Using center class method
   geodict2 = GeoDict.createDictFromCenter(cx,cy,dx,dy,xspan,yspan)

Other useful methods.

::
   geodict3 = geodict2.copy() #Make a complete copy of the input GeoDict
   assert geodict3 == geodict2 #Compare two GeoDict objects
   gd3dict = geodict3.asDict() #Return a dictionary of a GeoDict's
   lat,lon = geodict2.getLatLon(50,50)
   row,col = geodict2.getRowCol(lat,lon)
   xmin = geodict2.xmin #Retrieving xmin property of GeoDict
   xmax = geodict2.xmax #Retrieving xmin property of GeoDict, etc.
   

Interpolating a smaller area inside that grid.

::
   txmin,txmax,tymin,tymax = (-118.7834,-117.0123,32.8754,33.7896)
   dx,dy = (geodict.xdim,geodict.ydim)
   nrows = 50
   ncols = 50
   subgeodict = GeoDict({'xmin':txmin,'xmax':txmax,
                         'ymin':tymin,'ymax':tymax,
                         'dx':dx,'dy':dy,
                         'nrows,ncols,adjust='res')
   interpgrid = grid.interpolateToGrid(subgeodict)
     

mapio.grid2d module
--------------------

A partially abstract subclass of Grid to represent 2D lat/lon gridded datasets. Some basic methods
are implemented here, enough so that all functions of working with data (aside from loading and saving)
can be used with this class.  Grids are assumed to be pixel-registered - that is, grid coordinates
represent the value at the *center* of the cells.

Usage:

Creating a Grid2D object.

::
   from mapio.grid2d import Grid2D
   from mapio.geodict import GeoDict

   ny = 100
   nx = 100
   data = np.random.rand(100,100)
   xmin,xmax,ymin,ymax = (-119.0,-117.0,32.0,34.0)
   dx = -1
   dy = -1
   geodict = GeoDict({'xmin':xmin,'xmax':xmax,
                      'ymin':ymin,'ymax':ymax,
                      'dx':dx,'dy':dy,
                      'nx':nx,'ny':ny},adjust='res')
   grid = Grid2D(data,geodict)

Interpolating a smaller area inside that grid.

::
   txmin,txmax,tymin,tymax = (-118.7834,-117.0123,32.8754,33.7896)
   dx,dy = (geodict.xdim,geodict.ydim)
   nrows = 50
   ncols = 50
   subgeodict = GeoDict({'xmin':txmin,'xmax':txmax,
                         'ymin':tymin,'ymax':tymax,
                         'dx':dx,'dy':dy,
                         'nrows,ncols,adjust='res')
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
   from mapio.gdal import GDALGrid
   fgeodict = GDALGrid.getFileGeoDict(gmtfile)
   print fgeodict

Loading a subset of the data contained in the file

::

   xmin = fgeodict.xmin + (fgeodict.xmax-fgeodict.xmin/4.0)
   xmax = xmin + (fgeodict.xmax-fgeodict.xmin/4.0)
   ymin = fgeodict.ymin + (fgeodict.ymax-fgeodict.ymin/4.0)
   ymax = ymin + (fgeodict.ymax-fgeodict.ymin/4.0)
   dx,dy = (fgeodict.nx,fgeodict.dy)
   dy = dx = -1
   geodict = GeoDict.createDictFromBox(xmin,xmax,ymin,ymax,dx,dy)
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
   print fgeodict

Loading a subset of the data contained in the file

::

   xmin = fgeodict.xmin + (fgeodict.xmax-fgeodict.xmin/4.0)
   xmax = xmin + (fgeodict.xmax-fgeodict.xmin/4.0)
   ymin = fgeodict.ymin + (fgeodict.ymax-fgeodict.ymin/4.0)
   ymax = ymin + (fgeodict.ymax-fgeodict.ymin/4.0)
   dx,dy = (fgeodict.nx,fgeodict.dy)
   dy = dx = -1
   geodict = GeoDict.createDictFromBox(xmin,xmax,ymin,ymax,dx,dy)
   gmtgrid = GDALGrid.load(gmtfilename,samplegeodict=geodict)
     

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
   from mapio.multihaz import MultiHazardGrid
   from mapio.geodict import GeoDict

   fgeodict = MultiHazardGrid.getFileGeoDict(hazfile)
   print fgeodict

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
   ny = 100
   nx = 100
   lsprob = np.random.rand(ny,nx)
   lqprob = np.random.rand(ny,nx)

   xmin,xmax,ymin,ymax = (-119.0,-117.0,32.0,34.0)
   dx = -1
   dy = -1
   geodict = GeoDict({'xmin':xmin,'xmax':xmax,
                      'ymin':ymin,'ymax':ymax,
                      'dx':dx,'dy':dy,
                      'nx':nx,'ny':ny},adjust='res')
   
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
   metadata = {'creator':{'name':'Jane Smith','position':'Ground Failure Technician'}}
   hazgrid = MultiHazardGrid(layers,geodict,origin,header,metadata)

.. autoclass:: mapio.multihaz.MultiHazardGrid
   :members:
   :inherited-members:

mapio.shake module
--------------------

MultiGrid subclass for reading,writing, and manipulating XML format multi-hazard ShakeMap grids.

Usage:

Getting the geo-referencing information about a grid

::
   from mapio.shake import ShakeGrid
   from mapio.geodict import GeoDict

   fgeodict = ShakeGrid.getFileGeoDict(hazfile)
   print fgeodict

Loading all data contained in the file (subsetting not supported at
this time), and retrieving various metadata fields:

::
   
   shakegrid = ShakeGrid.load(hazfile)
   geodict = shakegrid.getGeoDict()
   pgagrid = shakegrid.getLayer('pga') #a Grid2D object

Creating a ShakeGrid and saving to a file

::
   
   pga = np.random.rand(100,100)
   pgv = np.random.rand(100,100)
   mmi = np.random.rand(100,100)
   xmin,xmax,ymin,ymax = (-119.0,-117.0,32.0,34.0)
   dx = -1
   dy = -1
   geodict = GeoDict({'xmin':xmin,'xmax':xmax,
                      'ymin':ymin,'ymax':ymax,
                      'dx':dx,'dy':dy,
                      'nx':nx,'ny':ny},adjust='res')
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
