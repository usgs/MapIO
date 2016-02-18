#!/usr/bin/env python

#python 3 compatibility
from __future__ import print_function

#stdlib imports
from xml.dom import minidom
from datetime import datetime
from collections import OrderedDict
import re
import sys
if sys.version_info.major == 2:
    import StringIO
else:
    from io import StringIO
import os.path

#third party
from .gridbase import Grid
from .multiple import MultiGrid
from .dataset import DataSetException
from .grid2d import Grid2D
from .geodict import GeoDict
import numpy as np



GRIDKEYS = {'event_id':'string',
            'shakemap_id':'string',
            'shakemap_version':'int',
            'code_version':'string',
            'process_timestamp':'datetime',
            'shakemap_originator':'string',
            'map_status':'string',
            'shakemap_event_type':'string'}

EVENTKEYS = {'event_id':'string',
             'magnitude':'float',
             'depth':'float',
             'lat':'float',
             'lon':'float',
             'event_timestamp':'datetime',
             'event_network':'string',
             'event_description':'string'}

SPECKEYS = {'lon_min':'float',
            'lon_max':'float',
            'lat_min':'float',
            'lat_max':'float',
            'nominal_lon_spacing':'float',
            'nominal_lat_spacing':'float',
            'nlon':'int',
            'nlat':'int'}

FIELDKEYS = OrderedDict()
FIELDKEYS['pga'] = ('pctg','%.2f')
FIELDKEYS['pgv'] = ('cms','%.2f')
FIELDKEYS['mmi'] = ('intensity','%.2f')
FIELDKEYS['psa03'] = ('pctg','%.2f')
FIELDKEYS['psa10'] = ('pctg','%.2f')
FIELDKEYS['psa30'] = ('pctg','%.2f')
FIELDKEYS['stdpga'] = ('ln(pctg)','%.2f')
FIELDKEYS['urat'] = ('','%.2f')
FIELDKEYS['svel'] = ('ms','%.2f')
          

TIMEFMT = '%Y-%m-%dT%H:%M:%S'

def _readElement(element,keys):
    """Convenience function for reading all attributes of a ShakeMap gridfile element, doing data conversion.
    :param element:
      XML DOM element with attributes
    :param keys:
      Dictionary of keys for different elements with types.
    :returns:
      Dictionary of named attributes with values from DOM element, converted to int, float or datetime where needed.
    """
    eldict = OrderedDict()
    for (key,dtype) in keys.items():
        if dtype == 'datetime':
            eldict[key] = datetime.strptime(element.getAttribute(key)[0:19],TIMEFMT)
        elif dtype == 'int':
            eldict[key] = int(element.getAttribute(key))
        elif dtype == 'float':
            eldict[key] = float(element.getAttribute(key))
        else:
            eldict[key] = element.getAttribute(key)
    return eldict

def _getXMLText(fileobj):
    """Convenience function for reading the XML header data in a ShakeMap grid file.
    :param fileobj:
      File-like object representing an open ShakeMap grid file.
    :returns:
      All XML header text.
    """
    tline = fileobj.readline() 
    datamatch = re.compile('grid_data')
    xmltext = ''
    tlineold = ''
    while not datamatch.search(tline) and tline != tlineold:
        tlineold = tline
        xmltext = xmltext+tline
        tline = fileobj.readline()    

    xmltext = xmltext+'</shakemap_grid>'
    return xmltext

def getHeaderData(shakefile):
    """Return all relevant header data from ShakeMap grid.xml file.
    :param fileobj:
      File-like object representing an open ShakeMap grid file.
    :returns:
      Tuple of dictionaries:
        - Dictionary representing the grid element in the ShakeMap header.
        - Dictionary representing the event element in the ShakeMap header.
        - Dictionary representing the grid_specification element in the ShakeMap header.
        - Dictionary representing the list of grid_field elements in the ShakeMap header.
        - Dictionary representing the list of event_specific_uncertainty elements in the ShakeMap header.
    """
    f = open(shakefile,'rt')
    griddict,eventdict,specdict,fields,uncertainties = _getHeaderData(f)
    f.close()
    return (griddict,eventdict,specdict,fields,uncertainties)

def _getHeaderData(fileobj):
    """Return all relevant header data from ShakeMap grid.xml file.
    :param fileobj:
      File-like object representing an open ShakeMap grid file.
    :returns:
      Tuple of dictionaries:
        - Dictionary representing the grid element in the ShakeMap header.
        - Dictionary representing the event element in the ShakeMap header.
        - Dictionary representing the grid_specification element in the ShakeMap header.
        - Dictionary representing the list of grid_field elements in the ShakeMap header.
        - Dictionary representing the list of event_specific_uncertainty elements in the ShakeMap header.
    """
    xmltext = _getXMLText(fileobj)
    root = minidom.parseString(xmltext)
    griddict = OrderedDict()
    gridel = root.getElementsByTagName('shakemap_grid')[0]
    griddict = _readElement(gridel,GRIDKEYS)
    eventel = root.getElementsByTagName('event')[0]
    eventdict = _readElement(eventel,EVENTKEYS)
    specel = root.getElementsByTagName('grid_specification')[0]
    specdict = _readElement(specel,SPECKEYS)
    field_elements = root.getElementsByTagName('grid_field')
    fields = []
    for fieldel in field_elements:
        att = fieldel.getAttribute('name').lower()
        if att in ['lon','lat']:
            continue
        fields.append(att)

    uncertainties = OrderedDict()
    unc_elements = root.getElementsByTagName('event_specific_uncertainty')
    for uncel in unc_elements:
        key = uncel.getAttribute('name')
        value = float(uncel.getAttribute('value'))
        try:
            numsta = int(uncel.getAttribute('numsta'))
        except:
            numsta = 0
        uncertainties[key] = (value,numsta)

    return (griddict,eventdict,specdict,fields,uncertainties)

def readShakeFile(fileobj,adjust=None):
    """Reads in the data and metadata for a ShakeMap object (can be passed to ShakeGrid constructor).
    :param fileobj:
      File-like object representing an open ShakeMap grid file.
    :param adjust:
      String (one of None,'bounds','res') - adjust some of the ShakeMap parameters as necessary (usually "res").
        None: All input parameters are assumed to be self-consistent, an exception will be raised if they are not.
        'bounds': dx/dy, nx/ny, xmin/ymax are assumed to be correct, xmax/ymin will be recalculated.
        'res': nx/ny, xmin/ymax, xmax/ymin and assumed to be correct, dx/dy will be recalculated.
    :returns:
      Tuple containing:
        - Ordered Dictionary with the data layers in ShakeMap (MMI, PGA, PGV, etc.)
        - Geo dictionary describing the spatial extent and resolution of all the layers.
        - Dictionary representing the event element in the ShakeMap header.
        - Dictionary representing the grid element in the ShakeMap header.
        - Dictionary representing the list of event_specific_uncertainty elements in the ShakeMap header.
    """
    griddict,eventdict,specdict,fields,uncertainties = _getHeaderData(fileobj)
    nx = specdict['nlon']
    ny = specdict['nlat']
    layers = OrderedDict()

    #use the numpy loadtxt function to read in the actual data
    #we're cheating by telling numpy.loadtxt that the last two lines of the XML file are comments
    data = np.loadtxt(fileobj,comments='<').astype('float32')
    data = data[:,2:] #throw away lat/lon columns
    for i in range(0,len(fields)):
        field = fields[i]
        layers[field] = data[:,i].reshape(ny,nx)

    #create the geodict from the grid_spec element
    geodict = GeoDict({'xmin':specdict['lon_min'],
                       'xmax':specdict['lon_max'],
                       'ymin':specdict['lat_min'],
                       'ymax':specdict['lat_max'],
                       'dx':specdict['nominal_lon_spacing'],
                       'dy':specdict['nominal_lat_spacing'],
                       'ny':specdict['nlat'],
                       'nx':specdict['nlon']},adjust=adjust)
    
    return (layers,geodict,eventdict,griddict,uncertainties)

class ShakeGrid(MultiGrid):
    """
    A class that implements a MultiGrid object around ShakeMap grid.xml data sets.
    """
    def __init__(self,layers,geodict,eventDict,shakeDict,uncertaintyDict):
        """Construct a ShakeGrid object.
        :param layers:
           OrderedDict containing ShakeMap data layers (keys are 'pga', etc., values are 2D arrays of data).
        :param geodict:
           Dictionary specifying the spatial extent,resolution and shape of the data.
        :param eventDict:
          Dictionary with elements:
            - event_id String of event ID (i.e., 'us2015abcd')
            - magnitude Float event magnitude
            - depth Float event depth
            - lat Float event latitude
            - lon Float event longitude
            - event_timestamp Datetime object representing event origin time.
            - event_network Event originating network (i.e., 'us')
        :param shakeDict:
          Dictionary with elements:
            - event_id String of event ID (i.e., 'us2015abcd')
            - shakemap_id String of ShakeMap ID (not necessarily the same as the event ID)
            - shakemap_version Integer ShakeMap version number (i.e., 1)
            - code_version String version of ShakeMap code that created this file (i.e.,'4.0')
            - process_timestamp Datetime of when ShakeMap data was created.
            - shakemap_originator String representing network that created the ShakeMap
            - map_status String, one of RELEASED, ??
            - shakemap_event_type String, one of ['ACTUAL','SCENARIO']
        :param uncertaintyDict:
          Dictionary with elements that have keys matching the layers keys, and values that are
           a tuple of that layer's uncertainty (float) and the number of stations used to determine that uncertainty (int).
        :returns:
           A ShakeGrid object.
        """
        self._layers = OrderedDict()
        self._geodict = geodict
        for (layerkey,layerdata) in layers.items():
            self._layers[layerkey] = Grid2D(data=layerdata,geodict=geodict)
        self._setEventDict(eventDict)
        self._setShakeDict(shakeDict)
        self._setUncertaintyDict(uncertaintyDict)

    @classmethod
    def getFileGeoDict(cls,shakefilename,adjust=None):
        """Get the spatial extent, resolution, and shape of grids inside ShakeMap grid file.
        :param filename:
           File name of ShakeMap grid file.
        :param adjust:
          String (one of None,'bounds','res') - adjust some of the ShakeMap parameters as necessary (usually "res").
            None: All input parameters are assumed to be self-consistent, an exception will be raised if they are not.
            'bounds': dx/dy, nx/ny, xmin/ymax are assumed to be correct, xmax/ymin will be recalculated.
            'res': nx/ny, xmin/ymax, xmax/ymin and assumed to be correct, dx/dy will be recalculated.
        :returns:
          GeoDict specifying spatial extent, resolution, and shape of grids inside ShakeMap grid file.
        """
        isFileObj = False
        if not hasattr(shakefilename,'read'):
            shakefile = open(shakefilename,'r')
        else:
            isFileObj = True
            shakefile = shakefilename
        griddict,eventdict,specdict,fields,uncertainties = _getHeaderData(shakefile)
        if isFileObj:
            shakefile.close()
        geodict = GeoDict({'xmin':specdict['lon_min'],
                   'xmax':specdict['lon_max'],
                   'ymin':specdict['lat_min'],
                   'ymax':specdict['lat_max'],
                   'dx':specdict['nominal_lon_spacing'],
                   'dy':specdict['nominal_lat_spacing'],
                   'ny':specdict['nlat'],
                   'nx':specdict['nlon']},adjust=adjust)
        return geodict

    @classmethod
    def load(cls,shakefilename,samplegeodict=None,resample=False,method='linear',doPadding=False,padValue=np.nan,adjust=None):
        """Create a ShakeGrid object from a ShakeMap grid.xml file.
        :param shakefilename:
          File name or File-like object of ShakeMap grid.xml file.
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
        :param adjust:
          String (one of None,'bounds','res') - adjust some of the ShakeMap parameters as necessary (usually "res").
            None: All input parameters are assumed to be self-consistent, an exception will be raised if they are not.
            'bounds': dx/dy, nx/ny, xmin/ymax are assumed to be correct, xmax/ymin will be recalculated.
            'res': nx/ny, xmin/ymax, xmax/ymin and assumed to be correct, dx/dy will be recalculated.
        :returns:
          ShakeGrid object.
        """
        #geodict can have dx/dy OR nx/ny.  If given both, dx/dy will be used to re-calculate ny/nx
        isFileObj = False
        if not hasattr(shakefilename,'read'):
            shakefile = open(shakefilename,'r')
        else:
            isFileObj = True
            shakefile = shakefilename

        if samplegeodict is not None:
            #fill in ny/nx or dx/dy, whichever is not specified.  dx/dy dictate if both pairs are specified.
            bounds = (samplegeodict.xmin,samplegeodict.xmax,samplegeodict.ymin,samplegeodict.ymax)
            dx,dy = samplegeodict.dx,samplegeodict.dy
            ny,nx = samplegeodict.ny,samplegeodict.nx


        #read the file using the available function
        layers,fgeodict,eventDict,shakeDict,uncertaintyDict = readShakeFile(shakefile,adjust=adjust)
            
        if not isFileObj:
            shakefile.close()

        if samplegeodict is None:
            geodict = fgeodict
        else:
            bounds = (samplegeodict.xmin,samplegeodict.xmax,samplegeodict.ymin,samplegeodict.ymax)
            isOutside = False
            xmin = fgeodict.xmin
            xmax = fgeodict.xmax
            ymin = fgeodict.ymin
            ymax = fgeodict.ymax
            if bounds[0] < xmin or bounds[1] > xmax or bounds[2] < ymin or bounds[3] > ymax:
                isOutside = True
            if isOutside and resample and not doPadding:
                raise DataSetException('Cannot resample data given input bounds, unless doPadding is set to True.')

            if doPadding:
                leftpad,rightpad,bottompad,toppad,geodict = super(MultiGrid,cls)._getPadding(fgeodict,samplegeodict,padValue)
                for (layername,layerdata) in layers.items():
                    #pad left side
                    layerdata = np.hstack((leftpad,layerdata))
                    #pad right side
                    layerdata = np.hstack((layerdata,rightpad))
                    #pad bottom
                    layerdata = np.vstack((layerdata,bottompad))
                    #pad top
                    layerdata = np.vstack((toppad,layerdata))
                    grid = Grid2D(layerdata,geodict)
                    if resample: #should I just do an interpolateToGrid() here?
                        grid.interpolateToGrid(samplegeodict,method=method)
                    layers[layername] = grid.getData()
                geodict = grid.getGeoDict().copy()
            else:
                tgeodict = fgeodict.getIntersection(samplegeodict)
                geodict = fgeodict.getBoundsWithin(tgeodict)
                for (layername,layerdata) in layers.items():
                    newgrid = Grid2D(layerdata,fgeodict)
                    if resample:
                        newgrid.interpolateToGrid(samplegeodict,method=method)
                    else:
                        newgrid = newgrid.cut(geodict.xmin,geodict.xmax,geodict.ymin,geodict.ymax)
                    layers[layername] = newgrid.getData()
                if resample:
                    geodict = samplegeodict
            
        return cls(layers,geodict,eventDict,shakeDict,uncertaintyDict)
    
    def save(self,filename,version=1):
        """Save a ShakeGrid object to the grid.xml format.
        :param filename:
          File name or file-like object.
        :param version:
          Integer Shakemap version number.
        """

        #handle differences btw python2 and python3
        isThree = True
        if sys.version_info.major == 2:
            isThree = False
        
        isFile = False
        if not hasattr(filename,'read'):
            isFile = True
            f = open(filename,'wb')
        else:
            f = filename    
        SCHEMA1 = 'http://www.w3.org/2001/XMLSchema-instance'
        SCHEMA2 = 'http://earthquake.usgs.gov/eqcenter/shakemap'
        SCHEMA3 = 'http://earthquake.usgs.gov http://earthquake.usgs.gov/eqcenter/shakemap/xml/schemas/shakemap.xsd'

        f.write(b'<?xml version="1.0" encoding="US-ASCII" standalone="yes"?>')
        fmt = '<shakemap_grid xmlns:xsi="%s" xmlns="%s" xsi:schemaLocation="%s" event_id="%s" shakemap_id="%s" shakemap_version="%i" code_version="%s" process_timestamp="%s" shakemap_originator="%s" map_status="%s" shakemap_event_type="%s">\n'
        tpl = (SCHEMA1,SCHEMA2,SCHEMA3,
               self._shakeDict['event_id'],self._shakeDict['shakemap_id'],self._shakeDict['shakemap_version'],
               self._shakeDict['code_version'],datetime.utcnow().strftime(TIMEFMT),
               self._shakeDict['shakemap_originator'],self._shakeDict['map_status'],self._shakeDict['shakemap_event_type'])
        if isThree:
            f.write(bytes(fmt % tpl,'ascii'))
        else:
            f.write(fmt % tpl)
        fmt = '<event event_id="%s" magnitude="%.1f" depth="%.1f" lat="%.4f" lon="%.4f" event_timestamp="%s" event_network="%s" event_description="%s"/>\n'
        tpl = (self._eventDict['event_id'],self._eventDict['magnitude'],self._eventDict['depth'],
               self._eventDict['lat'],self._eventDict['lon'],self._eventDict['event_timestamp'].strftime(TIMEFMT),
               self._eventDict['event_network'],self._eventDict['event_description'])
        if isThree:
            f.write(bytes(fmt % tpl,'ascii'))
        else:
            f.write(fmt % tpl)
        fmt = '<grid_specification lon_min="%.4f" lat_min="%.4f" lon_max="%.4f" lat_max="%.4f" nominal_lon_spacing="%.4f" nominal_lat_spacing="%.4f" nlon="%i" nlat="%i"/>'
        tpl = (self._geodict.xmin,self._geodict.ymin,self._geodict.xmax,self._geodict.ymax,
               self._geodict.dx,self._geodict.dy,self._geodict.nx,self._geodict.ny)
        if isThree:
            f.write(bytes(fmt % tpl,'ascii'))
        else:
            f.write(fmt % tpl)
        fmt = '<event_specific_uncertainty name="%s" value="%.4f" numsta="%i" />\n'
        for (key,unctuple) in self._uncertaintyDict.items():
            value,numsta = unctuple
            tpl = (key,value,numsta)
            if isThree:
                f.write(bytes(fmt % tpl,'ascii'))
            else:
                f.write(fmt % tpl)
        f.write(b'<grid_field index="1" name="LON" units="dd" />\n')
        f.write(b'<grid_field index="2" name="LAT" units="dd" />\n')
        idx = 3
        fmt = '<grid_field index="%i" name="%s" units="%s" />\n'
        data_formats = ['%.4f','%.4f']
        for field in self._layers.keys():
            tpl = (idx,field.upper(),FIELDKEYS[field][0])
            data_formats.append(FIELDKEYS[field][1])
            if isThree:
                db = bytes(fmt % tpl,'ascii')
            else:
                db = fmt % tpl
            f.write(db)
            idx += 1
        f.write(b'<grid_data>\n')
        lat,lon = Grid().getLatLonMesh(self._geodict)
        nfields = 2 + len(self._layers)
        data = np.zeros((self._geodict.ny*self._geodict.nx,nfields))
        data[:,0] = lat.flatten()
        data[:,1] = lon.flatten()
        fidx = 2
        for grid in self._layers.values():
            data[:,fidx] = grid.getData().flatten()
            fidx += 1
        np.savetxt(f,data,delimiter=' ',fmt=data_formats)
        f.write(b'</grid_data>\n')
        if isFile:
            f.close()

    def _checkType(self,key,dtype):
        """Internal method used to validate the types of the input dictionaries used in constructor.
        :param key:
          String key value
        :param dtype:
          Expected data type of key.
        :returns:
          True if key matches expected dtype, False if not.
        """
        if dtype == 'string' and (not isinstance(key,str) and not isinstance(key,unicode)):
            return False
        if dtype == 'int' and not isinstance(key,int):
            return False
        if dtype == 'float' and not isinstance(key,float):
            return False
        if dtype == 'datetime' and not isinstance(key,datetime):
            return False
        return True
    
    def _setEventDict(self,eventdict):
        """Set the event dictionary, validating all values in the dictionary.
        :param eventdict:
          Event dictionary (see constructor).
        :raises DataSetException:
          When one of the values in the dictionary does not match its expected type.        
        """
        for (key,dtype) in EVENTKEYS.items():
            if key not in eventdict:
                raise DataSetException('eventdict is missing key "%s"' % key)
            if not self._checkType(eventdict[key],dtype):
                raise DataSetException('eventdict key value "%s" is the wrong datatype' % str(eventdict[key]))
        self._eventDict = eventdict.copy()

    def getEventDict(self):
        """Get the event dictionary (the attributes of the "event" element in the ShakeMap header).
        :returns:
          Dictionary containing the following fields:
           - event_id: String like "us2016abcd".
           - magnitude: Earthquake magnitude.
           - lat: Earthquake latitude.
           - lon: Earthquake longitude.
           - depth: Earthquake depth.
           - event_timestamp: Earthquake origin time.
           - event_network: Network of earthquake origin.
           - event_description: Description of earthquake location.
        """
        return self._eventDict

    def _setShakeDict(self,shakedict):
        """Set the shake dictionary, validating all values in the dictionary.
        :param shakedict:
          Shake dictionary (see constructor).
        :raises DataSetException:
          When one of the values in the dictionary does not match its expected type.        
        """
        for (key,dtype) in GRIDKEYS.items():
            if key not in shakedict:
                raise DataSetException('shakedict is missing key "%s"' % key)
            if not self._checkType(shakedict[key],dtype):
                raise DataSetException('shakedict key value "%s" is the wrong datatype' % str(shakedict[key]))
        self._shakeDict = shakedict.copy()

    def getShakeDict(self):
        """Get the shake dictionary (the attributes of the "shakemap_grid" element in the ShakeMap header).
        :returns:
          Dictionary containing the following fields:
           - event_id: String like "us2016abcd".
           - shakemap_id: String like "us2016abcd".
           - shakemap_version: Version of the map that has been created.
           - code_version: Version of the ShakeMap software that was used to create the map.
           - shakemap_originator: Network that created the ShakeMap.
           - map_status: One of 'RELEASED' or 'REVIEWED'.
           - shakemap_event_type: One of 'ACTUAL' or 'SCENARIO'.
        """
        return self._shakeDict

    def _setUncertaintyDict(self,uncertaintyDict):
        """Set the uncertainty dictionary.
        :param uncertaintyDict:
          Uncertainty dictionary (see constructor).
        """
        self._uncertaintyDict = uncertaintyDict.copy()
        

    
