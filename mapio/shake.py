#!/usr/bin/env python

# python 3 compatibility
from __future__ import print_function

# stdlib imports
from xml.dom import minidom
from datetime import datetime
from collections import OrderedDict
import re
import sys

from xml.sax import saxutils

# third party
from .gridbase import Grid
from .multiple import MultiGrid
from .dataset import DataSetException
from .grid2d import Grid2D
from .geodict import GeoDict
import numpy as np
import pandas as pd


GRIDKEYS = {
    "event_id": "string",
    "shakemap_id": "string",
    "shakemap_version": "int",
    "code_version": "string",
    "process_timestamp": "datetime",
    "shakemap_originator": "string",
    "map_status": "string",
    "shakemap_event_type": "string",
}

EVENTKEYS = {
    "event_id": "string",
    "magnitude": "float",
    "depth": "float",
    "lat": "float",
    "lon": "float",
    "event_timestamp": "datetime",
    "event_network": "string",
    "event_description": "string",
}

SPECKEYS = {
    "lon_min": "float",
    "lon_max": "float",
    "lat_min": "float",
    "lat_max": "float",
    "nominal_lon_spacing": "float",
    "nominal_lat_spacing": "float",
    "nlon": "int",
    "nlat": "int",
}

FIELDKEYS = OrderedDict()
FIELDKEYS["lat"] = ("dd", "%.4f")
FIELDKEYS["lon"] = ("dd", "%.4f")


TIMEFMT = "%Y-%m-%dT%H:%M:%S"


def _readElement(element, keys):
    """Convenience function for reading all attributes of a ShakeMap gridfile element,
    doing data conversion.

    Args:
        element (XML DOM):
            XML DOM element with attributes
        keys (dict):
            Dictionary of keys for different elements with types.
    Returns:
        Dictionary of named attributes with values from DOM element, converted to int,
        float or datetime where needed.
    """
    eldict = OrderedDict()
    for (key, dtype) in keys.items():
        if dtype == "datetime":
            eldict[key] = datetime.strptime(element.getAttribute(key)[0:19], TIMEFMT)
        elif dtype == "int":
            try:
                eldict[key] = int(element.getAttribute(key))
            except ValueError:
                eldict[key] = int(float(element.getAttribute(key)))
        elif dtype == "float":
            eldict[key] = float(element.getAttribute(key))
        else:
            eldict[key] = element.getAttribute(key)
    return eldict


def _getXMLText(fileobj):
    """Convenience function for reading the XML header data in a ShakeMap grid file.

    Args:
        fileobj (IO):
            File-like object representing an open ShakeMap grid file.
    Returns:
        All XML header text.
    """
    tline = fileobj.readline()
    datamatch = re.compile("grid_data")
    xmltext = ""
    tlineold = ""
    while not datamatch.search(tline) and tline != tlineold:
        tlineold = tline
        xmltext = xmltext + tline
        tline = fileobj.readline()

    xmltext = xmltext + "</shakemap_grid>"
    return xmltext


def getHeaderData(shakefile):
    """Return all relevant header data from ShakeMap grid.xml file.

    Args
        shakefile (str):
            File-like object representing an open ShakeMap grid file.
    Returns:
        Tuple of dictionaries:
            - Dictionary representing the grid element in the ShakeMap header.
            - Dictionary representing the event element in the ShakeMap header.
            - Dictionary representing the grid_specification element in the ShakeMap
            header.
            - Dictionary representing the list of grid_field elements in the ShakeMap
            header.
            - Dictionary representing the list of event_specific_uncertainty elements in
            the ShakeMap header.
    """
    f = open(shakefile, "rt")
    griddict, eventdict, specdict, fields, uncertainties = _getHeaderData(f)
    f.close()
    return (griddict, eventdict, specdict, fields, uncertainties)


def _getHeaderData(fileobj):
    """Return all relevant header data from ShakeMap grid.xml file.

    Args:
        fileobj (IO):
            File-like object representing an open ShakeMap grid file.
    Returns:
        Tuple of dictionaries:
            - Dictionary representing the grid element in the ShakeMap header.
            - Dictionary representing the event element in the ShakeMap header.
            - Dictionary representing the grid_specification element in the ShakeMap
            header.
            - Dictionary representing the list of grid_field elements in the ShakeMap
            header.
            - Dictionary representing the list of event_specific_uncertainty elements in
            the ShakeMap header.
    """
    xmltext = _getXMLText(fileobj)
    root = minidom.parseString(xmltext)
    griddict = OrderedDict()
    gridel = root.getElementsByTagName("shakemap_grid")[0]
    griddict = _readElement(gridel, GRIDKEYS)
    eventel = root.getElementsByTagName("event")[0]
    eventdict = _readElement(eventel, EVENTKEYS)
    # un-xmlify the location string (convert &amp; to &)
    eventdict["event_description"] = saxutils.unescape(eventdict["event_description"])
    specel = root.getElementsByTagName("grid_specification")[0]
    specdict = _readElement(specel, SPECKEYS)
    field_elements = root.getElementsByTagName("grid_field")
    fields = []
    for fieldel in field_elements:
        att = fieldel.getAttribute("name").lower()
        if att in ["lon", "lat"]:
            continue
        fields.append(att)

    uncertainties = OrderedDict()
    unc_elements = root.getElementsByTagName("event_specific_uncertainty")
    for uncel in unc_elements:
        key = uncel.getAttribute("name")
        value = float(uncel.getAttribute("value"))
        try:
            numsta = int(uncel.getAttribute("numsta"))
        except:
            numsta = 0
        uncertainties[key] = (value, numsta)

    return (griddict, eventdict, specdict, fields, uncertainties)


def readShakeFile(fileobj, adjust="bounds"):
    """Reads in the data and metadata for a ShakeMap object (can be passed to ShakeGrid
    constructor).

    Args:
        fileobj (IO):
            File-like object representing an open ShakeMap grid file.
        adjust (str):
            String (one of 'bounds','res') - adjust some of the ShakeMap parameters as
            necessary (usually "bounds").
                None:
                    All input parameters are assumed to be self-consistent, an
                    exception will be raised if they are not.
                'bounds':
                    dx/dy, nx/ny, xmin/ymax are assumed to be correct, xmax/ymin
                    will be recalculated.
                'res':
                    nx/ny, xmin/ymax, xmax/ymin and assumed to be correct, dx/dy
                    will be recalculated.
    Returns:
        Tuple containing:
            - Ordered Dictionary with the data layers in ShakeMap (MMI, PGA, PGV, etc.)
            - Geo dictionary describing the spatial extent and resolution of all the
            layers.
            - Dictionary representing the event element in the ShakeMap header.
            - Dictionary representing the grid element in the ShakeMap header.
            - Dictionary representing the list of event_specific_uncertainty elements in
            the ShakeMap header.
    """
    griddict, eventdict, specdict, fields, uncertainties = _getHeaderData(fileobj)
    nx = specdict["nlon"]
    ny = specdict["nlat"]
    layers = OrderedDict()

    # use pandas read_csv to read in the actual data - this should be faster than
    # numpy's loadtxt
    columns = fields[:]
    columns.insert(0, "lat")
    columns.insert(0, "lon")
    dframe = pd.read_csv(
        fileobj, sep=r"\s+", names=columns, header=None, comment="<", dtype=np.float32
    )
    for field in fields:
        layers[field] = dframe[field].values.reshape(ny, nx)

    # use the numpy loadtxt function to read in the actual data
    # we're cheating by telling numpy.loadtxt that the last two lines of the XML
    # file are comments
    # data = np.loadtxt(fileobj,comments='<').astype('float32')
    # data = data[:,2:] #throw away lat/lon columns
    # for i in range(0,len(fields)):
    #     field = fields[i]
    #     layers[field] = data[:,i].reshape(ny,nx)

    # create the geodict from the grid_spec element
    geodict = GeoDict(
        {
            "xmin": specdict["lon_min"],
            "xmax": specdict["lon_max"],
            "ymin": specdict["lat_min"],
            "ymax": specdict["lat_max"],
            "dx": specdict["nominal_lon_spacing"],
            "dy": specdict["nominal_lat_spacing"],
            "ny": specdict["nlat"],
            "nx": specdict["nlon"],
        },
        adjust=adjust,
    )

    return (layers, geodict, eventdict, griddict, uncertainties)


class ShakeGrid(MultiGrid):
    """
    A class that implements a MultiGrid object around ShakeMap grid.xml data sets.
    """

    def __init__(
        self, layers, geodict, eventDict, shakeDict, uncertaintyDict, field_keys={}
    ):
        """Construct a ShakeGrid object.

        Args:
            layers (OrderedDict):
                OrderedDict containing ShakeMap data layers (keys are 'pga', etc.,
                values are 2D arrays of data).
            geodict (dict):
                Dictionary specifying the spatial extent,resolution and shape of the
                data.
            eventDict (dict):
                Dictionary with elements:
                    - event_id String of event ID (i.e., 'us2015abcd')
                    - magnitude Float event magnitude
                    - depth Float event depth
                    - lat Float event latitude
                    - lon Float event longitude
                    - event_timestamp Datetime object representing event origin time.
                    - event_network Event originating network (i.e., 'us')
            shakeDict (dict):
                Dictionary with elements:
                    - event_id String of event ID (i.e., 'us2015abcd')
                    - shakemap_id String of ShakeMap ID (not necessarily the same as
                    the event ID)
                    - shakemap_version Integer ShakeMap version number (i.e., 1)
                    - code_version String version of ShakeMap code that created this
                    file (i.e.,'4.0')
                    - process_timestamp Datetime of when ShakeMap data was created.
                    - shakemap_originator String representing network that created the
                    ShakeMap
                    - map_status String, one of RELEASED, ??
                    - shakemap_event_type String, one of ['ACTUAL','SCENARIO']
            uncertaintyDict (dict):
                Dictionary with elements that have keys matching the layers keys, and
                values that are a tuple of that layer's uncertainty (float) and the
                number of stations used to determine that uncertainty (int).
            field_keys (dict):
                Dictionary containing keys matching at least some of input layers. For
                each key, a tuple of (UNITS,DIGITS) where UNITS is a string indicating
                the units of the layer quantity (e.g, cm/s) and DIGITS is the number of
                significant digits that the layer column should be printed with.
        Returns:
            A ShakeGrid object.
        """
        self._descriptions = OrderedDict()
        self._layers = OrderedDict()
        self._geodict = geodict
        for (layerkey, layerdata) in layers.items():
            self._layers[layerkey] = Grid2D(data=layerdata, geodict=geodict)
            self._descriptions[layerkey] = ""
        self._setEventDict(eventDict)
        self._setShakeDict(shakeDict)
        self._setUncertaintyDict(uncertaintyDict)
        self._field_keys = FIELDKEYS.copy()

        # assign the units and digits the user wants
        for layer, layertuple in field_keys.items():
            units, digits = layertuple
            fmtstr = "%%.%ig" % digits
            self._field_keys[layer] = (units, fmtstr)

        # if the user missed any, fill in with default values
        for layer in self._layers.keys():
            if layer in self._field_keys:
                continue
            self._field_keys[layer] = ("", "%.4g")

    @classmethod
    def getFileGeoDict(cls, shakefilename, adjust="bounds"):
        """Get the spatial extent, resolution, and shape of grids inside ShakeMap grid
        file.

        Args:
            shakefilename (str):
                File name of ShakeMap grid file.
            adjust (str):
                String (one of 'bounds','res') - adjust some of the ShakeMap parameters
                as necessary (usually "bounds").
                None:
                    All input parameters are assumed to be self-consistent, an exception
                    will be raised if they are not.
                'bounds':
                    dx/dy, nx/ny, xmin/ymax are assumed to be correct, xmax/ymin will
                    be recalculated.
                'res':
                    nx/ny, xmin/ymax, xmax/ymin and assumed to be correct, dx/dy will be
                    recalculated.
        Returns:
            GeoDict specifying spatial extent, resolution, and shape of grids inside
            ShakeMap grid file.
        """
        isFileObj = False
        if not hasattr(shakefilename, "read"):
            shakefile = open(shakefilename, "r")
        else:
            isFileObj = True
            shakefile = shakefilename
        griddict, eventdict, specdict, fields, uncertainties = _getHeaderData(shakefile)
        if isFileObj:
            shakefile.close()
        geodict = GeoDict(
            {
                "xmin": specdict["lon_min"],
                "xmax": specdict["lon_max"],
                "ymin": specdict["lat_min"],
                "ymax": specdict["lat_max"],
                "dx": specdict["nominal_lon_spacing"],
                "dy": specdict["nominal_lat_spacing"],
                "ny": specdict["nlat"],
                "nx": specdict["nlon"],
            },
            adjust=adjust,
        )
        return geodict

    @classmethod
    def load(
        cls,
        shakefilename,
        samplegeodict=None,
        resample=False,
        method="linear",
        doPadding=False,
        padValue=np.nan,
        adjust="bounds",
    ):

        # readShakeFile takes a file object.  Figure out if shakefilename is a file
        # name or file object.
        if not hasattr(shakefilename, "read"):
            shakefile = open(shakefilename, "r")
        else:
            shakefile = shakefilename

        # read everything from the file
        (layers, fgeodict, eventDict, shakeDict, uncertaintyDict) = readShakeFile(
            shakefile, adjust=adjust
        )

        # If the sample grid is aligned with the host grid, then resampling won't
        # accomplish anything
        # if samplegeodict is not None and fgeodict.isAligned(samplegeodict):
        #     resample = False

        # get area of shakemap that intersects with the desired input sampling grid
        if samplegeodict is not None:
            sampledict = fgeodict.getIntersection(samplegeodict)
        else:
            sampledict = fgeodict

        # Ensure that the two grids at least 1) intersect and 2) are aligned if
        # resampling is True.
        # parent static method, may raise an exception
        Grid2D.verifyBounds(fgeodict, sampledict, resample=resample)

        pad_dict = Grid2D.getPadding(
            fgeodict, samplegeodict, doPadding=doPadding
        )  # parent static method
        newlayers = OrderedDict()
        newgeodict = None
        for layername, layerdata in layers.items():
            data, geodict = Grid2D.padGrid(layerdata, fgeodict, pad_dict)
            grid = Grid2D(data, geodict)
            if resample:
                grid = grid.interpolateToGrid(samplegeodict, method=method)

            if np.any(np.isinf(grid._data)):
                grid._data[np.isinf(grid._data)] = padValue

            newlayers[layername] = grid.getData()
            if newgeodict is None:
                newgeodict = grid.getGeoDict().copy()

        return cls(newlayers, newgeodict, eventDict, shakeDict, uncertaintyDict)

    def interpolateToGrid(self, geodict, method="linear"):
        """
        Given a geodict specifying another grid extent and resolution, resample all
        layers in ShakeGrid to match.

        Args:
            geodict (dict):
                geodict dictionary from another grid whose extents are inside the
                extent of this ShakeGrid.
            method (str):
                Optional interpolation method - ['linear', 'cubic','nearest']
        Raises:
            DataSetException:
                If the GeoDict object upon which this function is being called is not
                completely contained by the grid to which this ShakeGrid is being
                resampled.
            DataSetException:
                If the method is not one of ['nearest','linear','cubic'] If the
                resulting interpolated grid shape does not match input geodict.
        This function modifies the internal griddata and geodict object variables.
        """
        multi = super(ShakeGrid, self).interpolateToGrid(geodict, method=method)
        layers = OrderedDict()
        geodict = multi.getGeoDict()
        # I need to get the layer data here...
        for layername in multi.getLayerNames():
            layers[layername] = multi.getLayer(layername).getData()
        eventdict = self.getEventDict()
        shakedict = self.getShakeDict()
        uncdict = self._uncertaintyDict
        shakemap = ShakeGrid(layers, geodict, eventdict, shakedict, uncdict)
        return shakemap

    def subdivide(self, finerdict, cellFill="max"):
        """Subdivide the cells of the host grid into finer-resolution cells.

        Args:
            finerdict (geodict):
                GeoDict object defining a grid with a finer resolution than the host
                grid.
            cellFill (str):
                String defining how to fill cells that span more than one host grid
                cell.
                Choices are:
                    'max': Choose maximum value of host grid cells.
                    'min': Choose minimum value of host grid cells.
                    'mean': Choose mean value of host grid cells.
        Returns:
            ShakeGrid instance with host grid values subdivided onto finer grid.
        Raises:
            DataSetException:
                When finerdict is not a) finer resolution or b) does not intersect.x or
                cellFill is not valid.
        """
        shakemap = super.subdivide(finerdict, cellFill=cellFill)
        shakemap._setEventDict(self.getEventDict())
        shakemap._setShakeDict(self.getShakeDict())
        shakemap._setUncertaintyDict(self.self._uncertaintyDict)
        return shakemap

    def save(self, filename, version=1):
        """Save a ShakeGrid object to the grid.xml format.

        Args:
            filename (str):
                File name or file-like object.
            version (int):
                Integer Shakemap version number.
        """

        # handle differences btw python2 and python3
        isThree = True
        if sys.version_info.major == 2:
            isThree = False

        isFile = False
        if not hasattr(filename, "read"):
            isFile = True
            f = open(filename, "wb")
        else:
            f = filename
        SCHEMA1 = "http://www.w3.org/2001/XMLSchema-instance"
        SCHEMA2 = "http://earthquake.usgs.gov/eqcenter/shakemap"
        SCHEMA3 = "http://earthquake.usgs.gov http://earthquake.usgs.gov/eqcenter/shakemap/xml/schemas/shakemap.xsd"

        f.write(b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>')
        fmt = '<shakemap_grid xmlns:xsi="%s" xmlns="%s" xsi:schemaLocation="%s" event_id="%s" shakemap_id="%s" shakemap_version="%i" code_version="%s" process_timestamp="%s" shakemap_originator="%s" map_status="%s" shakemap_event_type="%s">\n'
        tpl = (
            SCHEMA1,
            SCHEMA2,
            SCHEMA3,
            self._shakeDict["event_id"],
            self._shakeDict["shakemap_id"],
            self._shakeDict["shakemap_version"],
            self._shakeDict["code_version"],
            datetime.utcnow().strftime(TIMEFMT),
            self._shakeDict["shakemap_originator"],
            self._shakeDict["map_status"],
            self._shakeDict["shakemap_event_type"],
        )
        if isThree:
            f.write(bytes(fmt % tpl, "utf-8"))
        else:
            f.write(fmt % tpl)

        # location string could have non-valid XML characters in it (like &). Make that
        # string safe for XML before we write it out
        locstr = saxutils.escape(self._eventDict["event_description"])

        fmt = '<event event_id="%s" magnitude="%.1f" depth="%.1f" lat="%.4f" lon="%.4f" event_timestamp="%s" event_network="%s" event_description="%s"%s />\n'
        event_extras = ""
        if "intensity_observations" in self._eventDict:
            event_extras += (
                ' intensity_observations="%s"'
                % self._eventDict["intensity_observations"]
            )
        if "seismic_stations" in self._eventDict:
            event_extras += (
                ' seismic_stations="%s"' % self._eventDict["seismic_stations"]
            )
        if "point_source" in self._eventDict:
            event_extras += ' point_source="%s"' % self._eventDict["point_source"]
        tpl = (
            self._eventDict["event_id"],
            self._eventDict["magnitude"],
            self._eventDict["depth"],
            self._eventDict["lat"],
            self._eventDict["lon"],
            self._eventDict["event_timestamp"].strftime(TIMEFMT),
            self._eventDict["event_network"],
            locstr,
            event_extras,
        )
        if isThree:
            f.write(bytes(fmt % tpl, "utf-8"))
        else:
            f.write(fmt % tpl)
        fmt = '<grid_specification lon_min="%.4f" lat_min="%.4f" lon_max="%.4f" lat_max="%.4f" nominal_lon_spacing="%.4f" nominal_lat_spacing="%.4f" nlon="%i" nlat="%i"/>'
        tpl = (
            self._geodict.xmin,
            self._geodict.ymin,
            self._geodict.xmax,
            self._geodict.ymax,
            self._geodict.dx,
            self._geodict.dy,
            self._geodict.nx,
            self._geodict.ny,
        )
        if isThree:
            f.write(bytes(fmt % tpl, "utf-8"))
        else:
            f.write(fmt % tpl)
        fmt = '<event_specific_uncertainty name="%s" value="%.4f" numsta="%i" />\n'
        for (key, unctuple) in self._uncertaintyDict.items():
            value, numsta = unctuple
            tpl = (key, value, numsta)
            if isThree:
                f.write(bytes(fmt % tpl, "utf-8"))
            else:
                f.write(fmt % tpl)
        f.write(b'<grid_field index="1" name="LON" units="dd" />\n')
        f.write(b'<grid_field index="2" name="LAT" units="dd" />\n')
        idx = 3
        fmt = '<grid_field index="%i" name="%s" units="%s" />\n'
        data_formats = ["%.4f", "%.4f"]
        for field in self._layers.keys():
            tpl = (idx, field.upper(), self._field_keys[field][0])
            data_formats.append(self._field_keys[field][1])
            if isThree:
                db = bytes(fmt % tpl, "utf-8")
            else:
                db = fmt % tpl
            f.write(db)
            idx += 1
        f.write(b"<grid_data>\n")
        lat, lon = Grid().getLatLonMesh(self._geodict)

        # let's see if we can use pandas to write data out as well
        # this was really slow, mostly because we had to make strings out
        # of each column in order to get column-specific formatting.
        # return to this someday and re-investigate.

        # ldict = OrderedDict()
        # for lname,lgrid in self._layers.items():
        #     ldict[lname] = lgrid.getData().flatten()

        # df = pd.DataFrame.from_dict(ldict)
        # df['lat'] = lat.flatten()
        # df['lon'] = lon.flatten()
        # cols = df.columns.tolist()
        # cols.remove('lat')
        # cols.remove('lon')
        # cols.insert(0,'lat')
        # cols.insert(0,'lon')
        # df = df[cols]
        # for field,fieldtpl in FIELDKEYS.items():
        #     fieldfmt = fieldtpl[1]
        #     df[field].map(lambda x: fieldfmt % x)
        # df.to_csv(f,sep=' ')

        nfields = 2 + len(self._layers)
        data = np.zeros((self._geodict.ny * self._geodict.nx, nfields))
        # the data are ordered from the top left, so we need to invert the latitudes to
        # start from the top left
        lat = lat[::-1]
        data[:, 0] = lon.flatten()
        data[:, 1] = lat.flatten()
        fidx = 2
        for grid in self._layers.values():
            data[:, fidx] = grid.getData().flatten()
            fidx += 1
        np.savetxt(f, data, delimiter=" ", fmt=data_formats)
        f.write(b"</grid_data>\n</shakemap_grid>\n")
        if isFile:
            f.close()

    def _checkType(self, key, dtype):
        """Internal method used to validate the types of the input dictionaries used in
        constructor.

        Args:
            key (str):
                String key value
            dtype (str):
                Expected data type of key.
        Returns:
            True if key matches expected dtype, False if not.
        """
        # In Python 3 str type is now unicode by default, no such thing as unicode any
        # more.
        if sys.version_info.major == 2:
            strtypes = str
        else:
            strtypes = (str,)
        if dtype == "string" and (not isinstance(key, strtypes)):
            return False
        if dtype == "int" and not isinstance(key, int):
            return False
        if dtype == "float" and not isinstance(key, float):
            return False
        if dtype == "datetime" and not isinstance(key, datetime):
            return False
        return True

    def _setEventDict(self, eventdict):
        """Set the event dictionary, validating all values in the dictionary.

        Args:
            eventdict (dict):
                Event dictionary (see constructor).
        Raises:
            DataSetException:
                When one of the values in the dictionary does not match its expected type.
        """
        for (key, dtype) in EVENTKEYS.items():
            if key not in eventdict:
                raise DataSetException('eventdict is missing key "%s"' % key)
            if not self._checkType(eventdict[key], dtype):
                raise DataSetException(
                    'eventdict key value "%s" is the wrong datatype'
                    % str(eventdict[key])
                )
        self._eventDict = eventdict.copy()

    def getEventDict(self):
        """Get the event dictionary (the attributes of the "event" element in the
        ShakeMap header).

        Returns:
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

    def _setShakeDict(self, shakedict):
        """Set the shake dictionary, validating all values in the dictionary.

        Args:
            shakedict (dict):
                Shake dictionary (see constructor).
        Raises:
            DataSetException:
                When one of the values in the dictionary does not match its expected
                type.
        """
        for (key, dtype) in GRIDKEYS.items():
            if key not in shakedict:
                raise DataSetException('shakedict is missing key "%s"' % key)
            if not self._checkType(shakedict[key], dtype):
                raise DataSetException(
                    'shakedict key value "%s" is the wrong datatype'
                    % str(shakedict[key])
                )
        self._shakeDict = shakedict.copy()

    def getShakeDict(self):
        """Get the shake dictionary (the attributes of the "shakemap_grid" element in
        the ShakeMap header).

        Returns:
            Dictionary containing the following fields:
                - event_id: String like "us2016abcd".
                - shakemap_id: String like "us2016abcd".
                - shakemap_version: Version of the map that has been created.
                - code_version: Version of the ShakeMap software that was used to
                create the map.
                - shakemap_originator: Network that created the ShakeMap.
                - map_status: One of 'RELEASED' or 'REVIEWED'.
                - shakemap_event_type: One of 'ACTUAL' or 'SCENARIO'.
        """
        return self._shakeDict

    def _setUncertaintyDict(self, uncertaintyDict):
        """Set the uncertainty dictionary.

        Args:
            uncertaintyDict (dict):
                Uncertainty dictionary (see constructor).
        """
        self._uncertaintyDict = uncertaintyDict.copy()
