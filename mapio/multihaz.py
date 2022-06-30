#!/usr/bin/env python

# python 3 compatibility
from __future__ import print_function

# stdlib imports
import sys
import collections
import datetime
import time
import os.path

# third party imports
import h5py
import numpy as np
from .multiple import MultiGrid
from .shake import ShakeGrid
from .grid2d import Grid2D
from .dataset import DataSetException
from .geodict import GeoDict


class MultiHazardGrid(MultiGrid):
    def __init__(self, layers, geodict, origin, header, metadata=None):
        """Construct a MultiHazardGrid object.

        Args:
        layers (OrderedDict):
          OrderedDict containing earthquake-induced hazard data layers (keys are 'pga',
          etc., values are 2D arrays of data).
        geodict (GeoDict):
          GeoDict object specifying the spatial extent,resolution and shape of the data.
        origin (dict):
          Dictionary with elements:
            - id String of event ID (i.e., 'us2015abcd')
            - source String containing originating network ('us')
            - time Python datetime of origin time, UTC.
            - lat Float event latitude
            - lon Float event longitude
            - depth Float event depth
            - magnitude Datetime object representing event origin time.
        header (header):
          Dictionary with elements:
            - type Type of multi-layer earthquake induced hazard ('shakemap','gfe')
            - version Integer product version (1)
            - process_time Python datetime indicating when data was created (UTC)
            - code_version String version of code that created this file (i.e.,'4.0')
            - originator String representing network that created the hazard grid.
            - product_id String representing the ID of the product (may be different
            from origin ID)
            - map_status String, one of RELEASED, ??
            - event_type String, one of ['ACTUAL','SCENARIO']
        metadata (dict):
          Dictionary of dictionaries containing other metadata users wish to preserve.
        Returns: A MultiHazardGrid object.
        """
        self._layers = collections.OrderedDict()
        self._geodict = geodict
        for (layerkey, layerdata) in layers.items():
            self._layers[layerkey] = Grid2D(data=layerdata, geodict=geodict)

        self.setHeader(header)
        self.setOrigin(origin)
        self.setMetadata(metadata)

    @classmethod
    def getFileGeoDict(cls, filename):
        """Get the spatial extent, resolution, and shape of grid inside MultiHazardGrid
        format HDF file.

        Args:
            filename (str):
                File name of HDF file.
        Returns:
            GeoDict specifying spatial extent, resolution, and shape of grid inside
            HDF file.
        """
        f = h5py.File(filename, "r")
        geodict = {}
        xvar = f["x"][:]
        yvar = f["y"][:]
        geodict["xmin"] = xvar[0]
        geodict["xmax"] = xvar[-1]
        geodict["ymin"] = yvar[0]
        geodict["ymax"] = yvar[-1]
        geodict["ny"] = len(yvar)
        geodict["nx"] = len(xvar)
        geodict["dx"] = xvar[1] - xvar[0]
        geodict["dy"] = yvar[1] - yvar[0]
        f.close()

        return GeoDict(geodict)

    def _saveDict(self, group, mydict):
        """
        Recursively save dictionaries into groups in an HDF file.

        Args:
            group (HDF):
                HDF group object to contain a given dictionary of data in HDF file.
            mydict (dict):
                Dictionary of values to save in group.  Dictionary can contain objects
                of the following types:
                    - str,unicode,int,float,long,list,tuple,np.ndarray,dict,
                    datetime.datetime,collections.OrderedDict
        """
        ALLOWED = [
            str,
            int,
            float,
            type(None),
            list,
            tuple,
            np.ndarray,
            dict,
            datetime.datetime,
            collections.OrderedDict,
        ]
        if sys.version_info.major == 2:
            ALLOWED.append(str)
            ALLOWED.append(int)

        for (key, value) in mydict.items():
            tvalue = type(value)
            if tvalue not in ALLOWED:
                raise DataSetException('Unsupported metadata value type "%s"' % tvalue)
            if not isinstance(value, dict):
                if isinstance(value, datetime.datetime):
                    value = time.mktime(value.timetuple())
                group.attrs[key] = value
            else:
                subgroup = group.create_group(key)
                self._saveDict(subgroup, value)

    @classmethod
    def _loadDict(cls, group):
        """Recursively load dictionaries from groups in an HDF file.

        Args:
            group (HDF5):
                HDF5 group object.
        Returns:
            Dictionary of metadata (possibly containing other dictionaries).
        """
        tdict = {}
        for (key, value) in group.attrs.items():  # attrs are NOT subgroups
            if key.find("time") > -1:
                value = value = datetime.datetime.utcfromtimestamp(value)
            tdict[key] = value
        for (key, value) in group.items():  # these are going to be the subgroups
            tdict[key] = cls._loadDict(value)
        return tdict

    def save(self, filename):
        """
        Save MultiHazardGrid object to HDF file.
        Georeferencing information will be saved as datasets "x" and "y".  Layers will
        be saved as datasets named by layer keys.  Dictionaries contained in "origin",
        and "header" will be saved in groups of those same names.  Dictionaries
        contained in the "metadata" dictionary will be contained in a series of
        recursive groups under a group called "metadata".

        Args:
            filename (str):
                Output desired filename (HDF format).
        """
        f = h5py.File(filename, "w")
        # Add in some attributes that will help make this GMT friendly...
        f.attrs["Conventions"] = "COARDS, CF-1.5"
        f.attrs["title"] = "filename"
        f.attrs["history"] = "Created with python MultiHazardGrid.save(%s)" % filename
        f.attrs["GMT_version"] = "NA"

        # create two top-level groups that should always be present
        header = f.create_group("header")
        self._saveDict(header, self._header)

        origin = f.create_group("origin")
        self._saveDict(origin, self._origin)

        # write out any other metadata, creating groups recursively as needed
        metadata = f.create_group("metadata")
        self._saveDict(metadata, self._metadata)

        xvar = np.linspace(self._geodict.xmin, self._geodict.xmax, self._geodict.nx)
        yvar = np.linspace(self._geodict.ymin, self._geodict.ymax, self._geodict.ny)
        x = f.create_dataset(
            "x", data=xvar, compression="gzip", shape=xvar.shape, dtype=str(xvar.dtype)
        )
        x.attrs["CLASS"] = "DIMENSION_SCALE"
        x.attrs["NAME"] = "x"
        x.attrs["_Netcdf4Dimid"] = 0  # no idea what this is
        x.attrs["long_name"] = "x"
        x.attrs["actual_range"] = np.array((xvar[0], xvar[-1]))

        y = f.create_dataset(
            "y", data=yvar, compression="gzip", shape=yvar.shape, dtype=str(yvar.dtype)
        )
        y.attrs["CLASS"] = "DIMENSION_SCALE"
        y.attrs["NAME"] = "y"
        y.attrs["_Netcdf4Dimid"] = 1  # no idea what this is
        y.attrs["long_name"] = "y"
        y.attrs["actual_range"] = np.array((yvar[0], yvar[-1]))

        for (layerkey, layer) in self._layers.items():
            dset = f.create_dataset(layerkey, data=layer.getData(), compression="gzip")
            dset.attrs["long_name"] = layerkey
            dset.attrs["actual_range"] = np.array(
                (np.nanmin(layer._data), np.nanmax(layer._data))
            )

        f.close()

    @classmethod
    def load(cls, filename):
        """
        Load data from an HDF file into a MultiHazardGrid object.

        Args:
            filename (str):
                HDF file containing data and metadata for ShakeMap or Secondary Hazards
                data.
        Returns:
            MultiHazardGrid object.
        """
        f = h5py.File(filename, "r")
        REQUIRED_GROUPS = ["origin", "header"]
        REQUIRED_DATASETS = ["x", "y"]
        for group in REQUIRED_GROUPS:
            if group not in f.keys():
                raise DataSetException('Missing required metadata group "%s"' % group)
        for dset in REQUIRED_DATASETS:
            if dset not in f.keys():
                raise DataSetException('Missing required data set "%s"' % dset)

        header = {}
        for (key, value) in f["header"].attrs.items():
            if key.find("time") > -1:
                value = datetime.datetime.utcfromtimestamp(value)
            header[key] = value

        origin = {}
        for (key, value) in f["origin"].attrs.items():
            if key.find("time") > -1:
                value = datetime.datetime.utcfromtimestamp(value)
            origin[key] = value

        if "metadata" in f.keys():
            metadata = cls._loadDict(f["metadata"])

        geodict = {}
        xvar = f["x"][:]
        yvar = f["y"][:]
        geodict["xmin"] = xvar[0]
        geodict["xmax"] = xvar[-1]
        geodict["ymin"] = yvar[0]
        geodict["ymax"] = yvar[-1]
        geodict["ny"] = len(yvar)
        geodict["nx"] = len(xvar)
        geodict["dx"] = xvar[1] - xvar[0]
        geodict["dy"] = yvar[1] - yvar[0]
        gd = GeoDict(geodict)
        layers = collections.OrderedDict()
        for key in f.keys():
            keytype = str(type(f[key]))
            if keytype.find("Dataset") > -1:
                if key in REQUIRED_DATASETS:
                    continue
                layers[key] = f[key][:]

        f.close()
        return cls(layers, gd, origin, header, metadata=metadata)

    def setHeader(self, header):
        """
        Set the header dictionary.

        Args:
            header (dict):
                Dictionary with elements:
                    - type Type of multi-layer earthquake induced hazard
                    ('shakemap','gfe')
                    - version Integer product version (1)
                    - process_time Python datetime indicating when data was created.
                    - code_version String version of code that created this file
                    (i.e.,'4.0')
                    - originator String representing network that created the hazard
                    grid.
                    - product_id String representing the ID of the product
                    (may be different from origin ID)
                    - map_status String, one of RELEASED, ??
                    - event_type String, one of ['ACTUAL','SCENARIO']
        """
        self._header = header.copy()  # validate later

    def setOrigin(self, origin):
        """
        Set the origin dictionary.

        Args:
            origin (dict):
                Dictionary with elements:
                    - id String of event ID (i.e., 'us2015abcd')
                    - source String containing originating network ('us')
                    - time Float event magnitude
                    - lat Float event latitude
                    - lon Float event longitude
                    - depth Float event depth
                    - magnitude Datetime object representing event origin time.
        """
        self._origin = origin.copy()  # validate later

    def setMetadata(self, metadata):
        """
        Set the metadata dictionary.

        Args:
            metadata (dict):
                Dictionary of dictionaries of metadata.  Each dictionary can contain
                any of the following types: str,unicode,int,float,long,list,tuple,
                np.ndarray,dict,datetime.datetime,collections.OrderedDict.
        """
        self._metadata = metadata.copy()

    def getHeader(self):
        """
        Return the header dictionary.

        Returns:
            Header dictionary (see setHeader()).
        """
        return self._header.copy()

    def getOrigin(self):
        """
        Return the origin dictionary.

        Returns:
            Origin dictionary (see setOrigin()).
        """
        return self._origin.copy()

    def getMetadata(self):
        """
        Return the dictionary of arbitrary metadata dictionaries.

        Returns:
          A dictionary of dictionaries containing arbitrary metadata.
        """
        return self._metadata.copy()


if __name__ == "__main__":
    shakefile = sys.argv[1]
    t1 = datetime.datetime.now()
    sgrid = ShakeGrid.load(shakefile)
    t2 = datetime.datetime.now()
    origin = {}
    origin["id"] = sgrid._eventDict["event_id"]
    origin["source"] = sgrid._eventDict["event_network"]
    origin["time"] = sgrid._eventDict["event_timestamp"]
    origin["lat"] = sgrid._eventDict["lat"]
    origin["lon"] = sgrid._eventDict["lon"]
    origin["depth"] = sgrid._eventDict["depth"]
    origin["magnitude"] = sgrid._eventDict["magnitude"]

    header = {}
    header["type"] = "shakemap"
    header["version"] = sgrid._shakeDict["shakemap_version"]
    header["process_time"] = sgrid._shakeDict["process_timestamp"]
    header["code_version"] = sgrid._shakeDict["code_version"]
    header["originator"] = sgrid._shakeDict["shakemap_originator"]
    header["product_id"] = sgrid._shakeDict["shakemap_id"]
    header["map_status"] = sgrid._shakeDict["map_status"]
    header["event_type"] = sgrid._shakeDict["shakemap_event_type"]

    layers = collections.OrderedDict()
    for (layername, layerdata) in sgrid.getData().items():
        layers[layername] = layerdata.getData()

    tdict = {"name": "fred", "family": {"wife": "wilma", "daughter": "pebbles"}}
    mgrid = MultiHazardGrid(
        layers, sgrid.getGeoDict(), origin, header, metadata={"flintstones": tdict}
    )
    mgrid.save("test.hdf")
    t3 = datetime.datetime.now()
    mgrid2 = MultiHazardGrid.load("test.hdf")
    t4 = datetime.datetime.now()
    xmlmb = os.path.getsize(shakefile) / float(1e6)
    hdfmb = os.path.getsize("test.hdf") / float(1e6)
    xmltime = (t2 - t1).seconds + (t2 - t1).microseconds / float(1e6)
    hdftime = (t4 - t3).seconds + (t4 - t3).microseconds / float(1e6)
    print("Input XML file size: %.2f MB (loading time %.3f seconds)" % (xmlmb, xmltime))
    print(
        "Output HDF file size: %.2f MB (loading time %.3f seconds)" % (hdfmb, hdftime)
    )
    os.remove("test.hdf")
