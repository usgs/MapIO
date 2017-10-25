#!/usr/bin/env python

# stdlib imports
from datetime import datetime
import collections
import time
import io
import copy

# third party imports
import h5py
import numpy as np
from impactutils.io.container import HDFContainer,_get_type_list,_drop_item

#local imports
from mapio.grid2d import Grid2D
from mapio.geodict import GeoDict


class GridHDFContainer(HDFContainer):
    def setGrid(self,name,grid,metadata=None):
        """Store a Grid2D object as a dataset.
        
        Args:
          name (str): Name of the Grid2D object to be stored.
          grid (Grid2D): Grid2D object to be stored.
          metadata (dict): Simple dictionary (values of strings and numbers).

        Returns:
          HDF Dataset containing grid and metadata.
        """
        grid_name = '__grid_%s__' % name
        array_metadata = grid.getGeoDict().asDict()
        data = grid.getData()
        if metadata is not None:
            for key,value in metadata.items():
                array_metadata[key] = value
        dset = self._hdfobj.create_dataset(grid_name, data=data)
        for key, value in array_metadata.items():
            dset.attrs[key] = value
        return dset

    def getGrid(self,name):
        """
        Retrieve a Grid2D object and any associated metadata from the container.

        Args:
            name (str):
                The name of the Grid2D object stored in the container.

        Returns:
            (tuple) Grid2D object, and a dictionary of metadata.
        """
        array_name = '__grid_%s__' % name
        if array_name not in self._hdfobj:
            raise LookupError('Array %s not in %s' % (name,self.getFileName()))
        dset = self._hdfobj[array_name]
        data = dset[()]
        metadata = {}
        for key, value in dset.attrs.items():
            metadata[key] = value
        grid_keys = ['xmin','xmax','ymin','ymax','nx','ny','dx','dy']
        array_metadata = {}
        meta_metadata = {}
        for key,value in metadata.items():
            if key in grid_keys:
                array_metadata[key] = value
            else:
                meta_metadata[key] = value
        geodict = GeoDict(array_metadata)
        grid = Grid2D(data,geodict)
        return grid,meta_metadata

    def getGrids(self):
        """
        Return list of names of Grid2D objects stored in container.

        Returns:
          (list) List of names of Grid2D objects stored in container.
        """
        
        grids = _get_type_list(self._hdfobj,'grid')
        return grids

    def dropGrid(self,name):
        """
        Delete Grid2D object from container.

        Args:
          name (str):
                The name of the Grid2D object to be deleted.

        """
        _drop_item(self._hdfobj,name,'grid')



