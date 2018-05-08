#!/usr/bin/env python

from mapio.gridcontainer import GridHDFContainer
from mapio.grid2d import Grid2D
from mapio.geodict import GeoDict
import numpy as np
import pandas as pd
from datetime import datetime
import tempfile
import os.path

def test_grid_hdf_container():
    f,fname = tempfile.mkstemp()
    os.close(f)
    try:
        #test grid container
        container = GridHDFContainer.create(fname)

        # before we put anything in here, let's make sure we get empty lists from
        # all of the methods that are supposed to return lists of stuff.
        assert container.getGrids() == []
        
        #test grid2d
        geodict = GeoDict.createDictFromBox(-118.5,-114.5,32.1,36.7,0.01,0.02)
        nrows,ncols = geodict.ny,geodict.nx
        data = np.random.rand(nrows,ncols)
        metadata = {'name':'Gandalf',
                    'color':'white',
                    'powers':'magic'}
        grid = Grid2D(data,geodict)
        container.setGrid('testgrid',grid,metadata=metadata)
        outgrid,outmetadata = container.getGrid('testgrid')
        np.testing.assert_array_equal(outgrid.getData(),data)
        assert outgrid.getGeoDict() == geodict
        assert outmetadata == metadata

        #set another grid without compression
        geodict = GeoDict.createDictFromBox(-119.5,-115.5,32.3,37.7,0.01,0.02)
        nrows,ncols = geodict.ny,geodict.nx
        data = np.random.rand(nrows,ncols)
        metadata = {'name':'Legolas',
                    'color':'green',
                    'powers':'stealth'}
        grid2 = Grid2D(data,geodict)
        container.setGrid('testgrid2',grid2,metadata=metadata,compression=False)
        outgrid2,outmetadata2 = container.getGrid('testgrid2')
        np.testing.assert_array_equal(outgrid2.getData(),data)
        assert outgrid2.getGeoDict() == geodict
        assert outmetadata2 == metadata
        
        #test getGridNames()
        names = container.getGrids()
        assert sorted(names) == ['testgrid','testgrid2']

        #test looking for a grid that does not exist
        try:
            container.getGrid('foo')
        except LookupError as le:
            pass

        #test dropping a grid
        container.dropGrid('testgrid2')

        container.close()
        container2 = GridHDFContainer.load(fname)
        names = container2.getGrids()
        assert sorted(names) == ['testgrid']
    except:
        assert 1==2
    finally:
        os.remove(fname)

if __name__ == '__main__':
    test_grid_hdf_container()
