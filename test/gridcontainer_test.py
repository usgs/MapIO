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

        #test getGridNames() and getArrayNames()
        names = container.getGrids()
        assert names == ['testgrid']

        container.close()
        container2 = GridHDFContainer.load(fname)
        names = container2.getGrids()
        assert names == ['testgrid']
    except:
        assert 1==2
    finally:
        os.remove(fname)

if __name__ == '__main__':
    test_grid_hdf_container()
