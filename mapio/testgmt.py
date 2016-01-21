#!/usr/bin/env python

#python 3 compatibility
from __future__ import print_function

import rasterio
from scipy.io import netcdf
import numpy as np
import subprocess
import sys

from gdal import GDALGrid
from gmt import GMTGrid

def getCommandOutput(cmd):
    """
    Internal method for calling external command.
    @param cmd: String command ('ls -l', etc.)
    @return: Three-element tuple containing a boolean indicating success or failure, 
    the stdout from running the command, and stderr.
    """
    proc = subprocess.Popen(cmd,
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                            )
    stdout,stderr = proc.communicate()
    retcode = proc.returncode
    if retcode == 0:
        retcode = True
    else:
        retcode = False
    return (retcode,stdout,stderr)

if __name__ == '__main__':
    #make a data set
    data = np.arange(0,16).reshape(4,4).astype(np.int32)
    geodict = {'xmin':0.5,'xmax':3.5,'ymin':0.5,'ymax':3.5,'xdim':1.0,'ydim':1.0,'nrows':4,'ncols':4}
    gmtgrid = GMTGrid(data,geodict)

    #save that data set to a grid
    gmtgrid.save('gmt_from_python.grd')

    #use gmt to get the value at 1.5,1.5 (should be 9)
    f = open('track.xy','wt')
    f.write('1.5 1.5\n')
    f.close()
    cmd = 'gmt grdtrack -nn track.xy -Ggmt_from_python.grd'
    res,stdout,stderr = getCommandOutput(cmd)
    
    print(stdout)

    #now create an XY file from our grid
    f = open('from_python.xyz','wt')
    for i in range(0,geodict['nrows']):
        for j in range(0,geodict['ncols']):
            lat,lon = gmtgrid.getLatLon(i,j)
            value = gmtgrid.getValue(lat,lon)
            f.write('%.1f %.1f %i\n' % (lon,lat,value))
    f.close()

    #now create a grid file from our XY file
    cmd = 'gmt xyz2grd -R0.5/3.5/0.5/3.5 -I1.0/1.0 from_python.xyz -Gfrom_gmt.grd'
    res,stdout,stderr = getCommandOutput(cmd)

    #now read in this grid using GMTGrid
    gmtgrid2 = GMTGrid.load('from_gmt.grd')
    np.testing.assert_almost_equal(data,gmtgrid2.getData())

    #now use gdal to convert that GMT grid to ESRI format
    cmd = 'gdal_translate from_gmt.grd from_gmt.bil -of EHdr'
    res,stdout,stderr = getCommandOutput(cmd)

    #now use our GDAL reader to get that grid data
    gdalgrid = GDALGrid.load('from_gmt.bil')
    np.testing.assert_almost_equal(data,gdalgrid.getData())

    #now use gdal to convert that ESRI grid back to netcdf
    cmd = 'gdal_translate from_gmt.bil from_gdal.grd -of GMT'
    res,stdout,stderr = getCommandOutput(cmd)

    #again use gmt to get the value at 1.5,1.5 (should be 9)
    cmd = 'gmt grdtrack -nn track.xy -Gfrom_gdal.grd'
    res,stdout,stderr = getCommandOutput(cmd)
    print(stdout)

    #now use our GMT reader to load that grid and compare to original
    gmtgrid3 = GMTGrid.load('from_gdal.grd')
    np.testing.assert_almost_equal(data,gmtgrid3.getData())
