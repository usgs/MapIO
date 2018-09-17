#!/bin/sh

rm samplegrid_*

# this makes the HDF5 (netcdf4) version of the test grid
# NB: GDAL DOES NOT LIKE THIS FILE. FIGURE OUT WHY!!!
gmt xyz2grd samplegrid.txt -Gsamplegrid_hdf.hdf=nf -I1.0 -R5.0/9.0/4.0/8.0 --IO_NC4_CHUNK_SIZE=5,5 --IO_NC4_DEFLATION_LEVEL=0
echo "Created sample HDF file."

# this makes the netcdf 3 version of the grid
gmt xyz2grd samplegrid.txt -Gsamplegrid_cdf.cdf=cf -I1.0 -R5.0/9.0/4.0/8.0
echo "Created sample CDF file."

# this makes the ESRI version of the grid
gdal_translate samplegrid_cdf.cdf samplegrid_flt.flt -of EHdr
echo "Created sample FLT file."

# make an CDF version of the "global" grid
gmt xyz2grd globalgrid.txt -Gglobalgrid_cdf.cdf=cf -R-120/180/-60/60 -I60/30
echo "Created global CDF file."

# clean up gmt history file
rm gmt.history
