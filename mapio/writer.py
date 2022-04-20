# stdlib imports
import warnings
import os.path
import sys
from collections import OrderedDict

# third party imports
import netCDF4 as netcdf
import rasterio
import numpy as np
from rasterio.profiles import DefaultGTiffProfile

# local imports
from mapio.geodict import affine_from_geodict


def get_nodata(grid):
    if not np.isnan(np.sum(grid._data)):
        # no nan values
        NODATA = None
        return NODATA
    # try to have a nice readable NODATA value in the header file
    if grid._geodict.nodata is not None:
        NODATA = grid._geodict.nodata
    elif np.isnan(np.sum(grid._data)):
        # this grid has nans
        NODATA = np.nan
    else:
        zmin = np.nanmin(grid._data)
        zmax = np.nanmax(grid._data)
        if grid._data.dtype in [np.int8, np.int16, np.int32]:
            nodata = np.array([-1 * int("9" * i) for i in range(3, 20)])
            if zmin > nodata[-1]:
                NODATA = nodata[np.where(nodata < zmin)[0][0]]
            else:
                # otherwise just pick an arbitrary value smaller than our
                # smallest
                NODATA = zmin - 1
        else:
            nodata = np.array([int("9" * i) for i in range(3, 20)])
            if zmin < nodata[-1]:
                NODATA = nodata[np.where(nodata > zmin)[0][0]]
            else:
                # otherwise just pick an arbitrary value smaller than our
                # smallest
                NODATA = zmax + 1
    return NODATA


def _getHeader(grid):
    hdr = {}
    if sys.byteorder == "little":
        hdr["BYTEORDER"] = "LSBFIRST"
    else:
        hdr["BYTEORDER"] = "MSBFIRST"
    hdr["LAYOUT"] = "BIL"
    hdr["NROWS"], hdr["NCOLS"] = grid._data.shape
    hdr["NBANDS"] = 1
    if grid._data.dtype == np.uint8:
        hdr["NBITS"] = 8
        hdr["PIXELTYPE"] = "UNSIGNEDINT"
    elif grid._data.dtype == np.int8:
        hdr["NBITS"] = 8
        hdr["PIXELTYPE"] = "SIGNEDINT"
    elif grid._data.dtype == np.uint16:
        hdr["NBITS"] = 16
        hdr["PIXELTYPE"] = "UNSIGNEDINT"
    elif grid._data.dtype == np.int16:
        hdr["NBITS"] = 16
        hdr["PIXELTYPE"] = "SIGNEDINT"
    elif grid._data.dtype == np.uint32:
        hdr["NBITS"] = 32
        hdr["PIXELTYPE"] = "UNSIGNEDINT"
    elif grid._data.dtype == np.int32:
        hdr["NBITS"] = 32
        hdr["PIXELTYPE"] = "SIGNEDINT"
    elif grid._data.dtype == np.float32:
        hdr["NBITS"] = 32
        hdr["PIXELTYPE"] = "FLOAT"
    elif grid._data.dtype == np.float64:
        hdr["NBITS"] = 32
        hdr["PIXELTYPE"] = "FLOAT"
    else:
        raise KeyError('Data type "%s" not supported.' % str(grid._data.dtype))
    hdr["BANDROWBYTES"] = hdr["NCOLS"] * (hdr["NBITS"] / 8)
    hdr["TOTALROWBYTES"] = hdr["NCOLS"] * (hdr["NBITS"] / 8)
    hdr["ULXMAP"] = grid._geodict.xmin
    hdr["ULYMAP"] = grid._geodict.ymax
    hdr["XDIM"] = grid._geodict.dx
    hdr["YDIM"] = grid._geodict.dy
    NODATA = get_nodata(grid)
    hdr["NODATA"] = NODATA
    keys = [
        "BYTEORDER",
        "LAYOUT",
        "NROWS",
        "NCOLS",
        "NBANDS",
        "NBITS",
        "BANDROWBYTES",
        "TOTALROWBYTES",
        "PIXELTYPE",
        "ULXMAP",
        "ULYMAP",
        "XDIM",
        "YDIM",
        "NODATA",
    ]
    hdr2 = OrderedDict()
    for key in keys:
        hdr2[key] = hdr[key]
    return hdr2


def write(grid, filename, format_type, do_compression=False):
    """Save a GMTGrid object to a file.

    Args:
        filename (str):
            Name of desired output file.
        format_type (str):
            One of 'hdf', 'esri', 'tiff'.
    Raises:
        KeyError -- When format_type not one of ('netcdf', 'hdf', 'esri', 'tiff')
    """
    if format_type not in ["netcdf", "hdf", "esri", "tiff"]:
        msg = 'Only "netcdf", "hdf", "esri", and "tiff" formats are supported.'
        raise KeyError(msg)
    if format_type in ["netcdf", "hdf"]:
        format_mapper = {"hdf": "NETCDF4", "netcdf": "NETCDF3_CLASSIC"}

        f = netcdf.Dataset(filename, "w", format=format_mapper[format_type])
        NODATA = get_nodata(grid)
        if NODATA is not None:
            f.setncattr("_Fill_Value", NODATA)
        m, n = grid._data.shape
        dx = f.createDimension("x", n)  # noqa
        dy = f.createDimension("y", m)  # noqa
        x = f.createVariable("x", np.float64, ("x"))
        x.axis = "X"
        y = f.createVariable("y", np.float64, ("y"))
        y.axis = "Y"
        if grid._geodict.xmin < grid._geodict.xmax:
            x[:] = np.linspace(grid._geodict.xmin, grid._geodict.xmax, grid._geodict.nx)
        else:
            x[:] = np.linspace(
                grid._geodict.xmin, grid._geodict.xmax + 360, grid._geodict.nx
            )
        y[:] = np.linspace(grid._geodict.ymin, grid._geodict.ymax, grid._geodict.ny)
        if NODATA is not None:
            z = f.createVariable("z", grid._data.dtype, ("y", "x"), fill_value=NODATA)
        else:
            z = f.createVariable("z", grid._data.dtype, ("y", "x"))
        z[:] = np.flipud(grid._data)
        z.actual_range = np.array((np.nanmin(grid._data), np.nanmax(grid._data)))
        f.close()
    elif format_type == "esri":
        hdr = _getHeader(grid)
        # create a reference to the data - this may be overridden by a
        # downcasted version for doubles
        data = grid._data
        if grid._data.dtype == np.float32:
            # so we can find/reset nan values without screwing up original data
            data = grid._data.astype(np.float32)
            data[np.isnan(data)] = hdr["NODATA"]
        elif grid._data.dtype == np.float64:
            data = grid._data.astype(np.float32)
            data[np.isnan(data)] = hdr["NODATA"]
            warnings.warn(
                UserWarning(
                    "Down-casting double precision "
                    "floating point to single precision"
                )
            )

        data.tofile(filename)
        # write out the header file
        basefile, ext = os.path.splitext(filename)
        hdrfile = basefile + ".hdr"
        f = open(hdrfile, "wt")
        for (key, value) in hdr.items():
            value = hdr[key]
            f.write("%s  %s\n" % (key, str(value)))
        f.close()
    elif format_type == "tiff":
        geodict = grid.getGeoDict()
        transform = affine_from_geodict(geodict)
        profile = DefaultGTiffProfile()
        # default profile has compression on - turn it off if requested.
        if not do_compression:
            profile.pop("compress", None)

        # stufff
        profile["height"] = geodict.ny
        profile["width"] = geodict.nx
        profile["count"] = 1
        profile["dtype"] = grid._data.dtype
        profile["crs"] = geodict.projection
        profile["transform"] = transform
        new_dataset = rasterio.open(filename, "w", **profile)
        new_dataset.write(grid._data, 1)
        new_dataset.close()
