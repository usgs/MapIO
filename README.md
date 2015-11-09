Introduction
------------

grid is a project designed to provide a library of classes for dealing
with various grid formats, and performing some simple spatial
operations on the data.  The code is documented fairly well, and there
are IPython notebooks in the distribution.  They can be viewed here:

 * GDAL (ESRI format) grids : https://github.com/mhearne-usgs/grid/blob/master/notebooks/GDALGrid.ipynb
 * GMT format grids: https://github.com/mhearne-usgs/grid/blob/master/notebooks/GMTGrid.ipynb
 * ShakeMap format grids: https://github.com/mhearne-usgs/grid/blob/master/notebooks/ShakeMap.ipynb
 * The Grid2D class (superclass of GDAL and GMT grids): https://github.com/mhearne-usgs/grid/blob/master/notebooks/Grid2D.ipynb

Dependencies and Installation
-----------------------------

This library depends on:
 * numpy: <a href="http://www.numpy.org/">http://www.numpy.org/</a>
 * scipy: <a href="http://scipy.org/scipylib/index.html">http://scipy.org/scipylib/index.html</a>
 * h5py: <a href="http://www.h5py.org/">http://www.h5py.org/</a>
 * rasterio: <a href="https://github.com/mapbox/rasterio">https://github.com/mapbox/rasterio</a>
 
These packages are all either installed automatically by the Anaconda scientific Python distribution, or easily installed using the conda command.  The final dependency:

 * openquake: <a href="http://www.globalquakemodel.org/openquake/about/">http://www.globalquakemodel.org/openquake/about/</a>

can be installed by using pip with git:

pip install git+git://github.com/gem/oq-hazardlib.git

To install this package:

pip install git+git://github.com/mhearne-usgs/grid.git

Uninstalling and Updating
-------------------------

To uninstall:

pip uninstall grid

To update:

pip install -U git+git://github.com/mhearne-usgs/grid.git

Interoperability
-----------------
The various grid classes in this repository read and write files in various formats.  

 - GDALGrid reads/writes ESRI format files, which can be opened in ArcGIS software and converted to Grids.
 - GMTGrid reads/writes GMT format files, which can be opened in GMT software and also ArcGIS.
 - MultiHazardGrid reads/writes HDF files, which can be opened in GMT (see below), Matlab (see sample function below), and many other programming languages with HDF support (C/C++, FORTRAN, Java, Perl).

Accessing MultiHazardGrid HDF files in GMT (requires that you know the layers in the HDF file):
<pre>
grdinfo filename.hdf?mmi

grdinfo filename.hdf?pga
</pre>

Accessing the MultiHazardGrid HDF files in Matlab can be accomplished using the h5info and h5read functions, as seen in sample function below:

<pre>
% readsmgrid - Read NEIC MultiHazardGrid HDF file.
% [gridstruct,geodict,origin,header] = readmultihaz(gridfile);
% Input:
% - gridfile is a valid path to a MultiHazardGrid HDF file.
% Output:
% - gridstruct is a structure with fields containing all of the data
%   layers.
% - geostruct is a Matlab structure, containing grid data and metadata
% describing it.  The fields are:
%  - ulxmap Upper left hand corner of upper left pixel X coordinate (decimal degrees).
%  - ulymap Upper left hand corner of upper left pixel X coordinate (decimal degrees).
%  - xdim Resolution in the X directioun (decimal degrees)
%  - ydim Resolution in the Y directioun (decimal degrees)
% - origin Structure containing fields: 
%     - id String of event ID (i.e., 'us2015abcd')
%     - source String containing originating network ('us')
%     - time Float event magnitude
%     - lat Float event latitude
%     - lon Float event longitude
%     - depth Float event depth
%     - magnitude Datetime object representing event origin time.
% - header Structure containing fields: 
%          - type Type of multi-layer earthquake induced hazard ('shakemap','gfe')
%          - version Integer product version (1)
%          - process_time Python datetime indicating when data was created.
%          - code_version String version of code that created this file (i.e.,'4.0')
%          - originator String representing network that created the hazard grid.
%          - product_id String representing the ID of the product (may be different from origin ID)
%          - map_status String, one of RELEASED, ??
%          - event_type String, one of ['ACTUAL','SCENARIO']
% NB - There is a recursive group called metadata that is not being parsed here.  
function [gridstruct,geodict,origin,header] = readmultihaz(gridfile)
    fstruct = h5info(gridfile);
    setnames = {fstruct.Datasets(:).Name};
    hasx = ~isempty(find(ismember(setnames,'x')));
    hasy = ~isempty(find(ismember(setnames,'y')));
    if ~hasx || ~hasy
        msg = sprintf('Missing one or both of required x and y datasets in %s',gridfile);
        ME = MException('readmultihaz:missingGeoref', msg);
        throw(ME)
    end
    gridstruct = struct();
    for i=1:length(setnames)
        if strcmpi(setnames{i},'x') || strcmpi(setnames{i},'y')
            continue;
        end
        setname = ['/' setnames{i}];
        gridstruct.(setnames{i}) = h5read(gridfile,setname);
    end
    
    xvar = h5read(gridfile,'/x');
    yvar = h5read(gridfile,'/y');
    ulxmap = xvar(1);
    ulymap = yvar(end);
    xdim = xvar(2)-xvar(1);
    ydim = yvar(2)-yvar(1);
    [nrows,ncols] = size(gridstruct.(setnames{1}));
    geodict = struct('ulxmap',ulxmap,'ulymap',ulymap,...
        'xdim',xdim,'ydim',ydim);
    
    groups = {fstruct.Groups(:).Name};
    headerindex = find(strcmpi(groups,'/header'));
    originindex = find(strcmpi(groups,'/origin'));
    metaindex = find(strcmpi(groups,'/metadata'));
    header = struct();
    headerstruct = fstruct.Groups(headerindex);
    for i=1:length(headerstruct.Attributes)
        attr = headerstruct.Attributes(i);
        header.(attr.Name) = attr.Value;
    end
    origin = struct();
    originstruct = fstruct.Groups(originindex);
    for i=1:length(originstruct.Attributes)
        attr = originstruct.Attributes(i);
        origin.(attr.Name) = attr.Value;
    end
</pre>