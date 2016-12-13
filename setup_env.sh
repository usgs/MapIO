#!/bin/bash

VENV=mapio
PYVER=3.5

DEPARRAY=(numpy scipy matplotlib rasterio pandas shapely h5py gdal pytest pytest pytest-cov pytest-mpl cartopy basemap jupyter)

#turn off whatever other virtual environment user might be in
source deactivate
    
#remove any previous virtual environments called pager
CWD=`pwd`
cd $HOME;
conda remove --name $VENV --all -y
cd $CWD
    
#create a new virtual environment called $VENV with the below list of dependencies installed into it
conda create --name $VENV --yes --channel conda-forge python=3.5 ${DEPARRAY[*]} -y

#activate the new environment
source activate $VENV

#install some items separately
#conda install -y sqlalchemy #at the time of this writing, this is v1.0, and I want v1.1
conda install -y psutil

#do pip installs of those things that are not available via conda.
#do pip installs of those things that are not available via conda.
pip -v install https://github.com/gem/oq-hazardlib/archive/master.zip
pip install flake8
pip install pep8-naming

#tell the user they have to activate this environment
echo "Type 'source activate ${VENV}' to use this new virtual environment."
