#!/bin/bash

VENV=impact
PYVER=3.6

unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
    DEPARRAY=(numpy=1.11 \
              scipy=0.19.1 \
              matplotlib=2.0.2 \
              rasterio=1.0a2 \
              pandas=0.20.3 \
              shapely=1.5.17
              h5py=2.7.0 \
              gdal=2.1.4 \
              pytest=3.2.0 \
              pytest-cov=2.5.1 \
              jupyter=1.0.0 \
              ipython=6.1.0 \
              cartopy=0.15.1 \
              fiona=1.7.8 \
              pycrypto=2.6.1 \
              paramiko=2.2.1 \
              beautifulsoup4=4.5.3)
elif [[ "$unamestr" == 'FreeBSD' ]] || [[ "$unamestr" == 'Darwin' ]]; then
    DEPARRAY=(numpy=1.13.1 \
              scipy=0.19.1 \
              matplotlib=2.0.2 \
              rasterio=1.0a9 \
              pandas=0.20.3 \
              shapely=1.5.17 \
              h5py=2.7.0 \
              gdal=2.1.4 \
              pytest=3.2.0 \
              pytest-cov=2.5.1 \
              ipython=6.1.0 \
              cartopy=0.15.1 \
              fiona=1.7.8 \
              pycrypto=2.6.1 \
              paramiko=2.2.1 \
              beautifulsoup4=4.5.3)
fi

#if we're already in an environment called pager, switch out of it so we can remove it
source activate root

#add channels
conda update -q -y conda
conda config --prepend channels conda-forge
conda config --append channels digitalglobe # for rasterio v 1.0a9
conda config --append channels ioos # for rasterio v 1.0a2

#remove any previous virtual environments called libcomcat
CWD=`pwd`
cd $HOME;
conda remove --name $VENV --all -y
cd $CWD
    
#create a new virtual environment called $VENV with the below list of dependencies installed into it
conda create --name $VENV --yes --channel conda-forge python=$PYVER ${DEPARRAY[*]} -y

#activate the new environment
source activate $VENV

#install openquake from github
curl --max-time 300 --retry 3 -L https://github.com/gem/oq-engine/archive/master.zip -o openquake.zip
pip -v install --no-deps openquake.zip
rm openquake.zip

#install impactutils library
echo "Installing impactutils..."
curl --retry 3 -L https://github.com/usgs/earthquake-impact-utils/archive/master.zip -o impact.zip
pip install impact.zip
rm impact.zip

# This package
echo "Installing mapio..."
pip install -e .

#tell the user they have to activate this environment
echo "Type 'source activate ${VENV}' to use this new virtual environment."
