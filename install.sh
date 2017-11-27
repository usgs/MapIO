#!/bin/bash

VENV=mapio
PYVER=3.5

# Is the reset flag set?
reset=0
while getopts r FLAG; do
  case $FLAG in
    r)
        reset=1
        echo "Letting conda sort out dependencies..."
      ;;
  esac
done



unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
    DEPARRAY=(numpy=1.13.3 \
              scipy=1.0.0 \
              matplotlib=2.1.0 \
              rasterio=0.36.0 \
              pandas=0.21.0 \
              shapely=1.6.2
              h5py=2.7.1 \
              gdal=2.1.4 \
              pytest=3.2.5 \
              pytest-cov=2.5.1 \
              ipython=6.2.1 \
              cartopy=0.15.1 \
              fiona=1.7.10 \
              pycrypto=2.6.1 \
              paramiko=2.3.1 \
              psutil=5.4.0 \
              beautifulsoup4=4.6.0)
elif [[ "$unamestr" == 'FreeBSD' ]] || [[ "$unamestr" == 'Darwin' ]]; then
    if [ $reset -eq 0 ]; then
        DEPARRAY=(numpy=1.13.3 \
                  scipy=1.0.0 \
                  matplotlib=2.1.0 \
                  rasterio=0.36.0 \
                  pandas=0.21.0 \
                  shapely=1.6.2 \
                  h5py=2.7.1 \
                  gdal=2.1.4 \
                  pytest=3.2.5 \
                  pytest-cov=2.5.1 \
                  ipython=6.2.1 \
                  cartopy=0.15.1 \
                  fiona=1.7.10 \
                  pycrypto=2.6.1 \
                  paramiko=2.3.1 \
                  psutil=5.4.0 \
                  beautifulsoup4=4.6.0)
    else
        echo "Letting conda sort out dependencies..."
        DEPARRAY=(numpy \
                  scipy \
                  matplotlib \
                  rasterio \
                  pandas \
                  shapely \
                  h5py \
                  gdal \
                  pytest \
                  pytest-cov \
                  ipython \
                  cartopy \
                  fiona \
                  pycrypto \
                  paramiko \
                  psutil \
                  beautifulsoup4)
    fi
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

if [ $? -ne 0 ]; then
    echo "Failed to create conda environment.  Resolve any conflicts, then try again."
    exit
fi

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
