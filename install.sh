#!/bin/bash

VENV=mapio

unamestr=`uname`
if [ "$unamestr" == 'Linux' ]; then
    source ~/.bashrc
elif [ "$unamestr" == 'FreeBSD' ] || [ "$unamestr" == 'Darwin' ]; then
    source ~/.bash_profile
fi

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

# Is conda installed?
conda --version
if [ $? -ne 0 ]; then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O miniconda.sh;
    bash miniconda.sh -f -b -p $HOME/miniconda
    . $HOME/miniconda/etc/profile.d/conda.sh
fi

# Choose an environment file based on platform
unamestr=`uname`
if [ "$unamestr" == 'Linux' ]; then
    env_file=environment_linux.yml
elif [ "$unamestr" == 'FreeBSD' ] || [ "$unamestr" == 'Darwin' ]; then
    env_file=environment_osx.yml
fi

# let's just always use the non-platform specific env file.
reset=1

# If the user has specified the -r (reset) flag, then create an
# environment based on only the named dependencies, without
# any versions of packages specified.
if [ $reset == 1 ]; then
    echo "Ignoring platform, letting conda sort out dependencies..."
    env_file=environment.yml
fi

echo "Environment file: $env_file"

# Turn off whatever other virtual environment user might be in
conda deactivate

# Create a conda virtual environment
echo "Creating the $VENV virtual environment:"
conda env create -f $env_file --force

# Activate the new environment
echo "Activating the $VENV virtual environment"
conda activate $VENV

# This package
echo "Installing mapio..."
pip install -e .

#tell the user they have to activate this environment
echo "Type 'conda activate ${VENV}' to use this new virtual environment."
