#!/usr/bin/env python

#python 3 compatibility
from __future__ import print_function

#stdlib imports
import os.path
import sys

#third party imports
import numpy as np

#hack the path so that I can debug these functions if I need to
homedir = os.path.dirname(os.path.abspath(__file__)) #where is this script?
mapiodir = os.path.abspath(os.path.join(homedir,'..'))
sys.path.insert(0,mapiodir) #put this at the front of the system path, ignoring any installed mapio stuff

from mapio.dataset import DataSet,DataSetException
from mapio.gridbase import Grid
from mapio.cloud import Cloud


def test():
    npoints = 1000
    lon = np.random.random_integers(-180000,180000,size=npoints)/1000.0
    lat = np.random.random_integers(-90000,90000,size=npoints)/1000.0
    data = np.random.rand(npoints)
    cloud = Cloud(lon,lat,data)
    print('Bounds of whole globe (basically)')
    print(cloud.getBounds())
    bounds = (-120.0,-70,20.0,55.0)
    cloud.trim(bounds)
    print('Bounds after trimming to %s' % str(bounds))
    print(cloud.getBounds())

if __name__ == '__main__':
    test()
