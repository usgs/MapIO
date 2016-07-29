#!/usr/bin/env python

#python 3 compatibility
from __future__ import print_function

#stdlib imports
import sys
import collections
import datetime
import time
import os.path
import shutil
import tempfile

#third party imports
import h5py
from scipy.io import netcdf 
import numpy as np

#hack the path so that I can debug these functions if I need to
homedir = os.path.dirname(os.path.abspath(__file__)) #where is this script?
mapiodir = os.path.abspath(os.path.join(homedir,'..'))
sys.path.insert(0,mapiodir) #put this at the front of the system path, ignoring any installed mapio stuff

from mapio.multihaz import MultiHazardGrid
from mapio.multiple import MultiGrid
from mapio.shake import ShakeGrid
from mapio.grid2d import Grid2D
from mapio.dataset import DataSetException

def test():
    shakefile = os.path.join(homedir,'data','northridge.xml')
    t1 = datetime.datetime.now()
    sgrid = ShakeGrid.load(shakefile,adjust='res')
    t2 = datetime.datetime.now()
    origin = {}
    origin['id'] = sgrid._eventDict['event_id']
    origin['source'] = sgrid._eventDict['event_network']
    origin['time'] = sgrid._eventDict['event_timestamp']
    origin['lat'] = sgrid._eventDict['lat']
    origin['lon'] = sgrid._eventDict['lon']
    origin['depth'] = sgrid._eventDict['depth']
    origin['magnitude'] = sgrid._eventDict['magnitude']

    header = {}
    header['type'] = 'shakemap'
    header['version'] = sgrid._shakeDict['shakemap_version']
    header['process_time'] = sgrid._shakeDict['process_timestamp']
    header['code_version'] = sgrid._shakeDict['code_version']
    header['originator'] = sgrid._shakeDict['shakemap_originator']
    header['product_id'] = sgrid._shakeDict['shakemap_id']
    header['map_status'] = sgrid._shakeDict['map_status']
    header['event_type'] = sgrid._shakeDict['shakemap_event_type']

    layers = collections.OrderedDict()
    for (layername,layerdata) in sgrid.getData().items():
        layers[layername] = layerdata.getData()

    tdict = {'name':'fred','family':{'wife':'wilma','daughter':'pebbles'}}
    mgrid = MultiHazardGrid(layers,sgrid.getGeoDict(),origin,header,metadata={'flintstones':tdict})
    tdir = tempfile.mkdtemp()
    testfile = os.path.join(tdir,'test.hdf')
    try:
        mgrid.save(testfile)
        t3 = datetime.datetime.now()
        mgrid2 = MultiHazardGrid.load(testfile)
        t4 = datetime.datetime.now()
        xmlmb = os.path.getsize(shakefile)/float(1e6)
        hdfmb = os.path.getsize(testfile)/float(1e6)
        xmltime = (t2-t1).seconds + (t2-t1).microseconds/float(1e6)
        hdftime = (t4-t3).seconds + (t4-t3).microseconds/float(1e6)
        print('Input XML file size: %.2f MB (loading time %.3f seconds)' % (xmlmb,xmltime))
        print('Output HDF file size: %.2f MB (loading time %.3f seconds)' % (hdfmb,hdftime))
    except DataSetException as obj:
        pass
    finally:
        if os.path.isdir(tdir):
            shutil.rmtree(tdir)

if __name__ == '__main__':
    test()
    
