#!/usr/bin/env python

#python 3 compatibility
from __future__ import print_function

#stdlib imports
from xml.dom import minidom
from datetime import datetime
from collections import OrderedDict
import re
import sys
if sys.version_info.major == 2:
    import StringIO
else:
    from io import StringIO
import os.path

#third party

#hack the path so that I can debug these functions if I need to
homedir = os.path.dirname(os.path.abspath(__file__)) #where is this script?
mapiodir = os.path.abspath(os.path.join(homedir,'..'))
sys.path.insert(0,mapiodir) #put this at the front of the system path, ignoring any installed mapio stuff

from mapio.shake import ShakeGrid
from mapio.gridbase import Grid
from mapio.multiple import MultiGrid
from mapio.dataset import DataSetException
from mapio.grid2d import Grid2D
from mapio.geodict import GeoDict
import numpy as np
        
def test_trim(shakefile):
    geodict = getShakeDict(shakefile)
    #bring in the shakemap by a half dimension (quarter on each side)
    lonrange = geodict['xmax'] - geodict['xmin']
    latrange = geodict['ymax'] - geodict['ymin']
    newxmin = geodict['xmin'] + lonrange/4.0
    newxmax = geodict['xmax'] - lonrange/4.0
    newymin = geodict['ymin'] + latrange/4.0
    newymax = geodict['ymax'] - latrange/4.0
    newbounds = (newxmin,newxmax,newymin,newymax)
    grid = ShakeGrid.load(shakefile)
    grid.trim(newbounds)

def test_read(xmlfile):
    try:
        shakegrid = ShakeGrid.load(xmlfile)
    except Exception as error:
        print('Failed to read grid.xml format file "%s". Error "%s".' % (xmlfile,str(error)))
    
def test_save():
    try:
        print('Testing save/read functionality for shakemap grids...')
        pga = np.arange(0,16,dtype=np.float32).reshape(4,4)
        pgv = np.arange(1,17,dtype=np.float32).reshape(4,4)
        mmi = np.arange(2,18,dtype=np.float32).reshape(4,4)
        geodict = GeoDict({'xmin':0.5,'xmax':3.5,
                           'ymin':0.5,'ymax':3.5,
                           'dx':1.0,'dy':1.0,
                           'ny':4,'nx':4})
        layers = OrderedDict()
        layers['pga'] = pga
        layers['pgv'] = pgv
        layers['mmi'] = mmi
        shakeDict = {'event_id':'usabcd1234',
                     'shakemap_id':'usabcd1234',
                     'shakemap_version':1,
                     'code_version':'4.0',
                     'process_timestamp':datetime.utcnow(),
                     'shakemap_originator':'us',
                     'map_status':'RELEASED',
                     'shakemap_event_type':'ACTUAL'}
        eventDict = {'event_id':'usabcd1234',
                     'magnitude':7.6,
                     'depth':1.4,
                     'lat':2.0,
                     'lon':2.0,
                     'event_timestamp':datetime.utcnow(),
                     'event_network':'us',
                     'event_description':'sample event'}
        uncDict = {'pga':(0.0,0),
                   'pgv':(0.0,0),
                   'mmi':(0.0,0)}
        shake = ShakeGrid(layers,geodict,eventDict,shakeDict,uncDict)
        
        print('Testing save/read functionality...')
        shake.save('test.xml',version=3)
        shake2 = ShakeGrid.load('test.xml')
        for layer in ['pga','pgv','mmi']:
            tdata = shake2.getLayer(layer).getData()
            np.testing.assert_almost_equal(tdata,layers[layer])

        print('Passed save/read functionality for shakemap grids.')

        print('Testing getFileGeoDict method...')
        fgeodict = ShakeGrid.getFileGeoDict('test.xml')
        print('Passed save/read functionality for shakemap grids.')
        
        print('Testing loading with bounds (no resampling or padding)...')
        sampledict = GeoDict({'xmin':-0.5,'xmax':3.5,
                              'ymin':-0.5,'ymax':3.5,
                              'dx':1.0,'dy':1.0,
                              'ny':5,'nx':5})
        shake3 = ShakeGrid.load('test.xml',samplegeodict=sampledict,resample=False,doPadding=False,padValue=np.nan)
        tdata = shake3.getLayer('pga').getData()
        np.testing.assert_almost_equal(tdata,layers['pga'])

        print('Passed loading with bounds (no resampling or padding)...')

        print('Testing loading shakemap with padding, no resampling...')
        newdict = GeoDict({'xmin':-0.5,'xmax':4.5,
                           'ymin':-0.5,'ymax':4.5,
                           'dx':1.0,'dy':1.0,
                           'ny':6,'nx':6})
        shake4 = ShakeGrid.load('test.xml',samplegeodict=newdict,resample=False,doPadding=True,padValue=np.nan)
        output = np.array([[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                           [np.nan,0.0,1.0,2.0,3.0,np.nan],
                           [np.nan,4.0,5.0,6.0,7.0,np.nan],
                           [np.nan,8.0,9.0,10.0,11.0,np.nan],
                           [np.nan,12.0,13.0,14.0,15.0,np.nan],
                           [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]])
        tdata = shake4.getLayer('pga').getData()
        np.testing.assert_almost_equal(tdata,output)
        print('Passed loading shakemap with padding, no resampling...')

        #make a bigger grid
        pga = np.arange(0,36,dtype=np.float32).reshape(6,6)
        pgv = np.arange(1,37,dtype=np.float32).reshape(6,6)
        mmi = np.arange(2,38,dtype=np.float32).reshape(6,6)
        layers = OrderedDict()
        layers['pga'] = pga
        layers['pgv'] = pgv
        layers['mmi'] = mmi
        geodict = GeoDict({'xmin':0.5,'xmax':5.5,
                           'ymin':0.5,'ymax':5.5,
                           'dx':1.0,'dy':1.0,
                           'ny':6,'nx':6})
        shake = ShakeGrid(layers,geodict,eventDict,shakeDict,uncDict)
        shake.save('test.xml',version=3)

        print('Testing resampling, no padding...')
        littledict = GeoDict({'xmin':2.0,'xmax':4.0,
                              'ymin':2.0,'ymax':4.0,
                              'dx':1.0,'dy':1.0,
                              'ny':3,'nx':3})
        shake5 = ShakeGrid.load('test.xml',samplegeodict=littledict,resample=True,doPadding=False,padValue=np.nan)
        output = np.array([[10.5,11.5,12.5],
                           [16.5,17.5,18.5],
                           [22.5,23.5,24.5]])
        tdata = shake5.getLayer('pga').getData()
        np.testing.assert_almost_equal(tdata,output)
        print('Passed resampling, no padding...')

        print('Testing resampling and padding...')
        pga = np.arange(0,16,dtype=np.float32).reshape(4,4)
        pgv = np.arange(1,17,dtype=np.float32).reshape(4,4)
        mmi = np.arange(2,18,dtype=np.float32).reshape(4,4)
        geodict = GeoDict({'xmin':0.5,'ymax':3.5,
                           'ymin':0.5,'xmax':3.5,
                           'dx':1.0,'dy':1.0,
                           'ny':4,'nx':4})
        layers = OrderedDict()
        layers['pga'] = pga
        layers['pgv'] = pgv
        layers['mmi'] = mmi
        shake = ShakeGrid(layers,geodict,eventDict,shakeDict,uncDict)
        shake.save('test.xml',version=3)
        bigdict = GeoDict({'xmin':0.0,'xmax':4.0,
                           'ymin':0.0,'ymax':4.0,
                           'dx':1.0,'dy':1.0,
                           'ny':5,'nx':5})
        shake6 = ShakeGrid.load('test.xml',samplegeodict=bigdict,resample=True,doPadding=True,padValue=np.nan)
        tdata = shake6.getLayer('pga').getData()
        output = np.array([[np.nan,np.nan,np.nan,np.nan,np.nan],
                           [np.nan,2.5,3.5,4.5,np.nan],
                           [np.nan,6.5,7.5,8.5,np.nan],
                           [np.nan,10.5,11.5,12.5,np.nan],
                           [np.nan,np.nan,np.nan,np.nan,np.nan]])
        np.testing.assert_almost_equal(tdata,output)
        print('Passed resampling and padding...')
    except AssertionError as error:
        print('Failed a shakemap load test:\n %s' % error)
    os.remove('test.xml')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        shakefile = sys.argv[1]
        test_read(shakefile)
        test_trim(shakefile)
    test_save()
    
