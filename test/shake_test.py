#!/usr/bin/env python

# python 3 compatibility
from __future__ import print_function

# stdlib imports
from datetime import datetime
from collections import OrderedDict
import tempfile
import time
import shutil
import os.path

# third party
from mapio.shake import ShakeGrid
from mapio.geodict import GeoDict
import numpy as np

homedir = os.path.dirname(os.path.abspath(__file__))  # where is this script?
mapiodir = os.path.abspath(os.path.join(homedir, '..'))


def test_modify():
    print('Testing ShakeGrid interpolate() method...')
    geodict = GeoDict({'xmin': 0.5, 'xmax': 6.5, 'ymin': 1.5,
                       'ymax': 6.5, 'dx': 1.0, 'dy': 1.0, 'ny': 6, 'nx': 7})
    data = np.arange(14, 56).reshape(6, 7)
    layers = OrderedDict()
    layers['pga'] = data
    shakeDict = {'event_id': 'usabcd1234',
                 'shakemap_id': 'usabcd1234',
                 'shakemap_version': 1,
                 'code_version': '4.0',
                 'process_timestamp': datetime.utcnow(),
                 'shakemap_originator': 'us',
                 'map_status': 'RELEASED',
                 'shakemap_event_type': 'ACTUAL'}
    eventDict = {'event_id': 'usabcd1234',
                 'magnitude': 7.6,
                 'depth': 1.4,
                 'lat': 2.0,
                 'lon': 2.0,
                 'event_timestamp': datetime.utcnow(),
                 'event_network': 'us',
                 'event_description': 'sample event'}
    uncDict = {'pga': (0.0, 0)}
    shake = ShakeGrid(layers, geodict, eventDict, shakeDict, uncDict)
    rdata = np.random.rand(data.shape[0], data.shape[1])
    shake.setLayer('pga', rdata)
    newdata = shake.getLayer('pga').getData()
    np.testing.assert_almost_equal(rdata, newdata)


def test_interpolate():
    print('Testing ShakeGrid interpolate() method...')
    geodict = GeoDict({'xmin': 0.5, 'xmax': 6.5, 'ymin': 1.5,
                       'ymax': 6.5, 'dx': 1.0, 'dy': 1.0, 'ny': 6, 'nx': 7})
    data = np.arange(14, 56).reshape(6, 7)
    layers = OrderedDict()
    layers['pga'] = data
    shakeDict = {'event_id': 'usabcd1234',
                 'shakemap_id': 'usabcd1234',
                 'shakemap_version': 1,
                 'code_version': '4.0',
                 'process_timestamp': datetime.utcnow(),
                 'shakemap_originator': 'us',
                 'map_status': 'RELEASED',
                 'shakemap_event_type': 'ACTUAL'}
    eventDict = {'event_id': 'usabcd1234',
                 'magnitude': 7.6,
                 'depth': 1.4,
                 'lat': 2.0,
                 'lon': 2.0,
                 'event_timestamp': datetime.utcnow(),
                 'event_network': 'us',
                 'event_description': 'sample event'}
    uncDict = {'pga': (0.0, 0)}
    shake = ShakeGrid(layers, geodict, eventDict, shakeDict, uncDict)
    sampledict = GeoDict({'xmin': 3.0, 'xmax': 4.0,
                          'ymin': 3.0, 'ymax': 4.0,
                          'dx': 1.0, 'dy': 1.0,
                          'ny': 2, 'nx': 2})
    shake2 = shake.interpolateToGrid(sampledict, method='linear')
    output = np.array([[34., 35.], [41., 42.]])
    np.testing.assert_almost_equal(output, shake2.getLayer('pga').getData())
    print('Passed test of ShakeGrid interpolate() method.')


def test_read():
    xmlfile = os.path.join(homedir, 'data', 'northridge.xml')
    tdir = tempfile.mkdtemp()
    testfile = os.path.join(tdir, 'test.xml')
    try:
        shakegrid = ShakeGrid.load(xmlfile, adjust='res')
        t1 = time.time()
        shakegrid.save(testfile)
        t2 = time.time()
        print('Saving shakemap took %.2f seconds' % (t2 - t1))
    except Exception as error:
        print('Failed to read grid.xml format file "%s". Error "%s".' %
              (xmlfile, str(error)))
        assert 0 == 1
    finally:
        if os.path.isdir(tdir):
            shutil.rmtree(tdir)


def test_save():
    tdir = tempfile.mkdtemp()
    testfile = os.path.join(tdir, 'test.xml')
    try:
        print('Testing save/read functionality for shakemap grids...')
        pga = np.arange(0, 16, dtype=np.float32).reshape(4, 4)
        pgv = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        mmi = np.arange(2, 18, dtype=np.float32).reshape(4, 4)
        geodict = GeoDict({'xmin': 0.5, 'xmax': 3.5,
                           'ymin': 0.5, 'ymax': 3.5,
                           'dx': 1.0, 'dy': 1.0,
                           'ny': 4, 'nx': 4})
        layers = OrderedDict()
        layers['pga'] = pga
        layers['pgv'] = pgv
        layers['mmi'] = mmi
        shakeDict = {'event_id': 'usabcd1234',
                     'shakemap_id': 'usabcd1234',
                     'shakemap_version': 1,
                     'code_version': '4.0',
                     'process_timestamp': datetime.utcnow(),
                     'shakemap_originator': 'us',
                     'map_status': 'RELEASED',
                     'shakemap_event_type': 'ACTUAL'}
        eventDict = {'event_id': 'usabcd1234',
                     'magnitude': 7.6,
                     'depth': 1.4,
                     'lat': 2.0,
                     'lon': 2.0,
                     'event_timestamp': datetime.utcnow(),
                     'event_network': 'us',
                     # line below tests escaping XML
                     'event_description': 'sample event & stuff'}
        uncDict = {'pga': (0.0, 0),
                   'pgv': (0.0, 0),
                   'mmi': (0.0, 0)}
        shake = ShakeGrid(layers, geodict, eventDict, shakeDict, uncDict)

        print('Testing save/read functionality...')
        shake.save(testfile, version=3)

        with open(testfile, 'rt') as f:
            data = f.read()
            assert data.find('&amp; stuff') > -1

        shake2 = ShakeGrid.load(testfile)
        for layer in ['pga', 'pgv', 'mmi']:
            tdata = shake2.getLayer(layer).getData()
            np.testing.assert_almost_equal(tdata, layers[layer])

        # check to see that our ampersand is back to an ampersand
        event = shake2.getEventDict()
        assert event['event_description'].find('& stuff') > -1

        print('Passed save/read functionality for shakemap grids.')

        print('Testing getFileGeoDict method...')
        _ = ShakeGrid.getFileGeoDict(testfile)
        print('Passed save/read functionality for shakemap grids.')

        print('Testing loading with bounds (no resampling or padding)...')
        sampledict = GeoDict({'xmin': -0.5, 'xmax': 3.5,
                              'ymin': -0.5, 'ymax': 3.5,
                              'dx': 1.0, 'dy': 1.0,
                              'ny': 5, 'nx': 5})
        shake3 = ShakeGrid.load(testfile, samplegeodict=sampledict,
                                resample=False, doPadding=False,
                                padValue=np.nan)
        tdata = shake3.getLayer('pga').getData()
        np.testing.assert_almost_equal(tdata, layers['pga'])

        print('Passed loading with bounds (no resampling or padding)...')

        print('Testing loading shakemap with padding, no resampling...')
        newdict = GeoDict({'xmin': -0.5, 'xmax': 4.5,
                           'ymin': -0.5, 'ymax': 4.5,
                           'dx': 1.0, 'dy': 1.0,
                           'ny': 6, 'nx': 6})
        shake4 = ShakeGrid.load(testfile, samplegeodict=newdict,
                                resample=False, doPadding=True,
                                padValue=np.nan)
        output = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                           [np.nan, 0.0, 1.0, 2.0, 3.0, np.nan],
                           [np.nan, 4.0, 5.0, 6.0, 7.0, np.nan],
                           [np.nan, 8.0, 9.0, 10.0, 11.0, np.nan],
                           [np.nan, 12.0, 13.0, 14.0, 15.0, np.nan],
                           [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
        tdata = shake4.getLayer('pga').getData()
        np.testing.assert_almost_equal(tdata, output)
        print('Passed loading shakemap with padding, no resampling...')

        # make a bigger grid
        pga = np.arange(0, 36, dtype=np.float32).reshape(6, 6)
        pgv = np.arange(1, 37, dtype=np.float32).reshape(6, 6)
        mmi = np.arange(2, 38, dtype=np.float32).reshape(6, 6)
        layers = OrderedDict()
        layers['pga'] = pga
        layers['pgv'] = pgv
        layers['mmi'] = mmi
        geodict = GeoDict({'xmin': 0.5, 'xmax': 5.5,
                           'ymin': 0.5, 'ymax': 5.5,
                           'dx': 1.0, 'dy': 1.0,
                           'ny': 6, 'nx': 6})
        shake = ShakeGrid(layers, geodict, eventDict, shakeDict, uncDict)
        shake.save(testfile, version=3)

        print('Testing resampling, no padding...')
        littledict = GeoDict({'xmin': 2.0, 'xmax': 4.0,
                              'ymin': 2.0, 'ymax': 4.0,
                              'dx': 1.0, 'dy': 1.0,
                              'ny': 3, 'nx': 3})
        shake5 = ShakeGrid.load(testfile, samplegeodict=littledict,
                                resample=True, doPadding=False,
                                padValue=np.nan)
        output = np.array([[10.5, 11.5, 12.5],
                           [16.5, 17.5, 18.5],
                           [22.5, 23.5, 24.5]])
        tdata = shake5.getLayer('pga').getData()
        np.testing.assert_almost_equal(tdata, output)
        print('Passed resampling, no padding...')

        print('Testing resampling and padding...')
        pga = np.arange(0, 16, dtype=np.float32).reshape(4, 4)
        pgv = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        mmi = np.arange(2, 18, dtype=np.float32).reshape(4, 4)
        geodict = GeoDict({'xmin': 0.5, 'ymax': 3.5,
                           'ymin': 0.5, 'xmax': 3.5,
                           'dx': 1.0, 'dy': 1.0,
                           'ny': 4, 'nx': 4})
        layers = OrderedDict()
        layers['pga'] = pga
        layers['pgv'] = pgv
        layers['mmi'] = mmi
        shake = ShakeGrid(layers, geodict, eventDict, shakeDict, uncDict)
        shake.save(testfile, version=3)
        bigdict = GeoDict({'xmin': 0.0, 'xmax': 4.0,
                           'ymin': 0.0, 'ymax': 4.0,
                           'dx': 1.0, 'dy': 1.0,
                           'ny': 5, 'nx': 5})
        shake6 = ShakeGrid.load(
            testfile, samplegeodict=bigdict, resample=True, doPadding=True,
            padValue=np.nan)
        tdata = shake6.getLayer('pga').getData()
        output = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan],
                           [np.nan, 2.5, 3.5, 4.5, np.nan],
                           [np.nan, 6.5, 7.5, 8.5, np.nan],
                           [np.nan, 10.5, 11.5, 12.5, np.nan],
                           [np.nan, np.nan, np.nan, np.nan, np.nan]])
        np.testing.assert_almost_equal(tdata, output)
        print('Passed resampling and padding...')
    except Exception as error:
        print('Failed to read grid.xml format file "%s". Error "%s".' %
              (testfile, str(error)))
        assert 0 == 1
    finally:
        if os.path.isdir(tdir):
            shutil.rmtree(tdir)


def test_meridian():
    shakeDict = {'event_id': 'usabcd1234',
                 'shakemap_id': 'usabcd1234',
                 'shakemap_version': 1,
                 'code_version': '4.0',
                 'process_timestamp': datetime.utcnow(),
                 'shakemap_originator': 'us',
                 'map_status': 'RELEASED',
                 'shakemap_event_type': 'ACTUAL'}
    eventDict = {'event_id': 'usabcd1234',
                 'magnitude': 7.6,
                 'depth': 1.4,
                 'lat': 2.0,
                 'lon': 2.0,
                 'event_timestamp': datetime.utcnow(),
                 'event_network': 'us',
                 # line below tests escaping XML
                 'event_description': 'sample event & stuff'}
    uncDict = {'pga': (0.0, 0),
               'pgv': (0.0, 0),
               'mmi': (0.0, 0)}
    mdict = {'digits': 4,
             'dx': 0.033333333333333333,
             'dy': 0.033333333333333333,
             'nx': 442,
             'ny': 319,
             'units': 'ln(g)',
             'xmax': 180.34999999999999,
             'xmin': 165.65000000000001,
             'ymax': -36.649999999999999,
             'ymin': -47.25}
    gdict = GeoDict(mdict)
    pga = np.ones((gdict.ny, gdict.nx))
    pgv = np.ones((gdict.ny, gdict.nx))
    mmi = np.ones((gdict.ny, gdict.nx))
    layers = OrderedDict()
    layers['pga'] = pga
    layers['pgv'] = pgv
    layers['mmi'] = mmi
    shake = ShakeGrid(layers, gdict, eventDict, shakeDict, uncDict)
    tdir = tempfile.mkdtemp()
    try:
        fname = os.path.join(tdir, 'temp.xml')
        shake.save(fname)
        x = 1
    except Exception:
        assert False
    finally:
        shutil.rmtree(tdir)


if __name__ == '__main__':
    test_meridian()
    test_modify()
    test_interpolate()
    test_read()
    test_save()
