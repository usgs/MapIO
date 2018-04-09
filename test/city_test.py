#!/usr/bin/env python

import tempfile
import shutil
import os.path
from mapio.city import Cities

def test_city():
    homedir = os.path.dirname(os.path.abspath(__file__)) #where is this script?
    cityfile = os.path.join(homedir,'data','cities1000.txt')
    cities = Cities.loadFromGeoNames(cityfile=cityfile)
    assert len(cities) == 145315
    print(cities)

    cities.sortByColumns(['pop'],ascending=False)
    assert cities.getDataFrame().iloc[0]['name'] == 'Shanghai'
        
    columns = cities.getColumns()
    assert columns == ['ccode', 'iscap', 'lat', 'lon', 'name', 'pop']
    
    tmpdir = tempfile.mkdtemp()
    try:
        csvfile = os.path.join(tmpdir,'cities.csv')
        cities.save(csvfile)
        cities2 = Cities.loadFromCSV(csvfile)
        assert len(cities) == len(cities2)

        
        
    except Exception as e:
        assert 1==2
    finally:
        shutil.rmtree(tmpdir)
    
if __name__ == '__main__':
    test_city()
