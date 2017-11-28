from distutils.core import setup
import mapio

myversion = mapio.__version__

setup(name='mapio',
      version=myversion,
      description='Grid Classes',
      author='Mike Hearne,Bruce Worden',
      author_email='mhearne@usgs.gov,cbworden@usgs.gov',
      url='',
      packages=['mapio', 'mapio.extern'],
      scripts = [],
)
