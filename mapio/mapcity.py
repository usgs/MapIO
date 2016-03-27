#stdlib imports
import os.path
import warnings

#local imports
from .city import Cities
from .dataset import DataSetException,DataSetWarning

#third party imports
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np

class MapCities(Cities):
    """
    A subclass of Cities that can remove cities whose labels on a map
    intersect with larger cities.

    """    
    def __init__(self,dataframe):
        """Construct a MapCities object from a pandas DataFrame.
        :param dataframe:
          pandas DataFrame,  where each row represents a city.
          Columns include:
           - name Name of the city (required).
           - lat Latitude of city (required).
           - lon Longitude of city (required).
           - pop Population of city (optional).
           - iscap Boolean indicating capital status (optional).
           - placement String indicating where city label 
                       should be placed relative to city coordinates, 
                       one of: E,W,N,S,NE,SE,SW,NW (optional).
           -xoff Longitude offset for label relative to city coordinates (optional).
           -yoff Latitude offset for label relative to city coordinates (optional).
        :raises DataSetException:
          When any of required columns are missing.
        :returns:
          MapCities instance.
        """
        if len(set(dataframe.columns).intersection(set(self.REQFIELDS))) < 3:
            raise DataSetException('Missing some of required keys: %s' % self.REQFIELDS)
        self._dataframe = dataframe.copy()
        tfontlist = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tfontlist = matplotlib.font_manager.get_fontconfig_fonts().keys()
        self._fontlist = []
        for tf in tfontlist:
            fontpath,fontfile = os.path.split(tf)
            font,fontext = os.path.splitext(fontfile)
            self._fontlist.append(font)
        
    def limitByMapCollision(self,fontname,fontsize,ax):
        """Create a smaller Cities dataset by removing smaller cities whose bounding
        boxes collide with larger cities.
        :param fontsize:
          Desired font size for city labels.
        :param fontname:
          Desired font name for city labels.
        :param ax:
          matplotlib.axes.Axes object.
        :raises DataSetException:
          When font name is not one of the supported Matplotlib font names OR
          when (if placement column exists) placement columns contain any values
          not in E,W,N,S,SE,SW,NE,N OR
          when cities have not been projected (projectCities() has not been called.)
        :returns:
          New Cities instance where smaller colliding cities have been removed.
        """
        
        if 'x' not in self._dataframe.columns and 'y' not in self._dataframe.columns:
            raise DataSetException('Cities object has not had projectCities() called yet.')
        
        if fontname not in self._fontlist:
            raise DataSetException('Font %s not in supported list.' % fontname)
        #TODO: - check placement column
        newdf = self._dataframe.copy()
        newdf = newdf.sort_values(by='pop',ascending=False)

        tops,bottoms,lefts,rights = self._getCityBoundingBoxes(newdf,fontname,fontsize,ax)
        ikeep = [0] #indices of rows to keep in dataframe
        for i in range(1,len(tops)):
            cname = newdf.iloc[i]['name']
            if cname.lower().find('vegas') > -1:
                foo = 1
            if np.isnan(lefts[i]):
                continue
            ileft_collisions = np.where((lefts[i] < rights[0:i-1]) & (lefts[i] > lefts[0:i-1]))[0]
            iright_collisions = np.where((rights[i] < rights[0:i-1]) & (rights[i] > lefts[0:i-1]))[0]
            ihor_collisions = np.union1d(ileft_collisions,iright_collisions)

            ibottom_collisions = np.where((bottoms[i] < tops[0:i-1]) & (bottoms[i] > bottoms[0:i-1]))[0]
            itop_collisions = np.where((tops[i] < tops[0:i-1]) & (tops[i] > bottoms[0:i-1]))[0]
            iver_collisions = np.union1d(ibottom_collisions,itop_collisions)

            icollide = np.intersect1d(ihor_collisions,iver_collisions)
            
            if not len(icollide):
                ikeep.append(i)
            else:
                foo = 1
                pass
        newdf = newdf.iloc[ikeep]
        return Cities(newdf)

    def _getCityBoundingBoxes(self,df,fontname,fontsize,ax):
        """Get the axes coordinate system bounding boxes for each city.
        :param df:
          DataFrame containing information about cities.
        :param fontname:
          fontname ('Arial','Helvetica', etc.)
        :param fontsize:
          Font size in points.
        :param ax:
          Axes object.
        :returns:
          Numpy arrays of top,bottom,left and right edges of city bounding boxes.
        """
        fig = ax.get_figure()
        fwidth,fheight = fig.get_figwidth(),fig.get_figheight()
        plt.sca(ax)
        axmin,axmax,aymin,aymax = plt.axis()
        axbox = ax.get_position().bounds
        newfig = plt.figure(figsize=(fwidth,fheight))
        newax = newfig.add_axes(axbox)
        newfig.canvas.draw()
        plt.sca(newax)
        plt.axis((axmin,axmax,aymin,aymax))
        #make arrays of the edges of all the bounding boxes
        tops = np.ones(len(df))*np.nan
        bottoms = np.ones(len(df))*np.nan
        lefts = np.ones(len(df))*np.nan
        rights = np.ones(len(df))*np.nan
        left,right,bottom,top = self._getCityEdges(df.iloc[0],newax,newfig,fontname,fontsize)
        lefts[0] = left
        rights[0] = right
        bottoms[0] = bottom
        tops[0] = top
        for i in range(1,len(df)):
            row = df.iloc[i]
            left,right,bottom,top = self._getCityEdges(row,newax,newfig,fontname,fontsize)
            #remove cities that have any portion off the map
            if left < axmin or right > axmax or bottom < aymin or top > aymax:
                foo = 1
                continue
            #print('%s: X - %.2f %.2f  Y - %.2f %.2f' % (row['name'],left,right,bottom,top))
            lefts[i] = left
            rights[i] = right
            bottoms[i] = bottom
            tops[i] = top
            
        #get rid of new figure and its axes
        plt.close(newfig)
        return (tops,bottoms,lefts,rights)

    def _getCityEdges(self,row,ax,fig,fontname,fontsize):
        """Return the edges of a city label on a given map axes.
        :param row:
          Row of a dataframe containing city information.
        :param ax:
          Axes instance.
        :param fig:
          Figure instance.
        :param fontname:
          Matplotlib compatible font name.
        :param fontsize:
          Font size in points.
        """
        ha = 'left'
        va = 'center'
        if 'placement' in row.index:
            if placement.find('E') > -1:
                ha = 'left'
            if placement.find('W') > -1:
                ha = 'right'
            else:
                ha = 'center'
            if placement.find('N') > -1:
                ha = 'top'
            if placement.find('S') > -1:
                ha = 'bottom'
            else:
                ha = 'center'
        th = plt.text(row['x'],row['y'],row['name'],fontname=fontname,
                      fontsize=fontsize,ha=ha,va=va)
        transf = ax.transData.inverted()
        bb = th.get_window_extent(renderer = fig.canvas.renderer)
        bbt = bb.transformed(transf)
        bbtc = bbt.corners()
        height = np.max(bbtc[:, 1])-np.min(bbtc[:, 1])
        width = np.max(bbtc[:, 0])-np.min(bbtc[:, 0])
        if va == 'top':
            tops = row['y']
            bottom = row['y']-height
        elif va == 'center':
            top = row['y'] + (height/2.0)
            bottom = row['y'] - (height/2.0)
        else:
            top = row['y'] + height
            bottom = row['y']
        if ha == 'left':
            left = row['x']
            right = row['x']+width
        elif ha == 'right':
            left = row['x'] - width
            right = row['x']
        else:
            left = row['x'] - (width/2.0)
            right = row['x'] + (width/2.0)
        return (left,right,bottom,top)

    def limitByBounds(self,bounds):
        """Search for cities within a bounding box (xmin,xmax,ymin,ymax).
        :param bounds: 
          Sequence containing xmin,xmax,ymin,ymax (decimal degrees).
        :returns:
          New Cities instance containing smaller cities data set.
        """
        #I'm implementing this method here because I can't figure out
        #another way to call the parent class method and get back a MapCities instance
        #instead of a Cities instance.
        parent = super(MapCities, self).limitByBounds(bounds)
        return MapCities(parent._dataframe)

    def limitByRadius(self,lat,lon,radius):
        """Search for cities within a radius (km) around a central point.
        :param lat:
          Central latitude coordinate (dd).
        :param lon:
          Central longitude coordinate (dd).
        :param radius:
          Radius (km) around which cities will be searched.
        :returns:
          New Cities instance containing smaller cities data set.
        """
        parent = super(MapCities, self).limitByRadius(lat,lon,radius)
        return MapCities(parent._dataframe)

    def limitByPopulation(self,pop,minpop=0):
        """Search for cities above a certain population threshold.
        :param pop:
          Population threshold.
        :param minpop:
          Population above which cities should be included.
        :raises DataSetException:
          When Cities instance does not contain population data.
          When minpop >= pop.
        :returns:
          New Cities instance containing cities where population > pop.
        """
        parent = super(MapCities, self).limitByPopulation(pop,minpop=minpop)
        return MapCities(parent._dataframe)

    def limitByGrid(self,nx=2,ny=2,cities_per_grid=20):
        """Create a smaller Cities dataset by gridding cities, then limiting cities in
        each grid by population.
        :param nx:
          Desired number of columns for grid.
        :param ny:
          Desired number of rows for grid.
        :param cities_per_cell:
          Maximum number of cities allowed per grid cell.
        :raises DataSetException:
          When Cities instance does not contain population data.
        :returns:
          New Cities instance containing cities limited by number in each grid cell.
        """
        parent = super(MapCities, self).limitByGrid(nx=nx,ny=ny,cities_per_grid=cities_per_grid)
        return MapCities(parent._dataframe)

    def limitByName(self,cityname):
        """Find all cities that match a given cityname (or regular expression).
        :param cityname:
          Input city name (i.e., "Los Angeles").
        :returns:
          Cities instance containing cities with names that match the input name/regular expression.
        """
        parent = super(MapCities, self).limitByName(cityname)
        return MapCities(parent._dataframe)

    
