#stdlib imports
import os.path
import warnings

#local imports
from .mapcity import MapCities
from .dataset import DataSetException,DataSetWarning

#third party imports
import matplotlib.font_manager
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

XOFFSET = 4 #how many pixels between the city dot and the city text

class CartopyCities(MapCities):
    """
    A subclass of Cities that can remove cities whose labels on a map
    intersect with larger cities.

    """    
       
    def limitByMapCollision(self,ax,fontname='Bitstream Vera Sans',fontsize=10.0,shadow=False):
        """Create a smaller Cities dataset by removing smaller cities whose bounding
        boxes collide with larger cities.


        #######################################
        
        #######################################
        
        :param fontsize:
          Desired font size for city labels.
        :param fontname:
          Desired font name for city labels.
        :param ax:
          Cartopy GeoAxes object.
        :raises DataSetException:
          When font name is not one of the supported Matplotlib font names OR
          when (if placement column exists) placement columns contain any values
          not in E,W,N,S,SE,SW,NE,N OR
          when cities have not been projected (project() has not been called.)
        :returns:
          New Cities instance where smaller colliding cities have been removed.
        """
        #get the transformation from display units (pixels) to data units
        #we'll use this to set an offset in pixels between the city dot and
        #the city name.
        self._display_to_data_transform = ax.transData.inverted()

        if fontname not in self._fontlist:
            raise DataSetException('Font %s not in supported list.' % fontname)
        #TODO: - check placement column
        newdf = self._dataframe.copy()
        #older versions of pandas use a different sort function
        if pd.__version__ < '0.17.0':
            newdf = newdf.sort(columns='pop',ascending=False)
        else:
            newdf = newdf.sort_values(by='pop',ascending=False)

        lefts,rights,bottoms,tops = self._getCityBoundingBoxes(newdf,fontname,fontsize,ax,shadow=shadow)
        ikeep = ~np.isnan(lefts)
        newdf = newdf.iloc[ikeep]
        lefts = lefts[ikeep]
        rights = rights[ikeep]
        bottoms = bottoms[ikeep]
        tops = tops[ikeep]
        ikeep = [0] #indices of rows to keep in dataframe
        for i in range(1,len(tops)):
            cname = newdf.iloc[i]['name']
            if cname.lower().find('pacific grove') > -1:
                allnames = newdf['name'].tolist()
                sidx = allnames.index('Salinas')
                sleft = lefts[sidx]
                sright = rights[sidx]
                sbottom = bottoms[sidx]
                stop = tops[sidx]
                foo = 1
            if np.isnan(lefts[i]):
                continue
            left = lefts[i]
            right = rights[i]
            bottom = bottoms[i]
            top = tops[i]

            clrx = (left > rights[0:i]) | (right < lefts[0:i])
            clry = (top < bottoms[0:i]) | (bottom > tops[0:i])
            allclr = (clrx | clry)
            if all(allclr):
                ikeep.append(i)
            else:
                foo = 1

        newdf = newdf.iloc[ikeep]
        newdf['top'] = tops[ikeep]
        newdf['bottom'] = bottoms[ikeep]
        newdf['left'] = lefts[ikeep]
        newdf['right'] = rights[ikeep]
        return type(self)(newdf)

    def renderToMap(self,ax,fontname='Bitstream Vera Sans',fontsize=10.0,zorder=10,shadow=False):
        """Render cities on Cartopy axes.

        :param ax:
          Matplotlib Axes instance.
        :param fontname:
          String name of font.
        :param fontsize:
          Font size in points.
        :param zorder:
          Matplotlib plotting order - higher zorder is on top. 
        :param shadow:
          Boolean indicating whether "drop-shadow" effect should be used.
        """
        #get the transformation from display units (pixels) to data units
        #we'll use this to set an offset in pixels between the city dot and
        #the city name.
        self._display_to_data_transform = ax.transData.inverted()
        
        for index,row in self._dataframe.iterrows():
            th = self._renderRow(row,ax,fontname,fontsize,shadow=shadow)
            ax.plot(row['lon'],row['lat'],'k.')
    
    def _getCityBoundingBoxes(self,df,fontname,fontsize,ax,shadow=False):
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
        left,right,bottom,top = self._getCityEdges(df.iloc[0],newax,newfig,fontname,fontsize,shadow=shadow)
        lefts[0] = left
        rights[0] = right
        bottoms[0] = bottom
        tops[0] = top
        for i in range(1,len(df)):
            row = df.iloc[i]
            left,right,bottom,top = self._getCityEdges(row,newax,newfig,fontname,fontsize)
            #remove cities that have any portion off the map
            if left < axmin or right > axmax or bottom < aymin or top > aymax:
                continue
            lefts[i] = left
            rights[i] = right
            bottoms[i] = bottom
            tops[i] = top

        #get rid of new figure and its axes
        plt.close(newfig)
        return (lefts,rights,bottoms,tops)

    def _renderRow(self,row,ax,fontname,fontsize,zorder=10,shadow=False):
        """Internal method to consistently render city names.
        :param row:
          pandas dataframe row.
        :param ax:
          Matplotlib Axes instance.
        :param fontname:
          String name of desired font.
        :param fontsize:
          Font size in points.
        :param zorder:
          Matplotlib plotting order - higher zorder is on top. 
        :param shadow:
          Boolean indicating whether "drop-shadow" effect should be used.
        :returns:
          Matplotlib Text instance.
        """
        ha = 'left'
        va = 'center'
        if 'placement' in row.index:
            if row['placement'].find('E') > -1:
                ha = 'left'
            if row['placement'].find('W') > -1:
                ha = 'right'
            else:
                ha = 'center'
            if row['placement'].find('N') > -1:
                ha = 'top'
            if row['placement'].find('S') > -1:
                ha = 'bottom'
            else:
                ha = 'center'
        
        display1 = (1,1)
        display2 = (1+XOFFSET,1)
        data1 = self._display_to_data_transform.transform((display1))
        data2 = self._display_to_data_transform.transform((display2))
        data_x_offset = data2[0] - data1[0]
        tx = row['lon'] + data_x_offset
        ty = row['lat']
        if shadow:  
            th = ax.text(tx,ty,row['name'],fontname=fontname,color='black',
                         fontsize=fontsize,ha=ha,va=va,zorder=zorder)
            th.set_path_effects([path_effects.Stroke(linewidth=2.0, foreground='white'),
                                 path_effects.Normal()])
        else:     
            th = ax.text(tx,ty,row['name'],fontname=fontname,
                         fontsize=fontsize,ha=ha,va=va,zorder=zorder)
            
        return th
        
    
    def _getCityEdges(self,row,ax,fig,fontname,fontsize,shadow=False):
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
        th = self._renderRow(row,ax,fontname,fontsize,shadow=shadow)
        bbox = th.get_window_extent(fig.canvas.renderer)
        axbox = bbox.inverse_transformed(ax.transData)
        left,bottom,right,top = axbox.extents
        return (left,right,bottom,top)


    
