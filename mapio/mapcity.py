# local imports
from .city import Cities
from .dataset import DataSetException

# third party imports
import matplotlib.font_manager


class MapCities(Cities):
    """
    A subclass of Cities that can remove cities whose labels on a map
    intersect with larger cities.

    """

    def __init__(self, dataframe):
        """Construct a MapCities object from a pandas DataFrame.

        Args:
            dataframe (dataframe):
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
                        -xoff Longitude offset for label relative to city coordinates
                        (optional).
                        -yoff Latitude offset for label relative to city coordinates
                        (optional).
        Raises:
            DataSetException:
                When any of required columns are missing.
        Returns:
            MapCities instance.
        """
        self.SUGGESTED_FONTS = [
            "Bitstream Vera Sans",
            "Times New Roman",
            "Courier New",
            "Palatino LinoType",
            "Arial",
            "Tahoma",
        ]
        self.DEFAULT_FONT = "Times New Roman"
        self.DEFAULT_FONT_SIZE = 10.0
        if len(set(dataframe.columns).intersection(set(self.REQFIELDS))) < 3:
            raise DataSetException("Missing some of required keys: %s" % self.REQFIELDS)
        self._dataframe = dataframe.copy()

        self._fontlist = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
        self._fontlist.sort()

    def limitByMapCollision(self):
        """Create a smaller Cities dataset by removing smaller cities whose bounding
        boxes collide with larger cities.

        Returns:
            New Cities instance where smaller colliding cities have been removed.
        """
        raise NotImplementedError

    def getFontList(self):
        """Return list of supported font names on this system.

        Returns:
            List of supported font names on this system.
        """
        return self._fontlist
