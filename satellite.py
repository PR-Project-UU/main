import ee

ee.Initialize()

from geopy import distance, Point
from math import floor
from logging import getLogger
from time import perf_counter
from typing import List, Tuple

from util import cloudmask_L457, spiralIndex

class Capture:
    collection = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR') # Stores the base collection for satellite data
    generated = []                                            # Stores the generated image names
    index = 0                                                 # Stores the index for data generation
    interval: Tuple[int, int] = (None, None)                  # Stores the time interval from start year to end year for image capturing
    origin: Point = (None, None)                              # Stores the center of the capture area as a geopy point
    log = getLogger('capture')                                # Stores the logger for the capture class
    size: int = None                                          # Stores the size of the capture area in 2x2km blocks
    
    def __init__(self, lat, lon, size = 25, start_year = 1999, end_year = 2020):
        '''Creates a Capture class instance

        Args:
            lat (float): The latitude of the origin of the capture area
            lon (float): The longitude of the origin of the capture area
            size (int, optional): The amount of 4km2 chunks to get. Defaults to 25.
            start_year (int, optional): The first year to get satellite images from. Defaults to 1999.
            end_year (int, optional): The last year to get satellite images from. Defaults to 2020.
        '''

        self.origin = Point(lat, lon)
        self.size = size
        self.interval = (start_year, end_year)

    def datesFromIndex(self, index) -> Tuple[str, str]:
        '''Generates a start and end date for the satellite images from a specified index

        Args:
            index (int): The index to generate start and end dates for

        Returns:
            Tuple[str, str]: A 2-tuple of strings representing the start and end dates respectively
        '''
        year = self.interval[0] + floor(index / self.size)

        if year > self.interval[1]:
            raise IndexError('The index \'%s\' is out of bounds, and generates a year beyond the end date', year)

        return ('%s-01-01' % year, '%s-12-31' % year)

    def generate(self, index=None):
        '''Generates an image

        Args:
            index (int, optional): The index of the generated sample. Defaults to None.
        '''

    	# Access the index, if none is provided generate the next ungenerated index
        if index is None:
            index = self.index
            self.index += 1

        date = self.datesFromIndex(index)

        geometry = ee.Geometry.Rectangle(self.rectangleFromIndex(index))

        data = self.collection.filterBounds(geometry)\
            .filterDate(date[0], date[1])\
            .map(cloudmask_L457)\
            .select(['B3', 'B2', 'B1'])

        name = self.nameFromIndex(index)

        task = ee.batch.Export.image.toDrive(**{
            'image': data.median(),
            'folder': 'landsat',
            'description': name,
            'scale': 30,
            'region': geometry,
            'maxPixels': 1e9
        })

        task.start()
        self.generated.append(name)
        self.log.info('Started task for image "%s", index: %s', name, index)

    def generateAll(self, download = False, save_path = './', delete = True, timeout = None):
        '''Generates all images in this capture area

        Args:
            download (bool, optional): Whether or not to download the images afterwards. Defaults to False.
            save_path (str, optional): The path to save the downloaded files to. Defaults to './'.
            delete (bool, optional): Whether or not to delete the file from Google Drive after downloading. Defaults to True.
            timeout (int, optional): The number of seconds to keep looking for downloadable files. Defaults to None, which means the number of generated files * 3 * 60.
        '''
        length = self.size * (self.interval[1] - self.interval[0])

        self.log.info('Generating %s images.', length)
        self.generated = []

        for i in range(length):
            self.generate(i)

        self.log.info('Generated all images. Expected load time is %s minutes', length * 3)

        to_download = set(self.generated)

        if download:
            from drive import Drive

            drive = Drive()
            start_time = perf_counter()
            timeout = timeout or length * 3 * 60
            time_difference = 0

            while len(to_download) > 0 and time_difference < timeout:
                downloaded = []

                for file in to_download:
                    if drive.download(file, '.tif', save_path, delete):
                        to_download.remove(file)
                        downloaded.append(file)

                time_difference = perf_counter() - start_time

                for dl_file in downloaded:
                    self.log.info('Downloaded file "%s", %i to go. (%i seconds left)', dl_file, len(to_download), time_difference)

            if len(to_download) == 0:
                self.log.info('Sucessfully generated and downloaded %i files', len(self.generated))
            else:
                self.log.warn('Failed to download %i of %i generated files.\nThey were: %s', len(to_download), len(self.generated), to_download)

    def nameFromIndex(self, index) -> str:
        '''Generates a filename from a specified index

        Args:
            index (int): The index to get the filename for

        Returns:
            str: The filename appropriate for the index
        '''

        return '{}_{}-{}-{}'.format(
            round(self.origin.latitude),                # A shorthand for the latitude of the origin
            round(self.origin.longitude),               # A shorthand for the longitude of the origin
            index % self.size,                          # The index of the tile (0 through self.size)
            self.interval[0] + floor(index / self.size) # The year of the image
        )

    def rectangleFromIndex(self, index) -> List[float]:
        '''Generates the rectangle of coordinates associated with a specified index

        Args:
            index (int): The index to create the rectangle for

        Returns:
            List[float]: The top left and bottom right corners of the rectangle in a four-item list
        '''

        # Access the topleft corner of the rectangle
        top_left = self.origin
        index = index % self.size
        
        if index > 0:
            (grid_x, grid_y) = spiralIndex(index)
            lon_movement = distance.geodesic(kilometers = 2 * abs(grid_x))
            lat_movement = distance.geodesic(kilometers = 2 * abs(grid_y))
            lon_bearing = 90 if grid_x >= 0 else 270
            lat_bearing = 0 if grid_y >= 0 else 180

            if grid_x != 0:
                top_left = lon_movement.destination(point = top_left, bearing = lon_bearing)

            if grid_y != 0:
                top_left = lat_movement.destination(point = top_left, bearing = lat_bearing)
        
        # Access the bottom right of the rectangle
        rectangle_side = distance.geodesic(kilometers = 3)
        bottom_right = rectangle_side.destination(point = rectangle_side.destination(point = top_left, bearing = 90), bearing = 180)

        return [top_left.longitude, top_left.latitude, bottom_right.longitude, bottom_right.latitude]