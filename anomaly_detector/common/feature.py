import os
from cachetools import cached, LRUCache
from cachetools.keys import hashkey

import numpy as np
import cv2

import anomaly_detector.consts as consts

class Feature(np.ndarray):
    """A Feature is the output of a Feature Extractor (values) with metadata as attributes"""

    def __new__(cls, input_array, x, y, w, h):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        
        obj.x = x
        obj.y = y
        obj.w = w
        obj.h = h

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        
        # Patch position
        self.x = getattr(obj, "x", None)
        self.y = getattr(obj, "y", None)

        # Patches per image
        self.w = getattr(obj, "w", None)
        self.h = getattr(obj, "h", None)

        self.camera_location = None
        self.time = None
        self.label = None
        self.rosbag = None
        self.tfrecord = None
        self.feature_extractor = None

        # Will eventually be array [x, y]
        # (call FeatureArray.calculate_locations)
        self.location = None

        self.__bins__ = {}
    
    camera_position   = property(lambda self: None if self.camera_location is None else
                                                np.array([self.camera_location[0],
                                                          self.camera_location[1]]))
    camera_rotation   = property(lambda self: None if self.camera_location is None else
                                                self.camera_location[5])

    image_cache = LRUCache(maxsize=20*60*2)  # Least recently used cache for images

    @cached(image_cache, key=lambda self, *args: hashkey(self.time)) # The cache should only be based on the timestamp
    def get_image(self, images_path=consts.IMAGES_PATH):
        return cv2.imread(os.path.join(images_path, "%i.jpg" % self.time))

    def get_bin(self, cell_size, extent=None):
        """Gets the indices for the bin the given feature belongs to

        Args:
            cell_size (float): Round to cell size (increases bounds to fit next cell size)
        
        Returns:
            Tuple containing the bin indices (u, v)
        """
        assert self.location is not None, "Feature locations need to be computed before computing bins"

        if cell_size in self.__bins__.keys():
            return self.__bins__[cell_size]
        else:
            if extent is None:
                raise ValueError("Extent needs to be specified for calculating the bin.")
            x_min, y_min, x_max, y_max = extent 
            self.__bins__[cell_size] = (int((self.location[0] - x_min) / cell_size),
                                        int((self.location[1] - y_min) / cell_size))
            return self.__bins__[cell_size]

    def cast(self, dtype):
        return np.array(self, dtype=dtype)

if __name__ == "__main__":
    f = Feature(np.array([2,3,4]), 0, 0, 10, 10)
    print f.w