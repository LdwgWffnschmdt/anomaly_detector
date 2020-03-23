import os
from cachetools import cached, Cache, LRUCache

import numpy as np
import cv2
import yaml

import consts

class FeatureProperty(object):
    __cache__ = Cache(maxsize=9999999999)  # Least recently used cache for metadata
    __changed__ = list()

    def __init__(self, key, default=None):
        self.key = key
        self.default = default

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.__get_meta__(obj).get(self.key, self.default)

    def __set__(self, obj, value):
        # Get metadata
        meta = self.__get_meta__(obj)
        if meta.get(self.key, self.default) != value:
            meta[self.key] = value

            # Update the cache only. The metadata file is only updated on save()
            self.__cache__[obj.time] = meta
            self.__changed__.append(obj.time)
    
    @staticmethod
    @cached(__cache__, key=lambda obj: obj.time) # The cache should only be based on the timestamp
    def __get_meta__(obj):
        # Load and decode metadata file
        with open(FeatureProperty.__meta_file__(obj), "r") as yaml_file:
            return yaml.safe_load(yaml_file)
    
    @staticmethod
    def __meta_file__(obj):
        res = os.path.join(obj.images_path, "%i.yml" % obj.time)

        if not os.path.exists(res) or not os.path.isfile(res):
            raise ValueError("Could not find metadata file (%s)" % res)

        return res

    @staticmethod
    def changed(obj):
        if not obj.time in FeatureProperty.__changed__:
            return False

        # If metadata for obj is not in the cache it was not loaded and thus not modified => do nothing
        if FeatureProperty.__cache__.get(obj.time, None) is None:
            return False

        return True
    
    @staticmethod
    def save(obj):
        if not FeatureProperty.changed(obj):
            return False

        # Get metadata from the cache
        meta = FeatureProperty.__cache__.get(obj.time, None)

        # Update file with metadata from the cache
        with open(FeatureProperty.__meta_file__(obj), "w") as yaml_file:
            yaml.dump(meta, yaml_file, default_flow_style=False)
            
        FeatureProperty.__changed__.remove(obj.time)
        return True

class Feature(np.ndarray):
    """A Feature is the output of a Feature Extractor (values) with metadata as attributes"""

    image_cache = LRUCache(maxsize=20*60*2)  # Least recently used cache for images

    def __new__(cls, input_array, time, x, y, w, h):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        
        obj.time = time

        obj.x = x
        obj.y = y
        obj.w = w
        obj.h = h

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        
        self.time = getattr(obj, "time", None)

        # Patch position
        self.x = getattr(obj, "x", None)
        self.y = getattr(obj, "y", None)

        # Patches per image
        self.w = getattr(obj, "w", None)
        self.h = getattr(obj, "h", None)

        self.feature_extractor = None

        self.images_path = consts.IMAGES_PATH # Can be overwritten

        # Will eventually be array [x, y]
        # (call FeatureArray.calculate_locations)
        self.location = None

        self.__bins__ = {}
    
    direction             = FeatureProperty("direction", 0)     # 0: Unknown, 1: CCW, 2: CW
    round_number          = FeatureProperty("round_number", 0)  # 0: Unknown, >=0: Round index (zero based)

    label                 = FeatureProperty("label", 0)         # 0: Unknown, 1: No anomaly, 2: Contains an anomaly
    rosbag                = FeatureProperty("rosbag")
    
    camera_rotation_x    = FeatureProperty("location/rotation/x")
    camera_rotation_y    = FeatureProperty("location/rotation/y")
    camera_rotation_z    = FeatureProperty("location/rotation/z")

    camera_translation_x = FeatureProperty("location/translation/x")
    camera_translation_y = FeatureProperty("location/translation/y")
    camera_translation_z = FeatureProperty("location/translation/z")

    camera_location = property(lambda self: np.array([self.camera_translation_x,
                                                      self.camera_translation_y,
                                                      self.camera_translation_z,
                                                      self.camera_rotation_x,
                                                      self.camera_rotation_y,
                                                      self.camera_rotation_z]))

    camera_translation = property(lambda self: np.array([self.camera_translation_x, self.camera_translation_y]))

    metadata_changed = property(lambda self: FeatureProperty.changed(self))

    def preload_metadata(self):
        return FeatureProperty.__get_meta__(self)

    def save_metadata(self):
        return FeatureProperty.save(self)

    @cached(image_cache, key=lambda self, *args: self.time) # The cache should only be based on the timestamp
    def get_image(self, images_path=None):
        if images_path is None: images_path = self.images_path
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
    f1 = Feature(np.array([2,3,4]), 1581005499270958533, 0, 0, 10, 10)
    f2 = Feature(np.array([2,3,4]), 1581005499270958533, 1, 0, 10, 10)

    print f1.w

    l = f1.label
    print l
    print f2.label

    f1.label = 2
    print f1.label
    print f2.label

    f1.save_metadata()

    f1.label = l
    print f1.label
    print f2.label
    
    f1.save_metadata()