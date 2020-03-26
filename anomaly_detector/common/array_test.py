import os
import time
import sys
import ast
from glob import glob

from datetime import datetime
import numpy as np
import numpy.lib.recfunctions
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

from common import utils, logger, ImageLocationUtility
import consts

class MetadataArray(np.recarray):
    def __new__(cls, filename=None):
        if filename is None:
            filename = os.path.join(consts.IMAGES_PATH, "metadata_cache.h5")
        
        datasets = dict()

        with h5py.File(filename, "r") as hf:
            def _c(x, y):
                if isinstance(y, h5py.Dataset):
                    datasets[x] = np.array(y)
                    
            hf.visititems(_c)
        
        metadata = np.rec.array(datasets.values(), dtype=[(x, datasets[x].dtype) for x in datasets])
        metadata = np.lib.recfunctions.append_fields(metadata, "changed", np.zeros(metadata.shape, dtype=np.bool))

        return metadata.view(cls)

class FeatureArray(np.ndarray):
    """Array with metadata."""

    def __new__(cls, filename=None, **kwargs):
        """Reads metadata and features from a HDF5 file.
        
        Args:
            filename (str): filename to read

        Returns:
            A new FeatureArray
        """
        logger.info("Reading metadata and features from: %s" % filename)

        _metadata = MetadataArray()

        images_path = kwargs.get("images_path", consts.IMAGES_PATH)

        # Check if file is h5 file
        if isinstance(filename, str) and filename.endswith(".h5"):
            with h5py.File(filename, "r") as hf:
                features = np.array(hf["features"], dtype=np.float32)
                locations = hf.get("locations")
                if locations is not None:
                    locations = np.array(locations)
        else:
            features = np.empty(shape=_metadata.shape + (1, 1), dtype=np.bool)
            locations = None
        
        obj = np.asarray(features).view(cls)
        obj._metadata = _metadata
        obj._locations = locations
        obj.filename = filename
        obj.images_path = images_path
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._metadata = getattr(obj, "_metadata", None)
        self._locations = getattr(obj, "_locations", None)
        self.filename = getattr(obj, "filename", None)
        self.images_path = getattr(obj, "images_path", consts.IMAGES_PATH)

        # Dict with cell size as key
        # - cell size
        self.__rasterizations__ = {}

    # Make direct access to metadata possible as attributes
    def __getattr__(self, name):
        if name != "_metadata" and name != "_locations":
            # Try finding the attribute in the metadata
            return self._metadata.__getattribute__(name)
    
    def __setattr__(self, key, value):
        try:
            # Try finding the attribute in the metadata (will throw if there it is not in metadata)
            old_value = self._metadata.__getattribute__(key)
            if not np.all(old_value == value): # Check if any value changed
                # Set changed to true, where there was a change
                if len(self._metadata.shape) > 0:
                    self._metadata.changed[old_value != value] = True
                else:
                    self._metadata.changed = True

                # Change the value
                self._metadata.__setattr__(key, value)
        except:
            np.ndarray.__setattr__(self, key, value)

    def __getitem__(self, key):
        result = np.ndarray.__getitem__(self, key)
        if isinstance(result, FeatureArray):
            # Slice metadata (if this is the base, metadata is per frame only)
            if self.base.base is None:
                if isinstance(key, tuple):
                    result._metadata = self._metadata[key[0]]
                else:
                    result._metadata = self._metadata[key]

            # Slice locations (for max 3 dimensions)
            if self._locations is not None:
                result._locations = self._locations[key]
        return result

    def flatten(self):


    #################
    #     Views     #
    #################
    
    unknown_anomaly = property(lambda self: self[self.label == 0])
    no_anomaly      = property(lambda self: self[self.label == 1])
    anomaly         = property(lambda self: self[self.label == 2])
    
    direction_unknown = property(lambda self: self[self.direction == 0])
    direction_ccw     = property(lambda self: self[self.direction == 1])
    direction_cw      = property(lambda self: self[self.direction == 2])
    
    round_number_unknown = property(lambda self: self[self.round_number == 0])
    def round_number(self, round_number):
        return self[self.round_number == round_number]

    metadata_changed = property(lambda self: self[self.changed])

    #################
    # Calculations  #
    #################
    
    def var(self):
        return np.var(self, axis=0, dtype=np.float64)

    def cov(self):
        return np.cov(self, rowvar=False)

    def mean(self):
        return np.mean(self, axis=0, dtype=np.float64)

    #################
    #      Misc     #
    #################
    
    def to_dataset(self):
        import tensorflow as tf

        def _gen():
            for f in self[:, 0, 0]:
                rgb = cv2.cvtColor(f.get_image(), cv2.COLOR_BGR2RGB)
                yield (np.array(rgb), f.time)

        raw_dataset = tf.data.Dataset.from_generator(
            _gen,
            output_types=(tf.uint8, tf.int64),
            output_shapes=((None, None, None), ()))

        return raw_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    

if __name__ == "__main__":
    features = FeatureArray(consts.FEATURES_FILE)

    # print features.shape
    # print features.metadata.shape

    # features.direction

    loc_0_0_0 = features._locations[0, 0, 0]

    print features[0, 0, 0]._locations, features[0, 0, 0]._locations == loc_0_0_0
    print features[0:4, 0, 0][0]._locations, features[0:4, 0, 0][0]._locations == loc_0_0_0
    print features[0, 0, 0, 0]#._locations, features[0, 0, 0, 0]._locations == loc_0_0_0
    print features[0][0][0]._locations, features[0][0][0]._locations == loc_0_0_0

    print features[0][0:3][0,0]._locations, features[0][0:3][0,0]._locations == loc_0_0_0

    # features[0].shape
    # features[0:4, 0, 0].shape
    # features[0, 0].shape
    # features[0,0,0].shape

    # features[0].label = 2
    # print features[0:5].label
    # print features[0:5].changed

    # features[0:2].label = 2

    # print features[0:5].label
    # print features[0:5].changed

    # print features.metadata_changed.shape

    # print features[0,...,1:2].shape
    # print features[0,...,1:2].label

    # print features[15:30].shape
    # print features[15:30].metadata.direction

    # print features[features.metadata.label == 0].shape
    # print features[features.metadata.label == 0].metadata.label.shape