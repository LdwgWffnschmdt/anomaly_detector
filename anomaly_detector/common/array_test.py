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
    def __new__(cls, filename=None, ignore=()):
        if filename is None:
            filename = os.path.join(consts.IMAGES_PATH, "metadata_cache.h5")
        
        datasets = dict()

        with h5py.File(filename, "r") as hf:
            def _c(x, y):
                if isinstance(y, h5py.Dataset) and not x in ignore:
                    datasets[x] = np.array(y)
                    
            hf.visititems(_c)
        
        if len(datasets) == 0:
            return None

        metadata = np.rec.array(datasets.values(), dtype=[(x, datasets[x].dtype) for x in datasets])
        metadata = np.lib.recfunctions.append_fields(metadata, "changed", np.zeros(metadata.shape, dtype=np.bool))

        return metadata.view(cls)

    def has_dataset(self, name):
        return name in self.dtype.names

    # def require_dataset(self, name, dtype=np.float32):
    #     if not self.has_dataset(name):
            

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

        _metadata_frame = MetadataArray()

        images_path = kwargs.get("images_path", consts.IMAGES_PATH)

        # Check if file is h5 file
        if isinstance(filename, str) and filename.endswith(".h5"):
            with h5py.File(filename, "r") as hf:
                features = np.array(hf["features"], dtype=np.float32)
            _metadata_features = MetadataArray(filename, ignore=("feaures"))
        else:
            features = np.empty(shape=_metadata_frame.shape + (0, 0), dtype=np.bool)
            _metadata_features = None
        
        obj = np.asarray(features).view(cls)
        obj._metadata_frame    = _metadata_frame
        obj._metadata_features = _metadata_features
        obj.filename = filename
        obj.images_path = images_path
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._metadata_frame    = getattr(obj, "_metadata_frame", None)
        self._metadata_features = getattr(obj, "_metadata_features", None)
        self.filename = getattr(obj, "filename", None)
        self.images_path = getattr(obj, "images_path", consts.IMAGES_PATH)

        # Dict with cell size as key
        # - cell size
        self.__rasterizations__ = {}

    # Make direct access to metadata possible as attributes
    def __getattr__(self, name):
        if name != "_metadata_frame" and name != "_metadata_features":
            try:
                # Try finding the attribute in the metadata
                return self._metadata_frame.__getattribute__(name)
            except:
                # Try finding the attribute in the per feature metadata
                return self._metadata_features.__getattribute__(name)
    
    def __setattr__(self, key, value):
        try:
            # Try finding the attribute in the metadata (will throw if there it is not in metadata)
            old_value = self._metadata_frame.__getattribute__(key)
            if not np.all(old_value == value): # Check if any value changed
                # Set changed to true, where there was a change
                if len(self._metadata_frame.shape) > 0:
                    self._metadata_frame.changed[old_value != value] = True
                else:
                    self._metadata_frame.changed = True

                # Change the value
                self._metadata_frame.__setattr__(key, value)
        except:
            try:
                # Try finding the attribute in the per feature metadata (will throw if there it is not in metadata)
                old_value = self._metadata_features.__getattribute__(key)
                if not np.all(old_value == value): # Check if any value changed
                    # Set changed to true, where there was a change
                    if len(self._metadata_features.shape) > 0:
                        self._metadata_features.changed[old_value != value] = True
                    else:
                        self._metadata_features.changed = True

                    # Change the value
                    self._metadata_features.__setattr__(key, value)
            except:
                # If both fails, call base
                np.ndarray.__setattr__(self, key, value)

    def __getitem__(self, key):
        result = np.ndarray.__getitem__(self, key)
        if isinstance(result, FeatureArray):
            # Slice metadata (if this is the base, metadata is per frame only)
            if self.base.base is None:
                if isinstance(key, tuple):
                    result._metadata_frame = self._metadata_frame[key[0]]
                else:
                    result._metadata_frame = self._metadata_frame[key]

            # Slice metadata_features
            if self._metadata_features is not None:
                result._metadata_features = self._metadata_features[key]
        return result

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
    # Spatial stuff #
    #################

    # def bin(self, bin, cell_size):
    #     """ Get a view of only the features that are in a specific bin

    #     Args:
    #         bin (Tuple): (u, v) Tuple with the bin coordinates
    #         cell_size (float): 
    #     """
    #     # Check if cell size is calculated
    #     if not cell_size in self.__rasterizations__.keys():
    #         self.calculate_rasterization(cell_size)
        
    #     if self.__rasterizations__[cell_size]["feature_indices"][bin] is None:
    #         return []

    #     return self.flatten()[self.__rasterizations__[cell_size]["feature_indices"][bin]]

    # def get_spatial_histogram(self, cell_size):
    #     if not cell_size in self.__rasterizations__.keys():
    #         # Try loading from file
    #         with h5py.File(self.filename, "r") as hf:
    #             g = hf.get("rasterizations/%.2f" % cell_size)
    #             if g is not None:
    #                 self.__rasterizations__[cell_size] = {}
    #                 self.__rasterizations__[cell_size]["feature_indices_count"] = np.array(hf["rasterizations/%.2f" % cell_size])
        
    #     # Check if cell size is calculated
    #     if not cell_size in self.__rasterizations__.keys():
    #         self.calculate_rasterization(cell_size)

    #     return self.__rasterizations__[cell_size]["feature_indices_count"]

    # def show_spatial_histogram(self, cell_size):
    #     """ Show the spatial histogram of features (like a map) """
    #     plt.imshow(self.get_spatial_histogram(cell_size), vmin=1)
    #     plt.show()

    def calculate_rasterization(self, cell_size):
        # Get extent
        x_min, y_min, x_max, y_max = self.get_extent(cell_size)

        # Create the bins
        bins_x = np.arange(x_min, x_max, cell_size)
        bins_y = np.arange(y_min, y_max, cell_size)

        # Use digitize to sort the locations in the bins
        # --> (#frames, h, w, 2) where the entries in the last
        #     dimension are the respective bin indices (u, v)
        indices = np.stack([np.digitize(self.locations[...,0], bins_x),
                            np.digitize(self.locations[...,1], bins_y)], axis=3)



        # Check if cell size is already calculated
        if cell_size in self.__rasterizations__.keys() and "feature_indices" in self.__rasterizations__[cell_size]:
            return self.__rasterizations__[cell_size]["feature_indices"].shape
        
        self.__rasterizations__[cell_size] = {}
        
        # Get extent
        extent = self.get_extent(cell_size)
        x_min, y_min, x_max, y_max = extent
        
        shape = (int(np.ceil((x_max - x_min) / cell_size)),
                 int(np.ceil((y_max - y_min) / cell_size)))

        logger.info("%i bins in x and %i bins in y direction (with cell size %.2f)" % (shape + (cell_size,)))

        logger.info("Calculating corresponding bin for every feature")
        self.__rasterizations__[cell_size]["feature_indices"] = np.empty(shape=shape, dtype=object)
        self.__rasterizations__[cell_size]["feature_indices_count"] = np.zeros(shape=shape, dtype=np.uint32)
        # Get the corresponding bin for every feature
        for i, f in enumerate(tqdm(self.flatten(), desc="Calculating bins"), file=sys.stderr):
            bin = f.get_bin(cell_size, extent)

            if self.__rasterizations__[cell_size]["feature_indices"][bin] is None:
                self.__rasterizations__[cell_size]["feature_indices"][bin] = list()
            self.__rasterizations__[cell_size]["feature_indices"][bin].append(i)
            self.__rasterizations__[cell_size]["feature_indices_count"][bin] += 1
        
        # Save to file
        try:
            with h5py.File(self.filename, "r+") as hf:
                hf["rasterizations/%.2f" % cell_size] = self.__rasterizations__[cell_size]["feature_indices_count"]
        except:
            pass

        return shape

    def get_extent(self, cell_size=None):
        """Calculates the extent of the features
        
        Args:
            cell_size (float): Round to cell size (increases bounds to fit next cell size)
        
        Returns:
            Tuple (x_min, y_min, x_max, y_max)
        """
        
        # Get the extent
        x_min = self.locations[...,0].min()
        y_min = self.locations[...,1].min()
        x_max = self.locations[...,0].max()
        y_max = self.locations[...,1].max()
        
        # Increase the extent to fit the cell size
        if cell_size is not None:
            x_min -= x_min % cell_size
            y_min -= y_min % cell_size
            x_max += cell_size - (x_max % cell_size)
            y_max += cell_size - (y_max % cell_size)
        
        return (x_min, y_min, x_max, y_max)

    def save_locations_to_file(self, filename=None):
        """ Save the locations of every patch to filename """

        if filename is None:
            filename = self.filename

        with h5py.File(filename, "r+") as hf:
            # Remove the old locations dataset
            if "locations" in hf.keys():
                del hf["locations"]
            
            start = time.time()
            self.calculate_locations()
            end = time.time()

            hf.create_dataset("locations", data=self.locations, dtype=np.float64)
            
            hf["locations"].attrs["Start"] = start
            hf["locations"].attrs["End"] = end
            hf["locations"].attrs["Duration"] = end - start
            hf["locations"].attrs["Duration (formatted)"] = utils.format_duration(end - start)
    
    def ensure_locations(self):
        if self.locations is None:
            self.calculate_locations()

    def calculate_locations(self):
        """Calculate the real world coordinates of every feature"""

        # If this FeatureArray is just a view, calculate the locations for the real FeatureArray
        if self.base is not None and isinstance(self.base, FeatureArray):
            print "Going down!"
            self.base.calculate_locations()
            return

        if self.shape[1] == 0:
            raise ValueError("Can't calculate locations on an empty FeatureArray.")

        ilu = ImageLocationUtility()

        logger.info("Calculating locations of every patch")
        _, h, w = self.shape
        
        image_locations = ilu.span_grid(w, h, offset_x=0.5, offset_y=0.5)
        relative_locations = ilu.image_to_relative(image_locations)
        
        camera_translations = np.stack([self.camera_translation_x, self.camera_translation_y], axis=1)

        locations = np.tile(relative_locations, (self.shape[0],) + relative_locations.shape)
        
        for i, y, x in tqdm(np.ndindex(self[...,0].shape), desc="Calculating locations", total=np.prod(self[...,0].shape), file=sys.stderr):
            locations[i, y, x] = ilu.relative_to_absolute(relative_locations[x, y], camera_translations[i], self.camera_rotation_z[i])
        # TODO: Muss natuerlich aufgeteilt werden in loc_x und y
        # Set locations
        if self._metadata_features is None:
            self._metadata_features = np.rec.array(locations, dtype=[("locations", locations.dtype)]).view(MetadataArray)
        else:
            self._metadata_features = np.lib.recfunctions.append_fields(self._metadata_features, ("locations"), locations)

    

    def add_location_as_feature_dimension(self):
        """Act as if the location of a feature would be two additional feature dimensions"""
        self.ensure_locations()
        logger.info("Adding locations as feature dimensions")
        return np.concatenate((self, self.locations), axis=-1).view(FeatureArray)

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

    loc_0_0_0 = features.locations[0, 0, 0]

    print features[0, 0, 0].locations, features[0, 0, 0].locations == loc_0_0_0
    print features[0:4, 0, 0][0].locations, features[0:4, 0, 0][0].locations == loc_0_0_0
    print features[0, 0, 0, 0]#.locations, features[0, 0, 0, 0].locations == loc_0_0_0
    print features[0][0][0].locations, features[0][0][0].locations == loc_0_0_0

    print features[0][0:3][0,0].locations, features[0][0:3][0,0].locations == loc_0_0_0

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