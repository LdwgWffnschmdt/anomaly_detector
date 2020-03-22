import os
import time
import sys
import ast
from glob import glob

import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

from common import utils, logger, Feature, ImageLocationUtility
import consts

class FeatureArray(np.ndarray):
    """Array with metadata."""

    def __new__(cls, filename, **kwargs):
        """Reads metadata and features from a HDF5 file.
        
        Args:
            filename (str): filename to read

        Returns:
            A new FeatureArray
        """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type

        logger.info("Reading metadata and features from: %s" % filename)

        # Check if filename is path to images
        if os.path.isdir(filename):
            attrs = dict()

            # Get all images
            files = glob(os.path.join(filename, "*.jpg"))

            features = np.empty(shape=(len(files), 1, 1), dtype=Feature)
            
            for i, f in enumerate(tqdm(files, desc="Loading empty features", file=sys.stderr)):
                time = int(os.path.splitext(os.path.basename(f))[0])
                feature = Feature(np.empty(()), time, 0, 0, 0, 0)
                feature.images_path = filename
                features[i, 0, 0] = feature

        # Check if file is h5 file
        elif os.path.splitext(filename)[1] == ".h5":
            with h5py.File(filename, "r") as hf:
                images_path = kwargs.get("images_path", consts.IMAGES_PATH)

                attrs = dict(hf.attrs)

                dt_str = h5py.string_dtype(encoding='ascii')

                features_raw      = np.array(hf["features"], dtype=np.float32)
                times             = np.array(hf["times"]   , dtype=np.uint64)
                feature_extractor = hf.attrs["Extractor"]

                locations = hf.get("locations")
                if locations is not None:
                    locations = np.array(locations)
                
                total, h, w, _ = features_raw.shape

                features = np.empty(shape=(total, h, w), dtype=Feature)
                
                for i, y, x in tqdm(np.ndindex(features.shape), desc="Loading features", total=np.prod(features.shape), file=sys.stderr):
                    feature = Feature(features_raw[i, y, x], times[i], x, y, w, h)
                    feature.feature_extractor = feature_extractor
                    feature.images_path = images_path
                    
                    if locations is not None:
                        feature.location = locations[i, y, x]

                    features[i, y, x] = feature
        
        else:
            raise ValueError("Filename has to be *.h5 or path to images with metadata files")
        
        obj = np.asarray(features).view(cls)
        obj.attrs = attrs
        obj.filename = filename
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.attrs = getattr(obj, "attrs", None)
        self.filename = getattr(obj, "filename", None)

        self.__extent__ = None

        # Dict with cell size as key
        # - cell size
        self.__rasterizations__ = {}

    def __get_property__(key):
        return lambda self: None if self.attrs is None or not key in self.attrs.keys() else self.attrs[key]

    extractor  = property(__get_property__("Extractor"))
    files      = property(__get_property__("Files"))
    batch_size = property(__get_property__("Batch size"))
    
    ### Often used views for convenience
    unknown_anomaly = property(lambda self: self[[f[0,0].label == 0 for f in self]])
    no_anomaly      = property(lambda self: self[[f[0,0].label == 1 for f in self]])
    anomaly         = property(lambda self: self[[f[0,0].label == 2 for f in self]])
    
    direction_unknown = property(lambda self: self[[f[0,0].direction == 0 for f in self]])
    direction_ccw     = property(lambda self: self[[f[0,0].direction == 1 for f in self]])
    direction_cw      = property(lambda self: self[[f[0,0].direction == 2 for f in self]])
    
    round_number_unknown = property(lambda self: self[[f[0,0].round_number == 0 for f in self]])
    def round_number(self, round_number):
        return self[[f[0,0].round_number == round_number for f in self]]

    metadata_changed = property(lambda self: self[[f[0,0].metadata_changed for f in self]])

    def save_metadata(self):
        for f in tqdm(self.metadata_changed.flatten(), desc="Saving metadata to file", file=sys.stderr):
            f.save_metadata()

    def preload_metadata(self):
        for f in tqdm(self.flatten(), desc="Preloading metadata", file=sys.stderr):
            f.preload_metadata()

    def bin(self, bin, cell_size):
        """ Get a view of only the features that are in a specific bin

        Args:
            bin (Tuple): (u, v) Tuple with the bin coordinates
            cell_size (float): 
        """
        # Check if cell size is calculated
        if not cell_size in self.__rasterizations__.keys():
            self.calculate_rasterization(cell_size)
        
        if self.__rasterizations__[cell_size]["feature_indices"][bin] is None:
            return []

        return self.flatten()[self.__rasterizations__[cell_size]["feature_indices"][bin]]

    def get_spatial_histogram(self, cell_size):
        if not cell_size in self.__rasterizations__.keys():
            # Try loading from file
            with h5py.File(self.filename, "r") as hf:
                g = hf.get("rasterizations/%.2f" % cell_size)
                if g is not None:
                    self.__rasterizations__[cell_size] = {}
                    self.__rasterizations__[cell_size]["feature_indices_count"] = np.array(hf["rasterizations/%.2f" % cell_size])
        
        # Check if cell size is calculated
        if not cell_size in self.__rasterizations__.keys():
            self.calculate_rasterization(cell_size)

        return self.__rasterizations__[cell_size]["feature_indices_count"]

    def show_spatial_histogram(self, cell_size):
        """ Show the spatial histogram of features (like a map) """
        plt.imshow(self.get_spatial_histogram(cell_size), vmin=1)
        plt.show()

    def calculate_rasterization(self, cell_size):
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
        
        if self.__extent__ is None:
            # Try loading from file
            with h5py.File(self.filename, "r") as hf:
                if "Extent" in hf.attrs.keys():
                    self.__extent__ = hf.attrs["Extent"]

        # Get the extent
        if self.__extent__ is None:
            x_min = self[0,0,0].location[0]
            y_min = self[0,0,0].location[1]
            x_max = self[0,0,0].location[0]
            y_max = self[0,0,0].location[1]

            for f in tqdm(self.flatten(), desc="Calculating extent", file=sys.stderr):
                if f.location[0] > x_max:
                    x_max = f.location[0]
                if f.location[0] < x_min:
                    x_min = f.location[0]

                if f.location[1] > y_max:
                    y_max = f.location[1]
                if f.location[1] < y_min:
                    y_min = f.location[1]
            self.__extent__ = (x_min, y_min, x_max, y_max)
            
            # Save to file
            with h5py.File(self.filename, "r+") as hf:
                hf.attrs["Extent"] = self.__extent__
            
            logger.info("Extent of features: (%i, %i)/(%i, %i)." % (x_min, y_min, x_max, y_max))
        else:
            x_min, y_min, x_max, y_max = self.__extent__
        

        # Increase the extent to fit the cell size
        if cell_size is not None:
            x_min -= x_min % cell_size
            y_min -= y_min % cell_size
            x_max += cell_size - (x_max % cell_size)
            y_max += cell_size - (y_max % cell_size)
        
        return (x_min, y_min, x_max, y_max)

    def __get_array_flat__(self):
        if hasattr(self, "__array_flat__"):
            return self.__array_flat__
        flat = self.flatten()
        self.__array_flat__ = np.concatenate(flat).reshape(flat.shape + flat[0].shape)
        return self.__array_flat__

    array_flat = property(__get_array_flat__)

    def var(self):
        return np.var(self.array_flat, axis=0, dtype=np.float64)

    def cov(self):
        return np.cov(self.array_flat, rowvar=False)

    def mean(self):
        return np.mean(self.array_flat, axis=0, dtype=np.float64)

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

            locations = np.reshape([f.location for f in self.flatten()], self.shape + (2,))
            hf.create_dataset("locations", data=locations, dtype=np.float64)
            
            hf["locations"].attrs["Start"] = start
            hf["locations"].attrs["End"] = end
            hf["locations"].attrs["Duration"] = end - start
            hf["locations"].attrs["Duration (formatted)"] = utils.format_duration(end - start)
    
    def ensure_locations(self):
        # Cheap check TODO: Improve on this and only calculate the locations not done yet
        if self.take(1).location is None:
            self.calculate_locations()

    def calculate_locations(self):
        """Calculate the real world coordinates of every feature"""

        # If this FeatureArray is just a view, calculate the locations for the real FeatureArray
        if self.base is not None and isinstance(self.base, FeatureArray):
            print "Going down!"
            self.base.calculate_locations()
            return

        ilu = ImageLocationUtility()

        logger.info("Calculating locations of every patch")
        _, h, w = self.shape
        
        image_locations = ilu.span_grid(w, h, offset_x=0.5, offset_y=0.5)
        relative_locations = ilu.image_to_relative(image_locations)

        for i, y, x in tqdm(np.ndindex(self.shape), desc="Calculating locations", total=np.prod(self.shape), file=sys.stderr):
            feature = self[i, y, x]
            feature.location = ilu.relative_to_absolute(relative_locations[x, y], feature.camera_translation, feature.camera_rotation)

    def add_location_as_feature_dimension(self):
        """Act as if the location of a feature would be two additional feature dimensions"""
        self.ensure_locations()

        logger.info("Adding locations as feature dimensions")

        features_flat = self.flatten()
        for i in range(len(features_flat)):
            features_flat[i] = np.append(features_flat[i], features_flat[i].location)

if __name__ == "__main__":
    import consts
    import timeit

    features = FeatureArray(consts.FEATURES_FILE)

    def _test_meta_load():
        x = features[0, 0, 0].label

    print timeit.repeat(_test_meta_load, repeat=1, number=5)