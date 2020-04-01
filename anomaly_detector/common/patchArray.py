import os
import time
import sys
import ast
from glob import glob
from cachetools import cached, Cache, LRUCache
from datetime import datetime

import numpy as np
import numpy.lib.recfunctions
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

from common import utils, logger, ImageLocationUtility
import consts

class Patch(np.record):
    image_cache = LRUCache(maxsize=20*60*2)  # Least recently used cache for images

    @cached(image_cache, key=lambda self, *args: self.times) # The cache should only be based on the timestamp
    def get_image(self, images_path=None):
        if images_path is None: images_path = consts.IMAGES_PATH
        return cv2.imread(os.path.join(images_path, "%i.jpg" % self.times))

    def __setattr__(self, attr, val):
        if attr in self.dtype.names:
            old_val = self.__getattribute__(attr)
            if not np.all(old_val == val): # Check if any value changed
                np.record.__setattr__(self, "changed", True)
                np.record.__setattr__(self, attr, val)
        else:
            np.record.__setattr__(self, attr, val)

class PatchArray(np.recarray):
    """Array with metadata."""
    
    __metadata_attrs__ = list()

    def __new__(cls, filename=None, metadata_filename=None, **kwargs):
        """Reads metadata and features from a HDF5 file.
        
        Args:
            filename (str): filename to read

        Returns:
            A new PatchArray
        """
        logger.info("Reading metadata and features from: %s" % filename)

        images_path = kwargs.get("images_path", consts.IMAGES_PATH)
        
        filename_metadata = os.path.join(images_path, "metadata_cache.h5")
        metadata = dict()

        with h5py.File(filename_metadata, "r") as hf:
            def _c(x, y):
                if isinstance(y, h5py.Dataset):
                    cls.__metadata_attrs__.append(x)
                    metadata[x] = np.array(y)
                    
            hf.visititems(_c)
        
        if len(metadata) == 0:
            raise ValueError("There should be at least a bit of metadata!")

        # Add missing datasets
        metadata["changed"] = np.zeros_like(metadata["labels"], dtype=np.bool)

        contains_features  = False
        contains_locations = False
        contains_bins      = False

        # Check if file is h5 file
        if isinstance(filename, str) and filename.endswith(".h5"):
            s = time.time()
            with h5py.File(filename, "r") as hf:
                logger.info("Opening %s: %f" % (filename, time.time() - s))

                # Metadata and Features are assumed to be sorted by time
                # But the features might only be a subset (think temporal patches, C3D)
                # So we first get the times and match them against each other
                feature_times = np.array(hf["times"])
                common_times, metadata_indices, feature_indices = np.intersect1d(metadata["times"], feature_times, assume_unique=True, return_indices=True)
                # Now we filter the metadata (there could be more metadata than features, eg. C3D)
                if common_times.shape != metadata["times"].shape or np.any(metadata_indices != np.arange(len(metadata_indices))):
                    for n, m in metadata.items():
                        metadata[n] = m[metadata_indices]

                # Test if everything worked out
                assert np.all(feature_times[feature_indices] == metadata["times"]), "Something went wrong"
                assert np.all(feature_indices == np.arange(len(feature_indices))), "Oops?"

                patches_dict = dict()

                if "features" in hf.keys():
                    contains_features = True
                    patches_dict["features"] = hf["features"]
                else:
                    raise ValueError("%s does not contain features." % filename)

                if "locations" in hf.keys():
                    contains_locations = True
                    patches_dict["locations"] = hf["locations"]
                else:
                    locations_shape = patches_dict["features"].shape[:-1]
                    patches_dict["locations"] = np.zeros(locations_shape, dtype=[("y", np.float32), ("x", np.float32)])

                locations_shape = patches_dict["features"].shape[:-1]
                patches_dict["bins"] = np.zeros(locations_shape, dtype=[("v", np.uint16), ("u", np.uint16)])
                patches_dict["cell_size"] = np.zeros(locations_shape, dtype=np.float64)

                # Broadcast metadata to the correct shape
                for n, m in metadata.items():
                    patches_dict[n] = np.moveaxis(np.broadcast_to(m, patches_dict["features"].shape[1:-1] + (m.size,)), -1, 0)

                # Create type
                t = [(x, patches_dict[x].dtype, patches_dict[x].shape[3:]) for x in patches_dict]

                s = time.time()
                patches = np.rec.fromarrays(patches_dict.values(), dtype=t)
                logger.info("Loading patches recarray: %f" % (time.time() - s))
        else:
            # Broadcast metadata to the correct shape
            for n, m in metadata.items():
                metadata[n] = np.moveaxis(np.broadcast_to(m, (2, 2, m.size)), -1, 0)

            # Create type
            t = [(x, metadata[x].dtype) for x in metadata]
            patches = np.rec.fromarrays(metadata.values(), dtype=t)
        
        obj = patches.view(cls)

        obj.filename = filename
        obj.images_path = images_path
        obj.contains_features  = contains_features
        obj.contains_locations = contains_locations
        obj.contains_bins      = contains_bins
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.filename    = getattr(obj, "filename", None)
        self.images_path = getattr(obj, "images_path", consts.IMAGES_PATH)
        self.contains_features  = getattr(obj, "contains_features", False)
        self.contains_locations = getattr(obj, "contains_locations", False)
        self.contains_bins      = getattr(obj, "contains_bins", False)
    
    def __setattr__(self, attr, val):
        if self.dtype.names is not None and attr in self.dtype.names:
            # Try finding the attribute in the metadata (will throw if there it is not in metadata)
            old_val = self.__getattribute__(attr)
            if not np.all(old_val == val): # Check if any value changed
                if attr != "changed" and attr in self.__metadata_attrs__:
                    # Set changed to true, where there was a change
                    if len(self.shape) > 0:
                        self["changed"][old_val != val] = True
                    else:
                        self["changed"] = True

                # Change the value
                np.recarray.__setattr__(self, attr, val)
        else:
            object.__setattr__(self, attr, val)

    def __getitem__(self, indx):
        obj = np.recarray.__getitem__(self, indx)

        if isinstance(obj, np.record):
            obj.__class__ = Patch
        
        return obj
    
    #################
    #     Views     #
    #################
    
    unknown_anomaly = property(lambda self: self[self.labels[:, 0, 0] == 0])
    no_anomaly      = property(lambda self: self[self.labels[:, 0, 0] == 1])
    anomaly         = property(lambda self: self[self.labels[:, 0, 0] == 2])
    
    direction_unknown = property(lambda self: self[self.directions[:, 0, 0] == 0])
    direction_ccw     = property(lambda self: self[self.directions[:, 0, 0] == 1])
    direction_cw      = property(lambda self: self[self.directions[:, 0, 0] == 2])
    
    round_number_unknown = property(lambda self: self[self.round_numbers[:, 0, 0] == 0])
    def round_number(self, round_number):
        return self[self.round_numbers[:, 0, 0] == round_number]

    metadata_changed = property(lambda self: self[self.changed[:, 0, 0]])

    def save_metadata(self, filename=None):
        if filename is None:
            filename = os.path.join(self.images_path, "metadata_cache.h5")
        with h5py.File(filename, "w") as hf:
            hf.attrs["Created"] = datetime.now().strftime("%d.%m.%Y, %H:%M:%S")
            hf.create_dataset("camera_locations", data=self.camera_location)
            hf.create_dataset("times",            data=self.time)
            hf.create_dataset("labels",           data=self.label)
            hf.create_dataset("directions",       data=self.direction)
            hf.create_dataset("round_numbers",    data=self.round_number)

    #################
    # Spatial stuff #
    #################

    # @profile
    def bin(self, bin, cell_size):
        """ Get a view of only the features that are in a specific bin

        Args:
            bin (Tuple): (v, u) Tuple with the bin coordinates
            cell_size (float): 
        """
        # Check if cell size is calculated
        if not self.contains_bins or self.cell_size.flat[0] != cell_size:
            self.calculate_rasterization(cell_size)
        
        b1 = self.bins.v == bin[0]
        r1 = self[b1]
        b2 = r1.bins.u == bin[1]
        r2 = r1[b2]

        return r2

    def get_spatial_histogram(self, cell_size):
        # Get extent
        x_min, y_min, x_max, y_max = self.get_extent(cell_size)

        bins_y = np.arange(y_min, y_max, cell_size)
        bins_x = np.arange(x_min, x_max, cell_size)

        return numpy.histogram2d(self.locations.y.ravel(), self.locations.x.ravel(), bins=[bins_y, bins_x])[0].T

    def show_spatial_histogram(self, cell_size):
        """ Show the spatial histogram of features (like a map) """
        plt.imshow(self.get_spatial_histogram(cell_size), vmin=1)
        plt.show()

    def calculate_rasterization(self, cell_size):
        # Get extent
        x_min, y_min, x_max, y_max = self.get_extent(cell_size)

        # Create the bins
        bins_y = np.arange(y_min, y_max, cell_size)
        bins_x = np.arange(x_min, x_max, cell_size)

        shape = (len(bins_y) + 1, len(bins_x) + 1)

        if self.cell_size.flat[0] == cell_size:
            return shape
        
        logger.info("%i bins in x and %i bins in y direction (with cell size %.2f)" % (shape + (cell_size,)))

        # Use digitize to sort the locations in the bins
        # --> (#frames, h, w, 2) where the entries in the last
        #     dimension are the respective bin indices (v, u)
        res = np.stack([np.digitize(self.locations.y, bins_y),
                        np.digitize(self.locations.x, bins_x)], axis=3)

        self.bins[...] = np.rec.fromarrays(res.transpose(), dtype=[("v", np.float32), ("u", np.float32)]).transpose()

        self.contains_bins = True
        self.cell_size[:] = cell_size

        return shape

    def get_extent(self, cell_size=None):
        """Calculates the extent of the features
        
        Args:
            cell_size (float): Round to cell size (increases bounds to fit next cell size)
        
        Returns:
            Tuple (x_min, y_min, x_max, y_max)
        """
        assert self.contains_locations, "Can only compute extent if there are patch locations"
        
        # Get the extent
        x_min = self.locations.x.min()
        y_min = self.locations.y.min()
        x_max = self.locations.x.max()
        y_max = self.locations.y.max()
        
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

        start = time.time()
        self.calculate_patch_locations()
        end = time.time()

        with h5py.File(filename, "r+") as hf:
            # Remove the old locations dataset
            if "locations" in hf.keys():
                del hf["locations"]
            
            hf.create_dataset("locations", data=self.locations)
            
            hf["locations"].attrs["Start"] = start
            hf["locations"].attrs["End"] = end
            hf["locations"].attrs["Duration"] = end - start
            hf["locations"].attrs["Duration (formatted)"] = utils.format_duration(end - start)
    
    def ensure_locations(self):
        assert self.contains_features, "Can only compute patch locations if there are patches"
        if not self.contains_locations:
            self.calculate_patch_locations()

    def calculate_patch_locations(self):
        """Calculate the real world coordinates of every feature"""
        assert self.contains_features, "Can only compute patch locations if there are patches"

        ilu = ImageLocationUtility()

        logger.info("Calculating locations of every patch")
        n, h, w = self.locations.shape
        
        image_locations = ilu.span_grid(h, w, offset_x=0.5, offset_y=0.5)   # (h, w)
        relative_locations = ilu.image_to_relative(image_locations)         # (h, w, 2)
        
        # # Construct inverse transformation matrices
        # s = np.sin(self.camera_locations["rotation"]["z"] - np.pi / 2.)     # (n)
        # c = np.cos(self.camera_locations["rotation"]["z"] - np.pi / 2.)     # (n)
        # tx = self.camera_locations["translation"]["x"]                      # (n)
        # ty = self.camera_locations["translation"]["y"]                      # (n)

        # matrices = np.zeros((n, 3, 3))                                      # (n, 3, 3)
        # # [[c,-s, tx],
        # #  [s, c, ty],
        # #  [0, 0,  1]]
        # matrices[:, 0, 0] = c
        # matrices[:, 0, 1] = -s
        # matrices[:, 0, 2] = tx
        # matrices[:, 1, 0] = s
        # matrices[:, 1, 1] = c
        # matrices[:, 1, 2] = ty
        # matrices[:, 2, 2] = 1
        
        for i in tqdm(range(n), desc="Calculating locations", file=sys.stderr):
            res = ilu.relative_to_absolute(relative_locations, self[i, 0, 0].camera_locations)
            res = np.rec.fromarrays(res.transpose(), dtype=[("y", np.float32), ("x", np.float32)]).transpose()
            self.locations[i] = res
        
        self.contains_locations = True

    # def add_location_as_feature_dimension(self):
    #     """Act as if the location of a feature would be two additional feature dimensions"""
    #     self.ensure_locations()
    #     logger.info("Adding locations as feature dimensions")
    #     return np.concatenate((self, self.locations), axis=-1).view(PatchArray)

    #################
    # Calculations  #
    #################
    
    def var(self):
        return np.var(self.ravel().features, axis=0, dtype=np.float64)

    def cov(self):
        return np.cov(self.ravel().features, rowvar=False)

    def mean(self):
        return np.mean(self.ravel().features, axis=0, dtype=np.float64)

    #################
    #      Misc     #
    #################
    
    def to_dataset(self):
        import tensorflow as tf

        def _gen():
            for i in range(self.shape[0]):
                rgb = cv2.cvtColor(self[i, 0, 0].get_image(), cv2.COLOR_BGR2RGB)
                yield (np.array(rgb), self[i, 0, 0].times)

        raw_dataset = tf.data.Dataset.from_generator(
            _gen,
            output_types=(tf.uint8, tf.int64),
            output_shapes=((None, None, None), ()))

        return raw_dataset.prefetch(tf.data.experimental.AUTOTUNE)

if __name__ == "__main__":
    from common import Visualize

    patches = PatchArray(consts.FEATURES_FILE)
    
    # r = patches.ravel()

    # for i, x in enumerate(r):
    #     if i >= len(patches) -2:
    #         print i

    # patches.show_spatial_histogram(1.0)

    # vis = Visualize(patches)
    # vis.show()

    patches[0, 0, 0].labels = 2
    print patches[0:5, 0, 0].labels
    print patches[0:5, 0, 0].changed

    patches[0:2, 0, 0].labels = 2

    print patches[0:5, 0, 0].labels
    print patches[0:5, 0, 0].changed