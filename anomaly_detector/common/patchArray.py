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

from shapely.strtree import STRtree
from shapely.geometry import Polygon, box

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

    root = None

    def __new__(cls, filename=None, metadata_filename=None, **kwargs):
        """Reads metadata and features from a HDF5 file.
        
        Args:
            filename (str): filename to read

        Returns:
            A new PatchArray
        """
        if cls.root is not None:
            logger.warning("There is already a root PatchArray loaded.")
        
        logger.info("Reading metadata and features from: %s" % filename)

        images_path = kwargs.get("images_path", consts.IMAGES_PATH)
        
        filename_metadata = os.path.join(images_path, "metadata_cache.h5")
        metadata = dict()

        with h5py.File(filename_metadata, "r") as hf:
            receptive_field = hf.attrs.get("Receptive field", None)
            image_size = hf.attrs.get("Image size", None)

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
        contains_bins      = {"bins_0.20": False, "bins_0.50": False}
        contains_mahalanobis_distances = False

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
                mahalanobis_dict = dict()

                add = ["features", "locations"]

                def _add(x, y):
                    if not isinstance(y, h5py.Dataset):
                        return
                    if x in add or x.startswith("bins"):
                        patches_dict[x] = y
                    elif x.endswith("/mahalanobis_distances"):
                        n = x.replace("/mahalanobis_distances", "")
                        mahalanobis_dict[n] = numpy.array(y)
                        mahalanobis_dict[n][np.isnan(mahalanobis_dict[n])] = -1

                hf.visititems(_add)

                if "features" in patches_dict.keys():
                    contains_features = True
                    if patches_dict["features"].ndim == 2:
                        patches_dict["features"] = np.expand_dims(np.expand_dims(patches_dict["features"], axis=1), axis=2)
                else:
                    raise ValueError("%s does not contain features." % filename)

                locations_shape = patches_dict["features"].shape[:-1]

                if "locations" in patches_dict.keys():
                    contains_locations = True
                else:
                    patches_dict["locations"] = np.zeros(locations_shape, dtype=[("tl", [("y", np.float32), ("x", np.float32)]),
                                                                                 ("tr", [("y", np.float32), ("x", np.float32)]),
                                                                                 ("bl", [("y", np.float32), ("x", np.float32)]),
                                                                                 ("br", [("y", np.float32), ("x", np.float32)])])

                if len(mahalanobis_dict) > 0:
                    contains_mahalanobis_distances = True
                    t = [(x, mahalanobis_dict[x].dtype) for x in mahalanobis_dict]
                    patches_dict["mahalanobis_distances"] = np.rec.fromarrays(mahalanobis_dict.values(), dtype=t)
                    patches_dict["mahalanobis_distances_filtered"] = np.zeros(locations_shape, dtype=np.float64)
                
                for k in contains_bins.keys():
                    if k in patches_dict.keys():
                        contains_bins[k] = True
                    else:
                        contains_bins[k] = False
                        patches_dict[k] = np.zeros(locations_shape, dtype=[("v", np.uint16), ("u", np.uint16), ("weight", np.float32)])

                # Broadcast metadata to the correct shape
                if patches_dict["features"].shape[1:-1] != ():
                    for n, m in metadata.items():
                        patches_dict[n] = np.moveaxis(np.broadcast_to(m, patches_dict["features"].shape[1:-1] + (m.size,)), -1, 0)

                # Create type
                t = [(x, patches_dict[x].dtype, patches_dict[x].shape[patches_dict["features"].ndim - 1:]) for x in patches_dict]

                s = time.time()
                patches = np.rec.fromarrays(patches_dict.values(), dtype=t)
                logger.info("Loading patches: %f" % (time.time() - s))
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
        obj.receptive_field = receptive_field
        obj.image_size = image_size
        obj.contains_features  = contains_features
        obj.contains_locations = contains_locations
        obj.contains_bins      = contains_bins
        obj.contains_mahalanobis_distances = contains_mahalanobis_distances

        cls.root = obj

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.filename    = getattr(obj, "filename", None)
        self.images_path = getattr(obj, "images_path", consts.IMAGES_PATH)
        self.receptive_field  = getattr(obj, "receptive_field", None)
        self.image_size  = getattr(obj, "image_size", 224)
        self.contains_features  = getattr(obj, "contains_features", False)
        self.contains_locations = getattr(obj, "contains_locations", False)
        self.contains_bins      = getattr(obj, "contains_bins", {"bins_0.20": False, "bins_0.50": False})
        self.contains_mahalanobis_distances = getattr(obj, "contains_mahalanobis_distances", False)
    
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
            hf.create_dataset("camera_locations", data=self.camera_locations[:, 0, 0])
            hf.create_dataset("times",            data=self.times[:, 0, 0])
            hf.create_dataset("labels",           data=self.labels[:, 0, 0])
            hf.create_dataset("directions",       data=self.directions[:, 0, 0])
            hf.create_dataset("round_numbers",    data=self.round_numbers[:, 0, 0])

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

    def calculate_rasterization(self, cell_size):
        # Check if cell size is already calculated
        if cell_size in self.__rasterizations__.keys() and "feature_indices" in self.__rasterizations__[cell_size]:
            return self.__rasterizations__[cell_size]["feature_indices"].shape
        
        # Get extent
        x_min, y_min, x_max, y_max = self.get_extent(cell_size)

        # Create the bins
        bins_y = np.arange(y_min, y_max, cell_size)
        bins_x = np.arange(x_min, x_max, cell_size)

        grid = [box(x, y, x + cell_size, y + cell_size)]

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

    def calculate_receptive_field(self, y, x):
        _, h, w = self.locations.shape
        center_y = y / float(h) * self.image_size
        center_x = x / float(w) * self.image_size

        rf_h = self.receptive_field.size[0] / 2.0
        rf_w = self.receptive_field.size[1] / 2.0
        
        tl = (max(0, center_y - rf_h), max(0, center_x - rf_w))
        tr = (max(0, center_y - rf_h), min(self.image_size, center_x + rf_w))
        bl = (min(self.image_size, center_y + rf_h), max(0, center_x - rf_w))
        br = (min(self.image_size, center_y + rf_h), min(self.image_size, center_x + rf_w))
        
        return np.recarray([tl, tr, bl, br], dtype=[("tl", [("y", np.float32), ("x", np.float32)]),
                                                    ("tr", [("y", np.float32), ("x", np.float32)]),
                                                    ("bl", [("y", np.float32), ("x", np.float32)]),
                                                    ("br", [("y", np.float32), ("x", np.float32)])]))

    def calculate_patch_locations(self):
        """Calculate the real world coordinates of every feature"""
        assert self.contains_features, "Can only compute patch locations if there are patches"

        ilu = ImageLocationUtility()

        logger.info("Calculating locations of every patch")
        n, h, w = self.locations.shape
        
        rfs = np.zeros((h, w), dtype=[("tl", [("y", np.float32), ("x", np.float32)]),
                                      ("tr", [("y", np.float32), ("x", np.float32)]),
                                      ("bl", [("y", np.float32), ("x", np.float32)]),
                                      ("br", [("y", np.float32), ("x", np.float32)])]))
        
        for (y, x) in np.ndindex((h, w)):
            rf = self.calculate_receptive_field(y + 0.5, x + 0.5)
            image_locations[y, x] = rf
        
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

    isview = property(lambda self: np.shares_memory(self, self.root))

    def get_batch(self, time, temporal_batch_size):
        time_index = np.argwhere(self.root.times == time).flat[0]
        res = None
        for res_i, arr_i in enumerate(range(time_index - temporal_batch_size, time_index)):
            image = cv2.cvtColor(self.root[max(0, arr_i), 0, 0].get_image(), cv2.COLOR_BGR2RGB)
            if res is None:
                res = np.zeros((temporal_batch_size,) + image.shape)
            res[res_i,...] = image
        return res

    def to_temporal_dataset(self, temporal_batch_size=16):
        import tensorflow as tf

        def _gen():
            for i in range(self.shape[0]):
                temporal_batch = self.get_batch(self[i, 0, 0].times, temporal_batch_size)
                yield (temporal_batch, self[i, 0, 0].times)

        raw_dataset = tf.data.Dataset.from_generator(
            _gen,
            output_types=(tf.uint8, tf.int64),
            output_shapes=((None, None, None, None), ()))

        return raw_dataset.prefetch(tf.data.experimental.AUTOTUNE)

if __name__ == "__main__":
    from common import Visualize
    from scipy.misc import imresize

    patches = PatchArray().anomaly

    vis = Visualize(patches)
    vis.show()
    for p in patches:
        from feature_extractor.Models.C3D.sports1M_utils import preprocess_input_python
        b = preprocess_input_python(patches.get_batch(p.times[0, 0], 16))
        im = numpy.concatenate([np.concatenate(b[4*i:4*i+4,...], axis=1) for i in range(4)], axis=0)
        im = im.astype(np.uint8)
        cv2.imshow("Batch", im)
        cv2.waitKey(1)

    # f = np.zeros(patches.shape[0], dtype=np.bool)
    # f[0:3] = True
    # f[5:10] = True
    # patches = patches[f]

    # print patches.isview

    # patches[0, 0, 0].labels = 2
    # print patches[0:5, 0, 0].labels
    # print patches[0:5, 0, 0].changed

    # patches[0:2, 0, 0].labels = 2

    # print patches[0:5, 0, 0].labels
    # print patches[0:5, 0, 0].changed

    # print patches.root[0:15, 0, 0].changed