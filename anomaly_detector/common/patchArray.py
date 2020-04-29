import os
import time
import sys
import ast
from glob import glob
from cachetools import cached, Cache, LRUCache
from datetime import datetime
from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np
import numpy.lib.recfunctions
import h5py
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import matplotlib.pyplot as plt
import cv2

from shapely.strtree import STRtree
from shapely.geometry import Polygon, box
from shapely.prepared import prep

from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from joblib import Parallel, delayed

from common import utils, logger, ImageLocationUtility
import consts

class Patch(np.record):
    # image_cache = LRUCache(maxsize=20*60*2)  # Least recently used cache for images

    # @cached(image_cache, key=lambda self, *args: self.times) # The cache should only be based on the timestamp
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
        contains_bins      = {"0.20": False, "0.50": False, "2.00": False}
        rasterizations     = {"0.20": None, "0.50": None, "2.00": None}
        contains_mahalanobis_distances = False

        receptive_field = None
        image_size = None

        # Check if file is h5 file
        if isinstance(filename, str) and filename.endswith(".h5"):
            s = time.time()
            with h5py.File(filename, "r") as hf:
                logger.info("Opening %s: %f" % (filename, time.time() - s))

                receptive_field = hf.attrs.get("Receptive field", None)
                image_size = hf.attrs.get("Image size", None)

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
                                                                                 ("br", [("y", np.float32), ("x", np.float32)]),
                                                                                 ("bl", [("y", np.float32), ("x", np.float32)])])

                if len(mahalanobis_dict) > 0:
                    contains_mahalanobis_distances = True
                    t = [(x, mahalanobis_dict[x].dtype) for x in mahalanobis_dict]
                    patches_dict["mahalanobis_distances"] = np.rec.fromarrays(mahalanobis_dict.values(), dtype=t)
                    patches_dict["mahalanobis_distances_filtered"] = np.zeros(locations_shape, dtype=np.float64)
                
                for k in contains_bins.keys():
                    if ("bins_" + k) in patches_dict.keys():
                        contains_bins[k] = True
                        rasterizations[k] = np.array(hf["rasterization_" + k])
                    else:
                        contains_bins[k] = False
                        patches_dict["bins_" + k] = np.zeros(locations_shape, dtype=object)

                # Broadcast metadata to the correct shape
                if patches_dict["features"].shape[1:-1] != ():
                    for n, m in metadata.items():
                        patches_dict[n] = np.moveaxis(np.broadcast_to(m, patches_dict["features"].shape[1:-1] + (m.size,)), -1, 0)

                # Add indices as metadata
                # patches_dict["index"] = np.mgrid[0:locations_shape[0], 0:locations_shape[1], 0:locations_shape[2]]

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
        obj.rasterizations     = rasterizations
        obj.contains_mahalanobis_distances = contains_mahalanobis_distances

        cls.root = obj

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.filename           = getattr(obj, "filename", None)
        self.images_path        = getattr(obj, "images_path", consts.IMAGES_PATH)
        self.receptive_field    = getattr(obj, "receptive_field", None)
        self.image_size         = getattr(obj, "image_size", 224)
        self.contains_features  = getattr(obj, "contains_features", False)
        self.contains_locations = getattr(obj, "contains_locations", False)
        self.contains_bins      = getattr(obj, "contains_bins", {"0.20": False, "0.50": False, "2.00": False})
        self.rasterizations     = getattr(obj, "rasterizations", {"0.20": None, "0.50": None, "2.00": None})
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

    unknown_anomaly = property(lambda self: self[self.labels[:, 0, 0] == 0] if self.ndim == 3 else self[self.labels[:] == 0])
    no_anomaly      = property(lambda self: self[self.labels[:, 0, 0] == 1] if self.ndim == 3 else self[self.labels[:] == 1])
    anomaly         = property(lambda self: self[self.labels[:, 0, 0] == 2] if self.ndim == 3 else self[self.labels[:] == 2])
    
    direction_unknown = property(lambda self: self[self.directions[:, 0, 0] == 0] if self.ndim == 3 else self[self.directions[:] == 0])
    direction_ccw     = property(lambda self: self[self.directions[:, 0, 0] == 1] if self.ndim == 3 else self[self.directions[:] == 1])
    direction_cw      = property(lambda self: self[self.directions[:, 0, 0] == 2] if self.ndim == 3 else self[self.directions[:] == 2])
    
    round_number_unknown = property(lambda self: self[self.round_numbers[:, 0, 0] == 0] if self.ndim == 3 else self[self.round_numbers[:] == 0])
    def round_number(self, round_number):
        if self.ndim == 3:
            return self[self.round_numbers[:, 0, 0] == round_number]
        else:
            return self[self.round_numbers[:] == round_number]

    
    def training_and_validation(self):
        p = self.root[0::6, 0, 0]

        f = np.zeros(p.shape, dtype=np.bool)
        f[:] = np.logical_and(p.directions == 1,                                   # CCW and
                            np.logical_or(p.labels == 2,                         #   Anomaly or
                                            np.logical_and(p.round_numbers >= 7,   #     Round between 7 and 9
                                                            p.round_numbers <= 7)))

        
        f[:] = np.logical_and(f[:], np.logical_and(p.camera_locations.translation.x > 20,
                            np.logical_and(p.camera_locations.translation.x < 25,
                            np.sqrt((p.camera_locations.rotation.z - (np.pi / 2)) ** 2) < 0.15)))

        # Let's make contiguous blocks of at least 10, so
        # we can do some meaningful temporal smoothing afterwards
        for i, b in enumerate(f):
            if b and i - 20 >= 0:
                f[i - 20:i] = True
        
        return self.root[0::6,...][f]

    def training(self):
        round_number = 7
        label = 1
        if self.ndim == 3:
            return self[np.logical_and(self.round_numbers[:, 0, 0] == round_number, self.labels[:, 0, 0] == label)]
        else:
            return self[np.logical_and(self.round_numbers[:] == round_number, self.labels[:] == label)]

    def validation(self):
        round_number = 7
        if self.ndim == 3:
            return self[self.round_numbers[:, 0, 0] != round_number]
        else:
            return self[self.round_numbers[:] != round_number]

    metadata_changed = property(lambda self: self[self.changed[:, 0, 0]] if self.ndim == 3 else self[self.changed[:]])

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
        key = "%.2f" % cell_size
        # Check if cell size is already calculated
        if not key in self.contains_bins.keys() or not self.contains_bins[key]:
            self.calculate_rasterization(cell_size)
        
        return self.ravel()[self.rasterizations[key][bin]]

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
        key = "%.2f" % cell_size
        # Check if cell size is already calculated
        if key in self.contains_bins.keys() and self.contains_bins[key]:
            return self["bins_" + key]
        
        # Get extent
        x_min, y_min, x_max, y_max = self.get_extent(cell_size)

        # Create the bins
        bins_y = np.arange(y_min, y_max, cell_size)
        bins_x = np.arange(x_min, x_max, cell_size)
        bin_area = cell_size * cell_size

        shape = (len(bins_y), len(bins_x))

        # Create the grid
        self.rasterizations[key] = np.zeros(shape, dtype=object)
        
        for v, y in enumerate(bins_y):
            for u, x in enumerate(bins_x):
                b = box(y, x, y + cell_size, x + cell_size)
                b.u = u
                b.v = v
                b.patches = list()
                self.rasterizations[key][v, u] = b
            
        # Create a search tree of spatial boxes
        grid = STRtree(self.rasterizations[key].ravel().tolist())
        
        rf_factor = self.receptive_field[0] / self.image_size

        logger.info("%i bins in x and %i bins in y direction (with cell size %.2f)" % (shape + (cell_size,)))

        # @profile
        def _bin(i):
            for y, x in np.ndindex(self.shape[1:]):
                if rf_factor < 2 or (y, x) == (0, 0):
                    f = self[i, y, x]
                    poly = Polygon([f.locations.tl, f.locations.tr, f.locations.br, f.locations.bl])
                    # poly = poly.buffer(0.99)
                    bins = grid.query(poly)

                    # pr = prep(poly)
                    # bins = filter(pr.intersects, bins)
                    
                def _loop():
                    for b in bins:
                        # weight = 1.0#b.intersection(poly).area / bin_area
                        b.patches.append(np.ravel_multi_index((i, y, x), self.shape))
                        yield np.ravel_multi_index((b.v, b.u), shape)
                    
                self[i, y, x]["bins_" + key] = np.array(list(_loop()), dtype=np.uint32)

        start = time.time()
        
        # Get the corresponding bin for every feature
        Parallel(n_jobs=2, prefer="threads")(
            delayed(_bin)(i) for i in tqdm(range(self.shape[0]), desc="Calculating bins", file=sys.stderr))

        end = time.time()

        logger.info("Opening %s" % self.filename)

        # Save to file
        with h5py.File(self.filename, "r+") as hf:
            # Remove the old dataset
            if "bins_" + key in hf.keys():
                logger.info("Deleting old bins_%s from file" % key)
                del hf["bins_" + key]
            
            logger.info("Writing bins_%s to file" % key)
            hf.create_dataset("bins_" + key, data=self["bins_" + key], dtype=h5py.vlen_dtype(np.uint32))
            
            # Remove the old dataset
            if "rasterization_" + key in hf.keys():
                logger.info("Deleting old rasterization_%s from file" % key)
                del hf["rasterization_" + key]

            if "rasterization_" + key + "_count" in hf.keys():
                logger.info("Deleting old rasterization_%s_count from file" % key)
                del hf["rasterization_" + key + "_count"]
            
            logger.info("Writing rasterization_%s and rasterization_%s_count to file" % (key, key))
            rasterization       = hf.create_dataset("rasterization_" + key,            shape=shape, dtype=h5py.vlen_dtype(np.uint32))
            rasterization_count = hf.create_dataset("rasterization_" + key + "_count", shape=shape, dtype=np.uint16)

            for b in grid._geoms:
                rasterization[b.v, b.u] = b.patches
                rasterization_count[b.v, b.u] = len(b.patches)

            hf["rasterization_" + key].attrs["Start"] = start
            hf["rasterization_" + key].attrs["End"] = end
            hf["rasterization_" + key].attrs["Duration"] = end - start
            hf["rasterization_" + key].attrs["Duration (formatted)"] = utils.format_duration(end - start)

        self.contains_bins[key] = True
        self.rasterizations[key] = np.vectorize(lambda b: b.patches, otypes=[object])(self.rasterizations[key])

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
        x_min = min(self.locations.tl.x.min(), self.locations.tr.x.min(), self.locations.br.x.min(), self.locations.bl.x.min())
        y_min = min(self.locations.tl.y.min(), self.locations.tr.y.min(), self.locations.br.y.min(), self.locations.bl.y.min())
        x_max = max(self.locations.tl.x.max(), self.locations.tr.x.max(), self.locations.br.x.max(), self.locations.bl.x.max())
        y_max = max(self.locations.tl.y.max(), self.locations.tr.y.max(), self.locations.br.y.max(), self.locations.bl.y.max())
        
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

    def calculate_receptive_field(self, y, x, scale_y=1.0, scale_x=1.0):
        """Calculate the receptive field of a patch

        Args:
            y, x (int): Patch indices
            scale_y, scale_x (float): Scale factor

        Returns:
            Tuple (tl, tr, br, bl) with pixel coordinated of the receptive field
        """
        image_h = self.image_size * scale_y
        image_w = self.image_size * scale_x

        _, h, w = self.locations.shape
        center_y = y / float(h) * image_h
        center_x = x / float(w) * image_w

        rf_h = self.receptive_field[0] * scale_y / 2.0
        rf_w = self.receptive_field[1] * scale_x / 2.0
        
        tl = (max(0, center_y - rf_h), max(0, center_x - rf_w))
        tr = (max(0, center_y - rf_h), min(image_w, center_x + rf_w))
        br = (min(image_h, center_y + rf_h), min(image_w, center_x + rf_w))
        bl = (min(image_h, center_y + rf_h), max(0, center_x - rf_w))
        
        return (tl, tr, br, bl)

    def calculate_patch_locations(self):
        """Calculate the real world coordinates of every feature"""
        assert self.contains_features, "Can only compute patch locations if there are patches"

        ilu = ImageLocationUtility()

        logger.info("Calculating locations of every patch")
        n, h, w = self.locations.shape
        
        image_locations = np.zeros((h, w, 4, 2), dtype=np.float32)
        
        for (y, x) in np.ndindex((h, w)):
            rf = self.calculate_receptive_field(y + 0.5, x + 0.5)
            image_locations[y, x, 0] = rf[0]
            image_locations[y, x, 1] = rf[1]
            image_locations[y, x, 2] = rf[2]
            image_locations[y, x, 3] = rf[3]
        
        relative_locations = ilu.image_to_relative(image_locations, image_width=self.image_size, image_height=self.image_size)         # (h, w, 4, 2)
        
        for i in tqdm(range(n), desc="Calculating locations", file=sys.stderr):
            res = ilu.relative_to_absolute(relative_locations, self[i, 0, 0].camera_locations)
            res = np.rec.fromarrays(res.transpose(), dtype=[("y", np.float32), ("x", np.float32)]).transpose()
            res = np.rec.fromarrays(res.transpose(), dtype=self.locations.dtype) # No transpose here smh
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

    def calculate_tsne(self):
        features = self.features.reshape(-1, self.features.shape[-1])

        feat_cols = ['feature' + str(i) for i in range(features.shape[1])]

        df = pd.DataFrame(features, columns=feat_cols)
        df["maha"] = self.mahalanobis_distances.SVG.ravel()
        df["l"] = self.labels.ravel()
        df["label"] = df["l"].apply(lambda l: "Anomaly" if l == 2 else "No anomaly")

        # For reproducability of the results
        np.random.seed(42)
        rndperm = np.random.permutation(df.shape[0])

        N = 10000
        df_subset = df.loc[rndperm[:N],:].copy()

        data_subset = df_subset[feat_cols].values

        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=1)
        tsne_results = tsne.fit_transform(data_subset)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
        
        df_subset['tsne-2d-one'] = tsne_results[:,0]
        df_subset['tsne-2d-two'] = tsne_results[:,1]
        fig = plt.figure(figsize=(16,10))
        fig.suptitle(os.path.basename(self.filename).replace(".h5", ""), fontsize=20)

        LABEL_COLORS = {
            "No anomaly": "#4CAF50",     # No anomaly
            "Anomaly": "#F44336"      # Contains anomaly
        }

        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="label",
            palette=LABEL_COLORS,
            data=df_subset,
            legend="brief",
            alpha=0.4,
            size="maha"
        )

        # plt.show()
        plt.savefig(self.filename.replace(".h5", "_TSNE.png"))

if __name__ == "__main__":
    p = PatchArray(consts.FEATURES_FILE)

    p.calculate_tsne()

    plt.show()

    print "Finished"

    # p.calculate_rasterization(2.0)

    # from common import Visualize
    # from scipy.misc import imresize

    # patches = PatchArray().anomaly

    # vis = Visualize(p)
    # vis.show()
    # for p in patches:
    #     from feature_extractor.Models.C3D.sports1M_utils import preprocess_input_python
    #     b = preprocess_input_python(patches.get_batch(p.times[0, 0], 16))
    #     im = numpy.concatenate([np.concatenate(b[4*i:4*i+4,...], axis=1) for i in range(4)], axis=0)
    #     im = im.astype(np.uint8)
    #     cv2.imshow("Batch", im)
    #     cv2.waitKey(1)

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