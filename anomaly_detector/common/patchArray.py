import os
import time
import sys
import ast
import shutil
import traceback
from glob import glob
from cachetools import cached, Cache, LRUCache
from datetime import datetime

import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

import numpy as np
import numpy.lib.recfunctions
import h5py
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize

from shapely.strtree import STRtree
from shapely.geometry import Polygon, box, Point
from shapely.prepared import prep

from scipy.ndimage.morphology import generate_binary_structure, grey_erosion, grey_dilation
from sklearn import metrics
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from joblib import Parallel, delayed

from common import utils, logger, ImageLocationUtility
import consts

class Patch(np.record):
    """A single feature vector (patch) with metadata."""

    # image_cache = LRUCache(maxsize=20*60*2)  # Least recently used cache for images

    # @cached(image_cache, key=lambda self, *args: self.times) # The cache should only be based on the timestamp
    def get_image(self, images_path=None):
        return cv2.imread(self.get_image_path(images_path))

    def get_image_path(self, images_path=None):
        if images_path is None: images_path = consts.IMAGES_PATH
        return os.path.join(images_path, "%i.jpg" % self.times)

    def __setattr__(self, attr, val):
        if attr in self.dtype.names:
            old_val = self.__getattribute__(attr)
            if not np.all(old_val == val): # Check if any value changed
                np.record.__setattr__(self, "changed", True)
                np.record.__setattr__(self, attr, val)
        else:
            np.record.__setattr__(self, attr, val)

class PatchArray(np.recarray):
    """Array with metadata. This is the central class of the anomaly detector
    and contains all feature vectors (patches) and/or images alongside their metadata.
    It is based on a numpy recarray which simplifies access to metadata."""
    
    __metadata_attrs__ = list()

    root = None

    def __new__(cls, filename=None, images_path=consts.IMAGES_PATH):
        """Array with metadata. This is the central class of the anomaly detector
        and contains all feature vectors (patches) and/or images alongside their metadata.

        It is based on a numpy recarray which simplifies access to metadata.

        Args:
            filename (str): Features file to read (*.h5). If None, only the images and their metadata are loaded.
            images_path (str): Path where images AND the metadata file "metadata_cache.h5" are located.

        Returns:
            A new PatchArray
        """
        if cls.root is not None:
            logger.warning("There is already a root PatchArray loaded.")
        
        assert os.path.exists(images_path) and os.path.isdir(images_path), "Path to images does not exist (%s)" % images_path

        filename_metadata = os.path.join(images_path, "metadata_cache.h5")
        assert os.path.exists(filename_metadata), "No metadata file called \"metadata_cache.h5\" in %s" % images_path

        logger.info("Reading metadata and features from: %s" % filename)

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

        contains_features     = False
        contains_locations    = False
        contains_patch_labels = False
        contains_bins         = {"fake_0.20": False, "fake_0.50": False, "fake_2.00": False, "0.20": False, "0.50": False, "2.00": False}
        rasterizations        = {"fake_0.20": None, "fake_0.50": None, "fake_2.00": None, "0.20": None, "0.50": None, "2.00": None}
        
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

                add = ["features", "locations", "fake_locations", "patch_labels"]

                def _add(x, y):
                    if not isinstance(y, h5py.Dataset):
                        return
                    if x in add or x.startswith("bins"):
                        patches_dict[x] = y
                    elif x.endswith("/mahalanobis_distances"):
                        n = x.replace("/mahalanobis_distances", "")
                        if "balanced_distribution" in y.parent.keys():
                            bd = y.parent["balanced_distribution"]
                            n = "%s/%i" % (n, bd.shape[0])
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

                if "fake_locations" in patches_dict.keys():
                    contains_locations = True
                else:
                    patches_dict["fake_locations"] = np.zeros(locations_shape, dtype=[("tl", [("y", np.float32), ("x", np.float32)]),
                                                                                 ("tr", [("y", np.float32), ("x", np.float32)]),
                                                                                 ("br", [("y", np.float32), ("x", np.float32)]),
                                                                                 ("bl", [("y", np.float32), ("x", np.float32)])])

                if "patch_labels" in patches_dict.keys():
                    contains_patch_labels = True
                else:
                    patches_dict["patch_labels"] = np.zeros(locations_shape, dtype=np.uint8)
                    patches_dict["patch_labels_values"] = np.zeros(locations_shape)

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

        obj._ilu = ImageLocationUtility()

        obj.filename        = filename
        obj.images_path     = images_path
        obj.receptive_field = receptive_field
        obj.image_size      = image_size
        obj.contains_features     = contains_features
        obj.contains_locations    = contains_locations
        obj.contains_bins         = contains_bins
        obj.contains_patch_labels = contains_patch_labels
        obj.rasterizations        = rasterizations
        obj.contains_mahalanobis_distances = contains_mahalanobis_distances

        cls.root = obj

        return obj

    def __array_finalize__(self, obj):
        """Only for internal use"""
        if obj is None: return
        self._ilu = getattr(obj, "_ilu", None)
        self.filename              = getattr(obj, "filename", None)
        self.images_path           = getattr(obj, "images_path", consts.IMAGES_PATH)
        self.receptive_field       = getattr(obj, "receptive_field", None)
        self.image_size            = getattr(obj, "image_size", 224)
        self.contains_features     = getattr(obj, "contains_features", False)
        self.contains_locations    = getattr(obj, "contains_locations", False)
        self.contains_bins         = getattr(obj, "contains_bins", {"0.20": False, "0.50": False, "2.00": False})
        self.contains_patch_labels = getattr(obj, "contains_patch_labels", False)
        self.rasterizations        = getattr(obj, "rasterizations", {"0.20": None, "0.50": None, "2.00": None})
        self.contains_mahalanobis_distances = getattr(obj, "contains_mahalanobis_distances", False)
    
    def __setattr__(self, attr, val):
        """Keep track if metadata is changed"""
        if self.dtype.names is not None and attr in self.dtype.names:
            # Try finding the attribute in the metadata (will throw if it is not in metadata)
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
        """Cast patches to the correct class"""
        obj = np.recarray.__getitem__(self, indx)

        if isinstance(obj, np.record):
            obj.__class__ = Patch
        
        return obj
    
    #########################
    # Commonly used subsets #
    # (unfortunately copies,#
    # not views)            #
    #########################

    def _filter(self, name, f):
        """Return a subset of this PatchArray

        Args:
            name (str): Metadata key
            f (func): Filter function

        Returns:
            A new PatchArray
        """
        return self[f(self[name][:, 0, 0])] if self.ndim == 3 else self[f(self[name][:])]

    unknown_anomaly = property(lambda self: self._filter("labels", lambda f: f == 0))
    no_anomaly      = property(lambda self: self._filter("labels", lambda f: f == 1))
    anomaly         = property(lambda self: self._filter("labels", lambda f: f == 2))
    
    stop_ok   = property(lambda self: self._filter("stop", lambda f: f == 0))
    stop_dont = property(lambda self: self._filter("stop", lambda f: f == 1))
    stop_do   = property(lambda self: self._filter("stop", lambda f: f == 2))
    
    direction_unknown = property(lambda self: self._filter("directions", lambda f: f == 0))
    direction_ccw     = property(lambda self: self._filter("directions", lambda f: f == 1))
    direction_cw      = property(lambda self: self._filter("directions", lambda f: f == 2))
    
    round_number_unknown = property(lambda self: self._filter("round_numbers", lambda f: f == 0))
    def round_number(self, round_number):
        return self._filter("round_numbers", lambda f: f == round_number)
    
    # @property
    # def training_and_validation(self):
    #     """ Subset of training and validation frames (FOR COMPARING ALL EXTRACTORS AND MODELS) """
    #     p = self.root[::6, 0, 0]

    #     f = np.zeros(p.shape, dtype=np.bool)
    #     f[:] = np.logical_and(p.directions == 1,                                   # CCW and
    #                           np.logical_or(p.labels == 2,                         #   Anomaly or
    #                                         np.logical_and(p.round_numbers >= 7,   #     Round between 7 and 9
    #                                                         p.round_numbers <= 7)))

        
    #     f[:] = np.logical_and(f[:], np.logical_and(p.camera_locations.translation.x > 20,
    #                         np.logical_and(p.camera_locations.translation.x < 25,
    #                         np.sqrt((p.camera_locations.rotation.z - (np.pi / 2)) ** 2) < 0.15)))

    #     # Let's make contiguous blocks of at least 10, so
    #     # we can do some meaningful temporal smoothing afterwards
    #     for i, b in enumerate(f):
    #         if b and i - 20 >= 0:
    #             f[i - 20:i] = True
        
    #     return self.root[::6,...][f]

    @property
    def training_and_validation(self):
        """ Subset of training and validation frames (FOR CHECKING THE BEST EXTRACTORS AND MODELS) """
        p = self.root[..., 0, 0]

        # f = np.zeros(p.shape, dtype=np.bool)
        # f[::6] = True
        # f[:] = np.logical_or(np.logical_and(f, p.stop != 0), p.stop == 2)

        # f[:] = np.logical_and(f[:], np.sqrt((p.camera_locations.rotation.z + (np.pi / 2)) ** 2) < 0.15)

        return self.root[::6,...]

    @property
    def training(self):
        return self.direction_ccw

    @property
    def validation(self):
        return self.direction_cw
        # round_number = 7
        # if self.ndim == 3:
        #     return self[self.round_numbers[:, 0, 0] != round_number].direction_ccw
        # else:
        #     return self[self.round_numbers[:] != round_number].direction_ccw

    @property
    def benchmark(self):
        return self[0:10]

    metadata_changed = property(lambda self: self[self.changed[:, 0, 0]] if self.ndim == 3 else self[self.changed[:]])

    def save_metadata(self, filename=None):
        """Save all metadata that changed. Will always create a backup.

        Args:
            filename (str): New metadata file (default: currently opened metadata file)

        Returns:
            success (bool)
        """
        if filename is None:
            filename = os.path.join(self.images_path, "metadata_cache.h5")
        try:
            if os.path.exists(filename):
                shutil.copyfile(filename, "%s_backup_%s" % (filename, datetime.now().strftime("%d_%m_%Y_%H_%M_%S")))
            with h5py.File(filename, "r+") as hf:
                hf.attrs["Last changed"] = datetime.now().strftime("%d.%m.%Y, %H:%M:%S")

                indices = np.argwhere(np.isin(hf["times"], self.metadata_changed[:, 0, 0].times))
                for index, frame in zip(indices, self.metadata_changed[:, 0, 0]):
                    for meta in self.__metadata_attrs__:
                        hf[meta][index] = frame[meta]
            return True
        except:
            logger.error(traceback.format_exc())
            return False

    def extract_current_patches(self):
        """Create a new metadata file and copy images to a new subfolder

        Returns:
            success (bool)
        """
        folder = os.path.join(consts.BASE_PATH, "Images_subset_%s" % datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
        os.mkdir(folder)

        try:
            for i in tqdm(range(self.shape[0]), desc="Copying images", file=sys.stderr):
                image = self[i, 0, 0].get_image_path()
                shutil.copyfile(image, os.path.join(folder, os.path.basename(image)))

            with h5py.File(os.path.join(folder, "metadata_cache.h5"), "w") as hf:
                hf.attrs["Last changed"] = datetime.now().strftime("%d.%m.%Y, %H:%M:%S")
                for meta in self.__metadata_attrs__:
                    hf[meta] = self[:,0,0][meta]
            return True
        except:
            logger.error(traceback.format_exc())
            return False

    ###################
    # Spatial binning #
    ###################

    def _calculate_grid(self, cell_size, fake=False):
        """Calculate the cells for spatial binning

        Args:
            cell_size (float): Spatial bin size
            fake (bool): Use simple non-overlapping receptive field

        Returns:
            grid (STRtree): Search tree with all (rectangle) cells
            shape (int, int): Grid shape
        """
        key = "%.2f" % cell_size
        if fake: key = "fake_" + key

        # Get extent
        x_min, y_min, x_max, y_max = self.get_extent(cell_size, fake=fake)

        # Create the bins
        bins_y = np.arange(y_min, y_max, cell_size)
        bins_x = np.arange(x_min, x_max, cell_size)
        bin_area = cell_size * cell_size

        shape = (len(bins_y), len(bins_x))

        # Create the grid
        self.rasterizations[key] = np.zeros(shape, dtype=object)
        
        for v, y in enumerate(bins_y):
            for u, x in enumerate(bins_x):
                b = box(x, y, x + cell_size, y + cell_size)#Point(x + cell_size / 2, y + cell_size / 2)# 
                b.u = u
                b.v = v
                b.patches = list()
                self.rasterizations[key][v, u] = b
            
        # Create a search tree of spatial boxes
        grid = STRtree(self.rasterizations[key].ravel().tolist())
        
        return (grid, shape)

    def _bin(self, i, grid, shape, rf_factor, key, fake, cell_size):
        """Calculate the corresponding spatial bins for the patches given defined by index i.

        Args:
            i (int): Frame index
            grid (STRtree): Search tree with all (rectangle) cells
            rf_factor (float): Receptive field size divided by image size
            key (str): Metadata key where the bin information is stored
            fake (bool): Use simple non-overlapping receptive field
            cell_size (float): Spatial bin size

        Returns:
            None
        """
        locations_key = "locations"
        if fake: locations_key = "fake_" + locations_key

        for y, x in np.ndindex(self.shape[1:]):
            # Calculate a new intersection
            if fake or rf_factor < 2 or (y, x) == (0, 0):
                f = self[i, y, x]
                poly = Polygon([(f[locations_key].tl.x, f[locations_key].tl.y),
                                (f[locations_key].tr.x, f[locations_key].tr.y),
                                (f[locations_key].br.x, f[locations_key].br.y),
                                (f[locations_key].bl.x, f[locations_key].bl.y)])
                
                # This will quickly find the possible intersection candidates
                bins = grid.query(poly)

                # Calculate the bins that really intersect the polygon
                bins = filter(poly.intersects, bins)
                
                # if len(bins) == 0:
                #     bins = [grid.nearest(poly)]

            def _loop():
                for b in bins:
                    # weight = 1.0#b.intersection(poly).area / bin_area
                    b.patches.append(np.ravel_multi_index((i, y, x), self.shape))
                    yield np.ravel_multi_index((b.v, b.u), shape)
                
            self[i, y, x]["bins_" + key] = np.array(list(_loop()), dtype=np.uint32)

    def _save_rasterization(self, key, grid, shape, start=None, end=None):
        """Save the spatial binning result to the currently opened features file

        Args:
            key (str): Metadata key where the bin information is stored
            grid (STRtree): Search tree with all (rectangle) cells
            shape (int, int): Grid shape
            start (int): Start timestamp
            end (int): End timestamp

        Returns:
            None
        """
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

            if start is not None and end is not None:
                hf["rasterization_" + key].attrs["Start"] = start
                hf["rasterization_" + key].attrs["End"] = end
                hf["rasterization_" + key].attrs["Duration"] = end - start
                hf["rasterization_" + key].attrs["Duration (formatted)"] = utils.format_duration(end - start)

    def calculate_rasterization(self, cell_size, fake=False):
        """Calculate the corresponding spatial bins for each patch.

        Args:
            cell_size (float): Spatial bin size
            fake (bool): Use simple non-overlapping receptive field

        Returns:
            None
        """
        key = "%.2f" % cell_size
        if fake: key = "fake_" + key

        # Check if cell size is already calculated
        if key in self.contains_bins.keys() and self.contains_bins[key]:
            return self["bins_" + key]
        
        grid, shape = self._calculate_grid(cell_size, fake=fake)

        rf_factor = self.receptive_field[0] / self.image_size

        logger.info("%i bins in x and %i bins in y direction (with cell size %.2f)" % (shape + (cell_size,)))

        start = time.time()
        
        # Get the corresponding bin for every feature
        Parallel(n_jobs=2, prefer="threads")(
            delayed(self._bin)(i, grid, shape, rf_factor, key, fake, cell_size) for i in tqdm(range(self.shape[0]), desc="Calculating bins", file=sys.stderr))

        end = time.time()

        self._save_rasterization(key, grid, shape, start, end)
        
        self.contains_bins[key] = True
        self.rasterizations[key] = np.vectorize(lambda b: b.patches, otypes=[object])(self.rasterizations[key])

        return shape

    def get_extent(self, cell_size=None, fake=False):
        """Calculates the extent of the features
        
        Args:
            cell_size (float): Round to cell size (increases bounds to fit next cell size)
            fake (bool): Use simple non-overlapping receptive field
        
        Returns:
            Tuple (x_min, y_min, x_max, y_max)
        """
        key = "locations"
        if fake: key = "fake_" + key

        assert self.contains_locations, "Can only compute extent if there are patch locations"
        
        # Get the extent
        x_min = min(self[key].tl.x.min(), self[key].tr.x.min(), self[key].br.x.min(), self[key].bl.x.min())
        y_min = min(self[key].tl.y.min(), self[key].tr.y.min(), self[key].br.y.min(), self[key].bl.y.min())
        x_max = max(self[key].tl.x.max(), self[key].tr.x.max(), self[key].br.x.max(), self[key].bl.x.max())
        y_max = max(self[key].tl.y.max(), self[key].tr.y.max(), self[key].br.y.max(), self[key].bl.y.max())
        
        # Increase the extent to fit the cell size
        if cell_size is not None:
            x_min -= x_min % cell_size
            y_min -= y_min % cell_size
            x_max += cell_size - (x_max % cell_size)
            y_max += cell_size - (y_max % cell_size)
        
        return (x_min, y_min, x_max, y_max)

    #################
    #   Locations   #
    #################
    
    def calculate_receptive_field(self, y, x, scale_y=1.0, scale_x=1.0, fake=False):
        """Calculate the receptive field of a patch

        Args:
            y, x (int): Patch indices
            scale_y, scale_x (float): Scale factor
            fake (bool): Use simple non-overlapping receptive field

        Returns:
            Tuple (tl, tr, br, bl) with pixel coordinated of the receptive field
        """
        image_h = self.image_size * scale_y
        image_w = self.image_size * scale_x

        _, h, w = self.locations.shape
        center_y = y / float(h) * image_h
        center_x = x / float(w) * image_w

        if fake:
            rf_h = self.image_size / float(self.shape[1]) * scale_y / 2.0
            rf_w = self.image_size / float(self.shape[2]) * scale_x / 2.0
        else:
            rf_h = self.receptive_field[0] * scale_y / 2.0
            rf_w = self.receptive_field[1] * scale_x / 2.0

        
        tl = (max(0, center_y - rf_h), max(0, center_x - rf_w))
        tr = (max(0, center_y - rf_h), min(image_w, center_x + rf_w))
        br = (min(image_h, center_y + rf_h), min(image_w, center_x + rf_w))
        bl = (min(image_h, center_y + rf_h), max(0, center_x - rf_w))
        
        return (tl, tr, br, bl)

    def _get_receptive_fields(self, fake=False):
        """ Get the receptive fields for each patch (in image coordinates)

        Args:
            fake (bool): Use simple non-overlapping receptive field

        Returns:
            image_locations (np.array): Receptive fields in image coordinates (h, w, 4, 2)
        """
        n, h, w = self.locations.shape
        
        image_locations = np.zeros((h, w, 4, 2), dtype=np.float32)
        
        for (y, x) in np.ndindex((h, w)):
            rf = self.calculate_receptive_field(y + 0.5, x + 0.5, fake=fake)
            image_locations[y, x, 0] = rf[0]
            image_locations[y, x, 1] = rf[1]
            image_locations[y, x, 2] = rf[2]
            image_locations[y, x, 3] = rf[3]
        
        return image_locations

    def _get_centers(self):
        """ Get the center for each patch (in image coordinates)

        Returns:
            image_locations (np.array): Image centers in image coordinates (h, w, 2)
        """
        n, h, w = self.locations.shape
        
        image_locations = np.zeros((h, w, 2), dtype=np.float32)
        
        for (y, x) in np.ndindex((h, w)):
            image_locations[y, x] = (y + 0.5, x + 0.5)
        
        return image_locations

    def _image_to_relative(self, image_locations):
        """Convert image coordinates to relative coordinates

        Args:
            image_locations (np.array): Input in image coordinates

        Returns:
            relative_locations (np.array): Input in relative coordinates
        """
        return self._ilu.image_to_relative(image_locations, image_width=self.image_size, image_height=self.image_size)         # (h, w, 4, 2)
    
    def _relative_to_absolute(self, relative_locations, camera_locations):
        """Convert relative coordinates to absolute coordinates

        Args:
            relative_locations (np.array): Input in relative coordinates
            camera_locations (np.array): Respective camera locations

        Returns:
            absolute_locations (np.array): Input in absolute coordinates
        """
        res = self._ilu.relative_to_absolute(relative_locations, camera_locations)
        res = np.rec.fromarrays(res.transpose(), dtype=[("y", np.float32), ("x", np.float32)]).transpose()
        return np.rec.fromarrays(res.transpose(), dtype=self.locations.dtype) # No transpose here smh

    def _save_patch_locations(self, key, start=None, end=None):
        """Save the patch locations to the currently opened features file

        Args:
            key (str): Metadata key where the location information is stored
            start (int): Start timestamp
            end (int): End timestamp

        Returns:
            None
        """
        with h5py.File(self.filename, "r+") as hf:
            # Remove the old locations dataset
            if key in hf.keys():
                del hf[key]
            
            hf.create_dataset(key, data=self[key])
            
            if start is not None and end is not None:
                hf[key].attrs["Start"] = start
                hf[key].attrs["End"] = end
                hf[key].attrs["Duration"] = end - start
                hf[key].attrs["Duration (formatted)"] = utils.format_duration(end - start)

    def calculate_patch_locations(self, fake=False):
        """Calculate the real world coordinates of every feature vector (patch)

        Args:
            fake (bool): Use simple non-overlapping receptive field

        Returns:
            None
        """
        key = "locations"
        if fake: key = "fake_" + key

        assert self.contains_features, "Can only compute patch locations if there are patches"

        logger.info("Calculating locations of every patch")
        
        start = time.time()
        image_locations = self._get_receptive_fields(fake)
        
        relative_locations = self._image_to_relative(image_locations)
        
        for i in tqdm(range(self[key].shape[0]), desc="Calculating locations", file=sys.stderr):
            self[key][i] = self._relative_to_absolute(relative_locations, self[i, 0, 0].camera_locations)

        end = time.time()

        self.contains_locations = True

        self._save_patch_locations(key, start, end)

    def calculate_patch_center_locations(self):
        """Calculate the real world coordinates of the center of every feature vector (patch)

        *This is currently not in use, but was intended for the spatial binning
        variant that only uses the central bin for anomaly detection*
        
        Returns:
            None
        """
        key = "locations_center"

        assert self.contains_features, "Can only compute patch locations if there are patches"

        logger.info("Calculating locations of every patch")
        
        start = time.time()
        image_locations = self._get_centers()
        
        relative_locations = self._image_to_relative(image_locations)
        
        for i in tqdm(range(self[key].shape[0]), desc="Calculating locations", file=sys.stderr):
            self[key][i] = self._relative_to_absolute(relative_locations, self[i, 0, 0].camera_locations)

        end = time.time()

        self._save_patch_locations(key, start, end)

    #################
    # Calculations  #
    #################
    
    def var(self):
        """Calculate the variance"""
        return np.var(self.ravel().features, axis=0, dtype=np.float64)

    def cov(self):
        """Calculate the covariance matrix"""
        return np.cov(self.ravel().features, rowvar=False)

    def mean(self):
        """Calculate the mean"""
        return np.mean(self.ravel().features, axis=0, dtype=np.float64)

    #################
    #      Misc     #
    #################
    
    def to_dataset(self):
        """Returns a TensorFlow Dataset of all contained images
        
        Returns:
            TensorFlow Dataset with all images in this PatchArray
        """
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

    def get_batch(self, frame, temporal_batch_size):
        """Gets a temporal batch for a given frame by 

        Args:
            frame (PatchArray): Frame to get the temporal batch for
            temporal_batch_size (int): Number of frames in the batch
        
        Returns:
            np.ndarray with the frames
        """
        # Only take patches from the current round (no jumps from the end of the last round).
        # Use the root array, so every frame is considered (e.g. no FPS reduction on root array)
        current_round = self.root.round_number(frame.round_numbers)
        time_index = np.argwhere(current_round.times == frame.times).flat[0]
        res = None
        for res_i, arr_i in enumerate(range(time_index - temporal_batch_size, time_index)):
            # Get and convert the image
            image = cv2.cvtColor(current_round[max(0, arr_i), 0, 0].get_image(), cv2.COLOR_BGR2RGB)
            if res is None:
                res = np.zeros((temporal_batch_size,) + image.shape)
            res[res_i,...] = image
        return res

    def to_temporal_dataset(self, temporal_batch_size=16):
        """Returns a TensorFlow Dataset of all contained images with temporal batches

        Args:
            temporal_batch_size (int): Number of frames in the batch
        
        Returns:
            TensorFlow Dataset with temporal batches of all images in this PatchArray
        """
        import tensorflow as tf

        def _gen():
            for i in range(self.shape[0]):
                temporal_batch = self.get_batch(self[i, 0, 0], temporal_batch_size)
                yield (temporal_batch, self[i, 0, 0].times)

        raw_dataset = tf.data.Dataset.from_generator(
            _gen,
            output_types=(tf.uint8, tf.int64),
            output_shapes=((None, None, None, None), ()))

        return raw_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    #################
    #    Metrics    #
    #################
    
    # def _get_stop_labels(self):
    #     slack = 5
    #     labels = self.stop[:,0,0].copy()
    #     f = labels == 2
    #     for i, b in enumerate(labels):
    #         if b and i + slack < labels.size:
    #             labels[i:i + slack] = 0
    #     labels[f] = 2

    class Metric(object):
        """Helper class for the calculation of different metrics"""
        COLORS = {
            -1: (100, 100, 100),
            0: (150, 150, 150),
            1: (80, 175, 76),
            2: (54, 67, 244)
        }
        
        def __init__(self, name, label_name, per_patch=False, mean=False, names=None, colors=None):
            """Create a new metric
            
            Args:
                name (str): Metric name
                label_name (str): Metadata key for the used label
                per_patch (bool): Calculate metric for each patch (referred to as "per-pixel" in thesis)
                mean (bool): True: Take the mean of anomaly scores per frame. False: Use the maximum AD score
                names (dict): Dictionary mapping the label values to readable names
                colors (dict): Dictionary mapping the label values to colors

            Returns:
                A new Metric
            """
            self.name = name
            self.label_name = label_name
            self.per_patch = per_patch
            self.mean = mean

            self.names = names if names is not None else {
                0: "Unknown",
                1: "No anomaly",
                2: "Anomaly"
            }

            self.current_threshold = -1

        def get_relevant(self, patches):
            """Get all patches that will be considered (label value != 0)"""
            return patches[patches[self.label_name][:,0,0] != 0]

        def get_labels(self, patches):
            """Get all labels"""
            if self.per_patch:
                return patches[self.label_name].ravel()
            else:
                return patches[self.label_name][:,0,0]
        
        def get_values(self, mahalanobis_distances):
            """Get all anomaly scores"""
            if self.per_patch:
                return mahalanobis_distances.ravel()
            elif self.mean:
                return np.mean(mahalanobis_distances, axis=(1,2))
            else:
                return np.max(mahalanobis_distances, axis=(1,2))
    
    METRICS = [
        Metric("patch",        "patch_labels", per_patch=True),
        Metric("frame (mean)", "labels", mean=True),
        Metric("frame (max)",  "labels"),
        Metric("stop (mean)",  "stop", mean=True, names={-1: "Not set", 0: "It's OK to stop", 1: "Don't stop", 2: "Stop"}),
        Metric("stop (max)",   "stop", names={-1: "Not set", 0: "It's OK to stop", 1: "Don't stop", 2: "Stop"})
    ]

    def calculate_tsne(self):
        """Calculate and visualize a t-SNE (DEPRECATED)"""
        assert self.contains_mahalanobis_distances, "Can't calculate t-SNE without mahalanobis distances calculated"

        # TODO: Maybe only validation?
        features = self.features.reshape(-1, self.features.shape[-1])

        feat_cols = ["feature" + str(i) for i in range(features.shape[1])]

        df = pd.DataFrame(features, columns=feat_cols)
        df["maha"] = self.mahalanobis_distances.SVG.ravel()
        df["l"] = self.patch_labels.ravel()
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
        logger.info("t-SNE done! Time elapsed: {} seconds".format(time.time() - time_start))
        
        df_subset["tsne-2d-one"] = tsne_results[:,0]
        df_subset["tsne-2d-two"] = tsne_results[:,1]
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

    def calculate_metrics(self):
        """Calculate the metrics for all considered variants using the features in this PatchArray"""
        assert self.contains_mahalanobis_distances, "Can't calculate ROC without mahalanobis distances calculated"
        # (Name, ROC_AUC, AUC_PR, f1)
        results = list()

        extractor = os.path.basename(self.filename).replace(".h5", "")

        gauss_filters = [None,    (0,1,1), (0,2,2),
                         (1,0,0), (1,1,1), (1,2,2)]
        # gauss_filters = [None,    (0,1,1), (0,2,2), (0,3,3),
        #                  (1,0,0), (1,1,1), (1,2,2), (1,3,3),
        #                  (2,0,0), (2,1,1), (2,2,2), (2,3,3)]
        other_filters = [None, "erosion", "dilation", "erosion+dilation"]

        val = self.validation

        for metric in self.METRICS:
            relevant = metric.get_relevant(val)
            labels = metric.get_labels(relevant)
            for other_filter in other_filters:
                for gauss_filter in gauss_filters:
                    # Don't compute gauss filters (in image space) for mean metrics (they take the average anyways)
                    if metric.mean and gauss_filter != None and gauss_filter[1] > 0:
                        continue

                    title = "Metrics for %s (%s, filter:%s + %s)" % (extractor, metric.name, gauss_filter, other_filter)
                    logger.info("Calculating %s" % title)
                    
                    scores = dict()
                    for n in sorted(self.mahalanobis_distances.dtype.names):
                        name = n.replace("fake", "simple")

                        maha = relevant.mahalanobis_distances[n]

                        if gauss_filter is not None:
                            maha = utils.gaussian_filter(maha, sigma=gauss_filter)
                        
                        if other_filter is not None:
                            struct = generate_binary_structure(2, 1)
                            if struct.ndim == 2:
                                z = np.zeros_like(struct, dtype=np.bool)
                                struct = np.stack((z, struct, z))
                            
                            if other_filter == "erosion":
                                maha = grey_erosion(maha, structure=struct)
                            elif other_filter == "dilation":
                                maha = grey_dilation(maha, structure=struct)
                            elif other_filter == "erosion+dilation":
                                maha = grey_erosion(maha, structure=struct)
                                maha = grey_dilation(maha, structure=struct)

                        scores[name] = metric.get_values(maha)
                        
                    filename = os.path.join(consts.METRICS_PATH, "%s_%s_%s_%s.jpg" % (extractor, metric.name, gauss_filter, other_filter))
                    result = self.__calculate_roc__(title, labels, scores, filename)
                    for model, roc_auc, auc_pr, max_f1, fpr0, fpr1, fpr2, fpr3, fpr4, fpr5 in result:
                        results.append((extractor, metric.name, model, gauss_filter, other_filter, roc_auc, auc_pr, max_f1, fpr0, fpr1, fpr2, fpr3, fpr4, fpr5))
        
        return results


    def __calculate_roc__(self, title, labels, scores, filename=None):
        """ Calculate the metrics and create ROC and PR diagrams """
        # (Name, ROC_AUC, AUC_PR, f1)
        results = list()

        no_skill = labels[labels == 2].size / float(labels.size)

        dpi = 96 if filename is None else 300

        fig, ax = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
        fig.suptitle(title)

        lw = 1
        
        ax[0].plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--", label="No skill", alpha=0.5)

        f_scores = np.linspace(0.2, 0.8, num=4)
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = ax[1].plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            ax[1].annotate("f1=%.1f" % f_score, xy=(0.85, y[45] + 0.02), color="gray")

        ax[1].plot([0, 1], [no_skill, no_skill], color="navy", lw=lw, linestyle="--", alpha=0.5)
        
        tprs = [0.9, 0.95, 0.99, 0.995, 0.999, 0.9999]

        # with h5py.File(self.filename, "r+") as hf:
        for name, score in scores.items():
            if name.startswith("BalancedDistribution"):
                if not "/%i/" % int(np.mean(self.mahalanobis_distances["SVG"])) in name:
                    continue
                name = "BalancedDistributionSVG"

            fpr, tpr, thresholds_roc = metrics.roc_curve(labels, score, pos_label=2)
            roc_auc = metrics.auc(fpr, tpr)
            
            precision, recall, thresholds_pr = metrics.precision_recall_curve(labels, score, pos_label=2)
            auc_pr = metrics.auc(recall, precision)
            f1 = [2 * ((p * r) / (p + r)) if (p + r) != 0 else 0 for p, r in zip(precision, recall)]
            max_f1_index = np.argmax(f1)
            max_f1 = f1[max_f1_index]

            if "SpatialBinSingle" in name:
                if "simple" in name:
                    if ".50" in name:
                        color = "#90CAF9"   # blue lighten-3
                    else:
                        color = "#2196F3"   # blue
                else:
                    if ".50" in name:
                        color = "#CE93D8"   # purple lighten-3
                    else:
                        color = "#9C27B0"   # purple
            elif "SpatialBin" in name:
                if "simple" in name:
                    if ".50" in name:
                        color = "#A5D6A7"   # green lighten-3
                    else:
                        color = "#4CAF50"   # green
                else:
                    if ".50" in name:
                        color = "#FFE082"   # amber lighten-3
                    else:
                        color = "#FFC107"   # amber
            # elif "MVG" in name:
            #     color = "#E91E63"   # pink  
            else:
                color = "#F44336"   # red  

            if "Balanced" in name:
                style = ":"    # dotted line style
            elif "MVG" in name:
                style = "--"    # dashed line style
            else:
                style = "-"     # solid line style
            

            max_f1_index_roc = (np.abs(thresholds_roc - thresholds_pr[max_f1_index])).argmin()
            l = ax[0].plot(fpr, tpr, linestyle=style, color=color, lw=lw, label=name)
            ax[0].plot(fpr[max_f1_index_roc], tpr[max_f1_index_roc], marker='o', markersize=5, color=color)

            ax[1].plot(recall, precision, linestyle=style, color=color, lw=lw)
            ax[1].plot(recall[max_f1_index], precision[max_f1_index], marker='o', markersize=5, color=color)

            fprs = tuple([fpr[(np.abs(tpr - t)).argmin()] for t in tprs])

            results.append((name, roc_auc, auc_pr, max_f1) + fprs)
            # ax[2].plot(f1, lw=lw)
                # hf[n].attrs["ROC_AUC"] = roc_auc
                # hf[n].attrs["AUC_PR"] = auc_pr
                # hf[n].attrs["Max. f1"] = max_f1
        
        ax[0].set_xlim([0.0, 1.0])
        ax[0].set_ylim([0.0, 1.03])
        ax[0].set_xlabel("False Positive Rate")
        ax[0].set_ylabel("True Positive Rate")
        ax[0].set_title("ROC curve")
        ax[0].grid(True, color="lightgray")
        
        ax[1].set_xlim([0.0, 1.0])
        ax[1].set_ylim([0.0, 1.03])
        ax[1].set_xlabel("Recall")
        ax[1].set_ylabel("Precision")
        ax[1].set_title("Precision/recall curve")
        ax[1].grid(True, color="lightgray")
        
        # ax[2].set_xlabel("Threshold index")
        # ax[2].set_ylabel("f1 score")
        # ax[2].set_title("f1 score")
        # ax[2].legend(loc="upper right")
        
        fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.3)

        handles, labels = ax[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='lower center', prop={'size': 5}, ncol=2)

        # # get the width of your widest label, since every label will need 
        # # to shift by this amount after we align to the right
        # shift = max([t.get_window_extent(renderer=fig.canvas.get_renderer()).width for t in legend.get_texts()])
        # for t in legend.get_texts():
        #     t.set_ha('right') # ha is alias for horizontalalignment
        #     t.set_position((shift,0))

        if filename is not None:
            
            plt.savefig(filename, dpi=dpi)
        else:
            plt.show()
        
        plt.close(fig)

        return results

    def calculate_patch_labels(self):
        """Calculate per-pixel labels from annotation images (red=anomaly).
        Annotated images need to be in a folder called "Labels" at the same level as the "Images" folder.
        """
        if not self.contains_features or self.filename is None:
            logger.error("Can only do this on feature files")
            return
        
        # Reset to unknown
        self.patch_labels.fill(0)

        for i in tqdm(range(self.root.shape[0]), desc="Calculating patch labels", file=sys.stderr):
            frame = self.root[i, 0, 0]
            mask_file = frame.get_image_path().replace("Images", "Labels")
            if os.path.exists(mask_file):
                mask = numpy.array(cv2.imread(mask_file), dtype=np.uint8)
                mask = (mask[..., 0] <= 5) & (mask[..., 1] <= 5) & (mask[..., 2] >= 250)
                self.root.patch_labels[i, ...] = (resize(mask, self.shape[1:], order=0, anti_aliasing=False, mode="constant") > 0.5) + 1

                # mask = resize(mask, (self.image_size, self.image_size), order=0, anti_aliasing=False, mode="constant")

                # for y, x in np.ndindex(self.shape[1:]):
                #     rf = self.calculate_receptive_field(y, x, fake=True)
                #     mean = mask[int(rf[0][0]):int(rf[2][0]), int(rf[0][1]):int(rf[2][1])].mean()
                #     self.patch_labels[i, y, x] = 2 if (mean > 0.5) else 1
                #     self.patch_labels_values[i, y, x] = mean
                
                # fig, axs = plt.subplots(1, 3, constrained_layout=True)
                # axs[0].imshow(mask)
                # axs[0].set_title("Mask")

                # axs[1].imshow(self.patch_labels[i, ...])
                # axs[1].set_title("Patch labels")

                # axs[2].imshow(resize(mask, self.shape[1:], order=1, anti_aliasing=True, mode="constant") > 0.5)
                # axs[2].set_title("Mask resized")

                # plt.show()
            else:
                self.patch_labels[i, ...] = frame.labels

        with h5py.File(self.filename, "r+") as hf:
            # Remove the old locations dataset
            if "patch_labels" in hf.keys():
                del hf["patch_labels"]
            
            hf.create_dataset("patch_labels", data=self.patch_labels)

if __name__ == "__main__":
    from common import Visualize
    
    p = PatchArray(consts.FEATURES_FILE)

    vis = Visualize(p)
    vis.show()