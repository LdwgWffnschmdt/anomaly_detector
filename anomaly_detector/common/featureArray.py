import os
import time
import logging
import ast

import numpy as np
import h5py

import utils
from feature import Feature
from imageLocationUtility import ImageLocationUtility

class FeatureArray(np.ndarray):
    """Array with metadata."""

    def __new__(cls, filename):
        """Reads metadata and features from a HDF5 file.
        
        Args:
            filename (str): filename to read

        Returns:
            A new FeatureArray
        """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type

        logging.info("Reading metadata and features from: %s" % filename)

        # Check file extension
        fn, file_extension = os.path.splitext(filename)
        if file_extension != ".h5":
            raise ValueError("Filename has to be *.h5")

        with h5py.File(filename, "r") as hf:
            attrs = dict(hf.attrs)

            # Parse metadata object
            metadata = np.array([ast.literal_eval(m) for m in hf["metadata"]])
            features_raw = np.array(hf["features"])
            locations = np.array(hf["locations"]) if "locations" in hf.keys() else None
            
            total, h, w, depth = features_raw.shape

            features = np.empty(shape=(total, h, w), dtype=Feature)
            
            for i in range(total):
                meta = metadata[i]
                for y in range(h):
                    for x in range(w):
                        features[i, y, x] = Feature(features_raw[i, y, x], meta, x, y, w, h)
                        if locations is not None:
                            features[i, y, x].location = locations[i, y, x]
        
        obj = np.asarray(features).view(cls)
        obj.attrs = attrs
        obj.filename = filename
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.attrs = getattr(obj, "attrs", None)
        self.filename = getattr(obj, "filename", None)

    def __get_property__(key):
        return lambda self: None if self.attrs is None or key in self.attrs.keys() else self.attrs[key]

    extractor  = property(__get_property__("Extractor"))
    files      = property(__get_property__("Files"))
    batch_size = property(__get_property__("Batch size"))
    
    # As of right now one image has one label, so we just look at the top left patch
    no_anomaly = property(lambda self: self[[f[0,0].label == 1 for f in self]])
    anomaly    = property(lambda self: self[[f[0,0].label == 2 for f in self]])
    
    def save_locations_to_file(self, filename=None):
        """ Save the locations of every patch to filename """

        if filename is None:
            filename = self.filename

        with h5py.File(filename, "r+") as hf:
            # Remove the old locations dataset
            if "locations" in hf.keys():
                del hf["locations"]
            self.calculate_locations()
            locations = np.reshape([f.location for f in self.flatten()], self.shape + (2,))
            hf.create_dataset("locations", data=locations, dtype=np.float64)
        
    def calculate_locations(self):
        """Calculate the real world coordinates of every feature"""

        # If this FeatureArray is just a view, calculate the locations for the real FeatureArray
        if self.base is not None and isinstance(self.base, FeatureArray):
            print "Going down!"
            self.base.calculate_locations()
            return

        # Cheap check TODO: Improve on this and only calculate the locations not done yet
        if self.take(1).location is not None:
            logging.info("Already calculated locations")
            return

        ilu = ImageLocationUtility()

        logging.info("Calculating locations of every patch")
        total, h, w = self.shape
        
        image_locations = ilu.span_grid(w, h, offset_x=0.5, offset_y=0.5)
        relative_locations = ilu.image_to_relative(image_locations)

        with utils.GracefulInterruptHandler() as g:
            start = time.time()
            utils.print_progress(0, total * h * w, prefix="Calculating locations:",
                                           suffix = "%i / %i" % (0, total * h * w))
            
            c = 0
            for i in range(total):
                for y in range(h):
                    for x in range(w):
                        c += 1
                        
                        if g.interrupted:
                            logging.warning("Interrupted!")
                            raise KeyboardInterrupt()
                        
                        feature = self[i, y, x]
                        feature.location = ilu.relative_to_absolute(relative_locations[x,y], feature.camera_position, feature.camera_rotation)

                # Print progress
                utils.print_progress(c,
                                     total * h * w,
                                     prefix = "Calculating locations:",
                                     suffix = "%i / %i" % (c, total * h * w),
                                     time_start = start)

    def add_location_as_feature_dimension(self):
        """Act as if the location of a feature would be two additional feature dimensions"""
        # Cheap check
        if self.take(1).location is None:
            self.calculate_locations()

        logging.info("Adding locations as feature dimensions")

        features_flat = self.flatten()
        for i in range(len(features_flat)):
            features_flat[i] = np.append(features_flat[i], features_flat[i].location)