import os
import logging
import ast

import numpy as np
import h5py

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
            # Parse metadata object
            metadata = np.array([ast.literal_eval(m) for m in hf["metadata"]])
            features_raw = np.array(hf["features"])
            
            total, h, w, depth = features_raw.shape

            features = np.empty(shape=(total, h, w), dtype=object)
            
            for i in range(total):
                meta = metadata[i]
                for y in range(h):
                    for x in range(w):
                        features[i, y, x] = Feature(features_raw[i, y, x], meta, x, y, w, h)
        
        return np.asarray(features).view(cls)
    
    # As of right now one image has one label, so we just look at the top left patch
    no_anomaly = property(lambda self: self[[f[0,0].label == 1 for f in self]])
    anomaly    = property(lambda self: self[[f[0,0].label == 2 for f in self]])

    def calculate_locations(self):
        """Calculate the real world coordinates of every feature"""
        # Cheap check
        if self.take(1).location is not None:
            logging.info("Already calculated locations")
            return

        ilu = ImageLocationUtility()

        logging.info("Calculating locations of every patch")
        total, h, w = self.shape
        
        image_locations = ilu.span_grid(w, h, offset_x=0.5, offset_y=0.5)
        relative_locations = ilu.image_to_relative(image_locations)

        for i in range(total):
            for y in range(h):
                for x in range(w):
                    feature = self[i, y, x]
                    feature.location = ilu.relative_to_absolute(relative_locations[x,y], feature.camera_position, feature.camera_rotation)

    def add_location_as_feature_dimension(self):
        """Act as if the location of a feature would be two additional feature dimensions"""
        # Cheap check
        if self.take(1).location is None:
            self.calculate_locations()

        logging.info("Adding locations as feature dimensions")

        features_flat = self.flatten()
        for i in range(len(features_flat)):
            features_flat[i] = np.append(features_flat[i], features_flat[i].location)