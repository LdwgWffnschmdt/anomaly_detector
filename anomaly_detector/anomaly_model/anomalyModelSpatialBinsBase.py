# -*- coding: utf-8 -*-

import os
import logging

import h5py
import numpy as np

from anomalyModelBase import AnomalyModelBase
import common.utils as utils

class AnomalyModelSpatialBinsBase(AnomalyModelBase):
    """ Base for anomaly models that create one model per spatial bin (grid cell) """
    def __init__(self, create_anomaly_model_func):
        AnomalyModelBase.__init__(self)
        self.CELL_SIZE = 0.2 # Width and height of spatial bin in meter
        self.CREATE_ANOMALY_MODEL_FUNC = create_anomaly_model_func

        self.ilu = utils.ImageLocationUtility()
    
    def classify(self, metadata, feature_vector, threshold=None):
        """The anomaly measure is defined as the Mahalanobis distance between a feature sample
        and the single variate Gaussian distribution along each dimension.
        """
        if threshold is None:
            threshold = self.threshold
        return self._mahalanobis_distance(feature) > threshold
    
    def _mahalanobis_distance(self, feature):
        """Calculate the Mahalanobis distance between the input and the model"""
        # Calculate position
        self.ilu.image_to_relative()

        # Find the correct model
        indices = np.stack([np.digitize(locations[:,:,:,0], bins_x),
                            np.digitize(locations[:,:,:,1], bins_y)], axis=3)

    def generate_model(self, features):
        # Get location for features
        locations = self.ilu.get_locations_for_features(metadata, features)

        # Get extent
        x_min = np.amin(locations[:,:,:,0], axis=(0,1,2))
        y_min = np.amin(locations[:,:,:,1], axis=(0,1,2))
        x_max = np.amax(locations[:,:,:,0], axis=(0,1,2))
        y_max = np.amax(locations[:,:,:,1], axis=(0,1,2))

        # Round to cell size (increases bounds to fit next cell size)
        x_min -= x_min % self.CELL_SIZE
        y_min -= y_min % self.CELL_SIZE
        x_max += self.CELL_SIZE - (x_max % self.CELL_SIZE)
        y_max += self.CELL_SIZE - (y_max % self.CELL_SIZE)

        # Create the bins
        bins_x = np.arange(x_min, x_max, self.CELL_SIZE)
        bins_y = np.arange(y_min, y_max, self.CELL_SIZE)

        # Use digitize to sort the locations in the bins
        # --> (#frames, h, w, 2) where the entries in the last
        #     dimension are the respective bin indices (u, v)
        indices = np.stack([np.digitize(locations[:,:,:,0], bins_x),
                            np.digitize(locations[:,:,:,1], bins_y)], axis=3)

        self.models = np.empty(shape=(len(bins_x), len(bins_y)), dtype=object)

        for u, x in enumerate(bins_x):
            for v, y in enumerate(bins_y):
                # Array containing booleans, whether the feature/metadata is in the current bin
                index_filter = [index[0] == u and index[1] == v for index in indices]

                # Create a new model
                model = self.CREATE_ANOMALY_MODEL_FUNC()
                model.generate_model(metadata[index_filter], features[index_filter])
                self.models[u, v] = model
                
        return True

# Only for tests
if __name__ == "__main__":
    model = AnomalyModelSpatialBinsBase()
    file_content = utils.read_features_file("/home/ludwig/ros/src/ROS-kate_bag/bags/real/TFRecord/Features/MobileNetV2_Block6.h5")
    model.generate_model(file_content.all.metadata, file_content.all.features)