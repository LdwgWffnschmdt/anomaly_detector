# -*- coding: utf-8 -*-

import os
import sys
import common.logger as logger
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from tqdm import tqdm

from anomalyModelBase import AnomalyModelBase
import common.utils as utils

class AnomalyModelBalancedDistribution(AnomalyModelBase):
    """Anomaly model formed by a Balanced Distribution of feature vectors
    Reference: https://www.mdpi.com/2076-3417/9/4/757
    """
    def __init__(self, initial_normal_features=1000, threshold_learning=20, threshold_classification=5, pruning_parameter=0.5):
        AnomalyModelBase.__init__(self)
        self.NAME += "/%i/%i/%i" % (initial_normal_features, threshold_learning, pruning_parameter)
        
        assert 0 < pruning_parameter < 1, "Pruning parameter out of range (0 < η < 1)"

        self.initial_normal_features    = initial_normal_features   # See reference algorithm variable N
        self.threshold_learning         = threshold_learning        # See reference algorithm variable α
        self.threshold_classification   = threshold_classification  # See reference algorithm variable β
        self.pruning_parameter          = pruning_parameter         # See reference algorithm variable η

        self.balanced_distribution        = None          # Array containing all the "normal" samples

        self._mean = None   # Mean
        self._covI = None   # Inverse of covariance matrix

    
    def classify(self, feature, threshold_classification=None):
        """The anomaly measure is defined as the Mahalanobis distance between a feature sample
        and the Balanced Distribution.
        """
        if threshold_classification is None:
            threshold_classification = self.threshold_classification
        
        if self._mean is None or self._covI is None:
            self._calculate_mean_and_covariance()

        return self._mahalanobis_distance(feature) > threshold_classification
    
    def _calculate_mean_and_covariance(self):
        """Calculate mean and inverse of covariance of the "normal" distribution"""
        assert not self.balanced_distribution is None and len(self.balanced_distribution) > 0, \
            "Can't calculate mean or covariance of nothing!"
        
        self._mean = self.balanced_distribution.mean()  # Mean
        cov = self.balanced_distribution.cov()          # Covariance matrix
        self._covI = np.linalg.pinv(cov)                # Inverse of covariance matrix
    
    def _mahalanobis_distance(self, feature):
        """Calculate the Mahalanobis distance between the input and the model"""
        assert not self._covI is None and not self._mean is None, \
            "You need to load a model before computing a Mahalanobis distance"

        assert feature.shape[0] == self._mean.shape[0] == self._covI.shape[0] == self._covI.shape[1], \
            "Shapes don't match (x: %s, μ: %s, Σ¯¹: %s)" % (feature.shape, self._mean.shape, self._covI.shape)
        
        return distance.mahalanobis(feature, self._mean, self._covI)

    def __generate_model__(self, features):
        # Reduce features to simple list
        features_flat = features.flatten()

        logger.info("Generating a Balanced Distribution from %i feature vectors of length %i" % (features_flat.shape[0], len(features_flat[0])))

        assert features_flat.shape[0] > self.initial_normal_features, \
            "Not enough initial features provided. Please decrease initial_normal_features (%i)" % self.initial_normal_features

        # Create initial set of "normal" vectors
        self.balanced_distribution = features_flat[:self.initial_normal_features]

        self._calculate_mean_and_covariance()

        # Loop over the remaining feature vectors
        with tqdm(desc="Creating balanced distribution",
                    initial=self.initial_normal_features,
                    total=features_flat.shape[0],
                    file=sys.stderr) as pbar:
            for feature in features_flat[self.initial_normal_features:]:
                # Calculate the Mahalanobis distance to the "normal" distribution
                dist = self._mahalanobis_distance(feature)
                if dist > self.threshold_learning:
                    # Add the vector to the "normal" distribution
                    self.balanced_distribution = np.append(self.balanced_distribution, [feature], axis=0)

                    # Recalculate mean and covariance
                    self._calculate_mean_and_covariance()
                
                # Print progress
                pbar.set_postfix("%i vectors in Balanced Distribution" % len(self.balanced_distribution))
                pbar.update()

        # Prune the distribution
        
        logger.info(np.mean(np.array([self._mahalanobis_distance(f) for f in self.balanced_distribution])))

        prune_filter = []
        pruned = 0

        with tqdm(desc="Pruning balanced distribution",
                    total=len(self.balanced_distribution),
                    file=sys.stderr) as pbar:
            for feature in self.balanced_distribution:
                prune = self._mahalanobis_distance(feature) < self.threshold_learning * self.pruning_parameter
                prune_filter.append(prune)

                if prune:
                    pruned += 1

                # Print progress
                pbar.set_postfix("%i vectors pruned" % pruned)
                pbar.update()

        self.balanced_distribution = self.balanced_distribution[prune_filter]

        logger.info("Generated Balanced Distribution with %i entries" % len(self.balanced_distribution))
    
        self._calculate_mean_and_covariance()
        return True

        
    def __load_model_from_file__(self, h5file):
        """Load a Balanced Distribution from file"""
        if not "balanced_distribution" in h5file.keys() or \
           not "initial_normal_features" in h5file.attrs.keys() or \
           not "threshold_learning" in h5file.attrs.keys() or \
           not "threshold_classification" in h5file.attrs.keys() or \
           not "pruning_parameter" in h5file.attrs.keys():
            return False
        
        self.balanced_distribution      = np.array(h5file["balanced_distribution"])
        self.initial_normal_features    = h5file.attrs["initial_normal_features"]
        self.threshold_learning         = h5file.attrs["threshold_learning"]
        self.threshold_classification   = h5file.attrs["threshold_classification"]
        self.pruning_parameter          = h5file.attrs["pruning_parameter"]
        assert 0 < self.pruning_parameter < 1, "Pruning parameter out of range (0 < η < 1)"
        self._calculate_mean_and_covariance()
        return True
    
    def save_model_to_file(self, h5file):
        """Save the model to disk"""
        h5ffile.create_dataset("balanced_distribution",        data=self.balanced_distribution, dtype=np.float64)
        h5ffile.attrs["initial_normal_features"]    = self.initial_normal_features
        h5ffile.attrs["threshold_learning"]         = self.threshold_learning
        h5ffile.attrs["threshold_classification"]   = self.threshold_classification
        h5ffile.attrs["pruning_parameter"]          = self.pruning_parameter
        return True

# Only for tests
if __name__ == "__main__":
    model = AnomalyModelBalancedDistribution()
    if model.load_or_generate(load_features=True):
        
        def _feature_to_color(feature):
            b = 100 if feature in model.balanced_distribution else 0
            g = 0
            r = model._mahalanobis_distance(feature) * (255 / 60)
            #r = 100 if self.model._mahalanobis_distance(feature) > threshold else 0
            return (b, g, r)

        def _pause(feature):
            return feature in test.model.balanced_distribution

        model.visualize(threshold=60, feature_to_color_func=_feature_to_color)