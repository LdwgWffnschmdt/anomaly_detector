# -*- coding: utf-8 -*-

import os
import sys
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from tqdm import tqdm

from anomalyModelBase import AnomalyModelBase
from common import utils, logger

class AnomalyModelBalancedDistribution(AnomalyModelBase):
    """Anomaly model formed by a Balanced Distribution of feature vectors
    Reference: https://www.mdpi.com/2076-3417/9/4/757
    """
    def __init__(self, initial_normal_features=1000, threshold_learning=300, threshold_classification=5, pruning_parameter=0.3):
        AnomalyModelBase.__init__(self)
        self.NAME += "/%i/%i/%.2f" % (initial_normal_features, threshold_learning, pruning_parameter)
        
        assert 0 < pruning_parameter < 1, "Pruning parameter out of range (0 < η < 1)"

        self.initial_normal_features    = initial_normal_features   # See reference algorithm variable N
        self.threshold_learning         = threshold_learning        # See reference algorithm variable α
        self.threshold_classification   = threshold_classification  # See reference algorithm variable β
        self.pruning_parameter          = pruning_parameter         # See reference algorithm variable η

        self.balanced_distribution        = None          # Array containing all the "normal" samples

        self._mean = None   # Mean
        self._covI = None   # Inverse of covariance matrix

    
    def classify(self, patch, threshold_classification=None):
        """The anomaly measure is defined as the Mahalanobis distance between a feature sample
        and the Balanced Distribution.
        """
        if threshold_classification is None:
            threshold_classification = self.threshold_classification
        
        if self._mean is None or self._covI is None:
            self._calculate_mean_and_covariance()

        return self.__mahalanobis_distance__(patch) > threshold_classification
    
    def _calculate_mean_and_covariance(self):
        """Calculate mean and inverse of covariance of the "normal" distribution"""
        assert not self.balanced_distribution is None and len(self.balanced_distribution) > 0, \
            "Can't calculate mean or covariance of nothing!"
        
        self._mean = np.mean(self.balanced_distribution["features"], axis=0, dtype=np.float64)  # Mean
        cov = np.cov(self.balanced_distribution["features"], rowvar=False)                      # Covariance matrix
        try:
            self._covI = np.linalg.pinv(cov)                                            # Pseudo Inverse of covariance matrix
        except np.linalg.LinAlgError:
            self._covI = np.linalg.inv(cov)                                            # Inverse of covariance matrix
    
    def __mahalanobis_distance__(self, patch):
        """Calculate the Mahalanobis distance between the input and the model"""
        assert not self._covI is None and not self._mean is None, \
            "You need to load a model before computing a Mahalanobis distance"

        feature = patch["features"]
        assert feature.shape[0] == self._mean.shape[0] == self._covI.shape[0] == self._covI.shape[1], \
            "Shapes don't match (x: %s, μ: %s, Σ⁻¹: %s)" % (feature.shape, self._mean.shape, self._covI.shape)
        
        return distance.mahalanobis(feature, self._mean, self._covI)

    def __generate_model__(self, patches, silent=False):
        if not silent: logger.info("Generating a Balanced Distribution from %i feature vectors of length %i" % (len(patches.ravel()), patches.features.shape[-1]))

        patches_flat = patches.ravel()

        if patches_flat.shape[0] < self.initial_normal_features:
            self.initial_normal_features = patches_flat.shape[0]
        # assert patches_flat.shape[0] < self.initial_normal_features, \
        #     "Not enough initial features provided. Please decrease initial_normal_features (%i)" % self.initial_normal_features

        # Create initial set of "normal" vectors
        self.balanced_distribution = patches_flat[:self.initial_normal_features]

        self._calculate_mean_and_covariance()

        if patches.size <= self.initial_normal_features:
            return True

        # Loop over the remaining feature vectors
        if not silent:
            pbar = tqdm(desc="Creating balanced distribution",
                    initial=self.initial_normal_features,
                    total=patches_flat.shape[0],
                    file=sys.stderr)

        for patch in patches_flat[self.initial_normal_features:]:
            # Calculate the Mahalanobis distance to the "normal" distribution
            dist = self.__mahalanobis_distance__(patch)
            if dist > self.threshold_learning:
                # Add the vector to the "normal" distribution
                self.balanced_distribution = np.append(self.balanced_distribution, [patch], axis=0)

                # Recalculate mean and covariance
                self._calculate_mean_and_covariance()
            
            if not silent:
                # Print progress
                pbar.set_postfix({"Balanced Distribution": len(self.balanced_distribution)})
                pbar.update()
        
        if not silent:
            pbar.close()

        # Abort early if there are very few entries in the balanced distribution
        if self.balanced_distribution.size <= self.initial_normal_features:
            return True

        # Prune the distribution
        
        if not silent:
            logger.info(np.mean(np.array([self.__mahalanobis_distance__(f) for f in self.balanced_distribution])))
            pbar = tqdm(desc="Pruning balanced distribution",
                    total=len(self.balanced_distribution),
                    file=sys.stderr)

        prune_filter = np.empty(self.balanced_distribution.shape, dtype=np.bool)
        pruned = 0
    
        for i, patch in enumerate(self.balanced_distribution):
            not_prune = self.__mahalanobis_distance__(patch) > self.threshold_learning * self.pruning_parameter
            prune_filter[i] = not_prune

            if not not_prune:
                pruned += 1

            if not silent:
                # Print progress
                pbar.set_postfix({"Pruned": pruned})
                pbar.update()

        # Only apply pruning if it wouldn't result in an empty distribution
        if self.balanced_distribution[prune_filter].size > 0:
            self.balanced_distribution = self.balanced_distribution[prune_filter]

        if not silent:
            pbar.close()
            logger.info("Generated Balanced Distribution with %i entries" % len(self.balanced_distribution))
    
        if len(self.balanced_distribution) <= 0:
            logger.error("No entry in balanced distribution")

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
        
        self.balanced_distribution      = np.array([h5file["balanced_distribution"]], dtype=[("features", np.float64)]).squeeze()
        self.initial_normal_features    = h5file.attrs["initial_normal_features"]
        self.threshold_learning         = h5file.attrs["threshold_learning"]
        self.threshold_classification   = h5file.attrs["threshold_classification"]
        self.pruning_parameter          = h5file.attrs["pruning_parameter"]
        assert 0 < self.pruning_parameter < 1, "Pruning parameter out of range (0 < η < 1)"
        self._calculate_mean_and_covariance()
        return True
    
    def __save_model_to_file__(self, h5file):
        """Save the model to disk"""
        h5file.create_dataset("balanced_distribution", data=self.balanced_distribution["features"], dtype=np.float64)
        h5file.create_dataset("balanced_distribution_times", data=self.balanced_distribution["times"], dtype=np.float64)
        h5file.attrs["initial_normal_features"]    = self.initial_normal_features
        h5file.attrs["threshold_learning"]         = self.threshold_learning
        h5file.attrs["threshold_classification"]   = self.threshold_classification
        h5file.attrs["pruning_parameter"]          = self.pruning_parameter
        return True

# Only for tests
if __name__ == "__main__":
    model = AnomalyModelBalancedDistribution()
    if model.load_or_generate(load_patches=True):
        
        def _patch_to_color(patch):
            b = 100 if patch in model.balanced_distribution else 0
            g = 0
            r = model.__mahalanobis_distance__(patch) * (255 / 60)
            #r = 100 if self.model.__mahalanobis_distance__(patch) > threshold else 0
            return (b, g, r)

        def _pause(patch):
            return patch in model.balanced_distribution

        model.visualize(patch_to_color_func=_patch_to_color)