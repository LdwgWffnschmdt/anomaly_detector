# -*- coding: utf-8 -*-

import os
import sys
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from tqdm import tqdm

from anomalyModelBalancedDistribution import AnomalyModelBalancedDistribution
from common import utils, logger

class AnomalyModelBalancedDistributionSVG(AnomalyModelBalancedDistribution):
    """Anomaly model formed by a Balanced Distribution of feature vectors
    Reference: https://www.mdpi.com/2076-3417/9/4/757
    """
    def __init__(self, initial_normal_features=1000, threshold_learning=50, threshold_classification=5, pruning_parameter=0.5):
        AnomalyModelBalancedDistribution.__init__(self, initial_normal_features, threshold_learning, threshold_classification, pruning_parameter)
        self._var       = None # Variance σ²

    def _calculate_mean_and_covariance(self):
        """Calculate mean and inverse of covariance of the "normal" distribution"""
        assert not self.balanced_distribution is None and len(self.balanced_distribution) > 0, \
            "Can't calculate mean or covariance of nothing!"
        
        self._mean = np.mean(self.balanced_distribution["features"], axis=0, dtype=np.float64)  # Mean
        self._var = np.var(self.balanced_distribution["features"], axis=0, dtype=np.float64)    # Variance
    
    def __mahalanobis_distance__(self, patch):
        """Calculate the Mahalanobis distance between the input and the model"""
        assert not self._var is None and not self._mean is None, \
            "You need to load a model before computing a Mahalanobis distance"

        feature = patch["features"]
        assert feature.shape == self._var.shape == self._mean.shape, \
            "Shapes don't match (x: %s, μ: %s, σ²: %s)" % (feature.shape, self._mean.shape, self._var.shape)
        
        # TODO: This is a hack for collapsed SVGs. Should normally not happen
        if not self._var.any(): # var contains only zeros
            if (feature == self._mean).all():
                return 0.0
            else:
                return np.nan

        return np.sqrt(np.sum(np.divide((feature - self._mean) **2, self._var, out=np.zeros_like(self._var), where=self._var!=0)))

# Only for tests
if __name__ == "__main__":
    from common import PatchArray
    import consts
    patches = PatchArray(consts.FEATURES_FILE)

    threshold_learning = int(np.nanmax(patches.mahalanobis_distances["SVG"]) * 0.7)

    model = AnomalyModelBalancedDistributionSVG(threshold_learning=threshold_learning)
    if model.load_or_generate(patches):
        model.visualize(threshold=200)