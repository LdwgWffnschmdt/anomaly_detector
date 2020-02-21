# -*- coding: utf-8 -*-

import os
import logging

import h5py
import numpy as np
# import tensorflow_probability as tfp
# from scipy.spatial import distance

from anomalyModelBase import AnomalyModelBase
import common.utils as utils

class AnomalyModelSVG(AnomalyModelBase):
    """Anomaly model formed by a Single Variate Gaussian (SVG) with model parameters Θ_SVG = (μ,σ²)
    Reference: https://www.mdpi.com/1424-8220/16/11/1904/htm
    """
    def __init__(self):
        AnomalyModelBase.__init__(self)
        self._var       = None # Variance σ²
        self._mean      = None # Mean μ
        self.threshold  = None # Threshold for classification
    
    def classify(self, feature, threshold=None):
        """The anomaly measure is defined as the Mahalanobis distance between a feature sample
        and the single variate Gaussian distribution along each dimension.
        """
        if threshold is None:
            threshold = self.threshold
        return self._mahalanobis_distance(feature) > threshold
    
    def _mahalanobis_distance(self, feature):
        """Calculate the Mahalanobis distance between the input and the model"""
        assert not self._var is None and not self._mean is None, \
            "You need to load a model before computing a Mahalanobis distance"
        assert feature.shape == self._var.shape == self._mean.shape, \
            "Shapes don't match (x: %s, μ: %s, σ²: %s)" % (feature.shape, self._mean.shape, self._var.shape)
        
        return np.sqrt(np.sum((feature - self._mean) **2 / self._var))
        
        ### scipy implementation is way slower
        # if self._varI is None:
        #     self._varI = np.linalg.inv(np.diag(self._var))
        # return distance.mahalanobis(feature, self._mean, self._varI)

    def generate_model(self, features):
        # Reduce features to simple list
        features_flat = features.flatten()

        logging.info("Generating a Single Variate Gaussian (SVG) from %i feature vectors of length %i" % (features_flat.shape[0], len(features_flat[0])))

        # Get the variance
        logging.info("Calculating the variance")
        # self._var = tfp.stats.variance(features_flat)
        self._var = np.var(features_flat, axis=0, dtype=np.float64)
        # --> one variance per feature vector entry

        # Get the mean
        logging.info("Calculating the mean")
        self._mean = np.mean(features_flat, axis=0, dtype=np.float64)
        # --> one mean per feature vector entry

        # Get maximum mahalanobis distance as threshold
        # logging.info("Calculating the threshold")
        # dists = np.array(list(map(self._mahalanobis_distance, features_flat)))
        # self.threshold = np.amax(dists)

        return True

    def load_model_from_file(self, model_file):
        """Load a SVG model from file"""
        logging.info("Reading model parameters from: %s" % model_file)
        with h5py.File(model_file, "r") as hf:
            self._var       = np.array(hf["var"])
            self._mean      = np.array(hf["mean"])
            self.threshold  = np.array(hf["threshold"])
        assert len(self._var) == len(self._mean), "Dimensions of variance and mean do not match!"
        logging.info("Successfully loaded model parameters of dimension %i" % len(self._var))
    
    def save_model_to_file(self, output_file = ""):
        """Save the model to disk"""
        logging.info("Writing model parameters to: %s" % output_file)
        with h5py.File(output_file, "w") as hf:
            hf.create_dataset("var",        data=self._var, dtype=np.float64)
            hf.create_dataset("mean",       data=self._mean, dtype=np.float64)
            hf.create_dataset("threshold",  data=self.threshold, dtype=np.float64)
        logging.info("Successfully written model parameters to: %s" % output_file)

# Only for tests
if __name__ == "__main__":
    from anomalyModelTest import AnomalyModelTest
    test = AnomalyModelTest(AnomalyModelSVG())

    # test.calculateMahalobisDistances()
    # test.showMahalanobisDistribution()
    test.visualize(10)#26.525405)