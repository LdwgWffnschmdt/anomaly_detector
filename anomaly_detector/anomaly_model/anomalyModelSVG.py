# -*- coding: utf-8 -*-

import os
import logging

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
        
        # TODO: This is a hack for collapsed SVGs. Should normally not happen
        if not self._var.any(): # var contains only zeros
            return (feature == self._mean).all()

        return np.sqrt(np.sum((feature - self._mean) **2 / self._var))
        
        ### scipy implementation is way slower
        # if self._varI is None:
        #     self._varI = np.linalg.inv(np.diag(self._var))
        # return distance.mahalanobis(feature, self._mean, self._varI)

    def generate_model(self, features):
        AnomalyModelBase.generate_model(self, features) # Call base
        
        # Reduce features to simple list
        features_flat = features.flatten()

        logging.info("Generating SVG from %i feature vectors of length %i" % (features_flat.shape[0], len(features_flat[0])))

        if features_flat.shape[0] == 1:
            logging.warn("Trying to generate SVG from a single value.")

        # Get the variance
        logging.info("Calculating the variance")
        # np.var(FeatureArray) results in a FeatureArray with only one Feature
        # whose values represent the variance values. So we need to take that
        # one Feature using .item() to be able to use and store it.
        self._var = np.var(features_flat).item()
        # --> one variance per feature dimension

        # Get the mean
        logging.info("Calculating the mean")
        self._mean = np.mean(features_flat, axis=0).item() # See variance
        # --> one mean per feature dimension

        # Get maximum mahalanobis distance as threshold
        logging.info("Calculating the threshold")
        dists = np.array(list(map(self._mahalanobis_distance, features_flat)))
        self.threshold = np.amax(dists)

        return True

    def __load_model_from_file__(self, h5file):
        """Load a SVG model from file"""
        self._var       = np.array(h5file["var"])
        self._mean      = np.array(h5file["mean"])
        self.threshold  = h5file.attrs["threshold"]
        assert len(self._var) == len(self._mean), "Dimensions of variance and mean do not match!"
        logging.info("Successfully loaded SVG parameters of dimension %i" % len(self._var))
    
    def __save_model_to_file__(self, h5file):
        """Save the model to disk"""
        h5file.create_dataset("var",        data=self._var)
        h5file.create_dataset("mean",       data=self._mean)
        h5file.attrs["threshold"] = self.threshold

# Only for tests
if __name__ == "__main__":
    from anomalyModelTest import AnomalyModelTest
    test = AnomalyModelTest(AnomalyModelSVG())

    # test.calculateMahalobisDistances()
    # test.showMahalanobisDistribution()
    test.visualize(10)#26.525405)