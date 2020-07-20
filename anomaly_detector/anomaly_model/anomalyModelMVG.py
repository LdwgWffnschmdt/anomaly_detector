# -*- coding: utf-8 -*-

import os

import numpy as np
# import tensorflow_probability as tfp
from scipy.spatial import distance

from anomalyModelBase import AnomalyModelBase
from common import utils, logger

class AnomalyModelMVG(AnomalyModelBase):
    """Anomaly model formed by a multivariate Gaussian (MVG) with model parameters Θ_MVG = (μ,Ʃ)
    """
    def __init__(self):
        AnomalyModelBase.__init__(self)
        self._var       = None # Covariance matrix Ʃ
        self._varI      = None # Inverse of covariance matrix Ʃ⁻¹
        self._mean      = None # Mean μ
    
    def classify(self, patch, threshold=None):
        """The anomaly measure is defined as the Mahalanobis distance between a feature sample
        and the multivariate Gaussian distribution along each dimension.
        """
        return self.__mahalanobis_distance__(patch) > threshold
    
    def __mahalanobis_distance__(self, patch):
        """Calculate the Mahalanobis distance between the input and the model"""
        assert not self._var is None and not self._mean is None, \
            "You need to load a model before computing a Mahalanobis distance"
            
        feature = patch.features
        assert feature.shape[0] == self._var.shape[0] == self._var.shape[1] == self._mean.shape[0], \
            "Shapes don't match (x: %s, μ: %s, Ʃ: %s)" % (feature.shape, self._mean.shape, self._var.shape)
        
        # TODO: This is a hack for collapsed MVGs. Should normally not happen
        if not self._var.any(): # var contains only zeros
            if (feature == self._mean).all():
                return 0.0
            else:
                return np.nan

        if self._varI is None:
            self._varI = np.linalg.inv(self._var)
        return distance.mahalanobis(feature, self._mean, self._varI)

    def __generate_model__(self, patches, silent=False):
        if not silent: logger.info("Generating MVG from %i feature vectors of length %i" % (len(patches.ravel()), patches.features.shape[-1]))

        if not silent and patches.size == 1:
            logger.warning("Trying to generate MVG from a single value.")

        # Get the variance
        if not silent: logger.info("Calculating the covariance")
        self._var = patches.cov()
        self._varI = np.linalg.pinv(self._var)
        # --> one variance per feature dimension

        # Get the mean
        if not silent: logger.info("Calculating the mean")
        self._mean = patches.mean()
        # --> one mean per feature dimension

        return True

    def __load_model_from_file__(self, h5file):
        """Load a MVG model from file"""
        if not "var" in h5file.keys() or not "varI" in h5file.keys() or not "mean" in h5file.keys():
            return False
        self._var  = np.array(h5file["var"])
        self._varI = np.array(h5file["varI"])
        self._mean = np.array(h5file["mean"])
        return True
    
    def __save_model_to_file__(self, h5file):
        """Save the model to disk"""
        h5file.create_dataset("var",  data=self._var)
        h5file.create_dataset("varI", data=self._varI)
        h5file.create_dataset("mean", data=self._mean)
        return True

# Only for tests
if __name__ == "__main__":
    model = AnomalyModelMVG()
    if model.load_or_generate(load_patches=True):
        model.visualize(threshold=200)