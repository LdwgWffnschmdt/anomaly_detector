# -*- coding: utf-8 -*-

import os

import numpy as np
# import tensorflow_probability as tfp
from scipy.spatial import distance

from anomalyModelBase import AnomalyModelBase
from common import utils, logger

class AnomalyModelSVG(AnomalyModelBase):
    """Anomaly model formed by a Single Variate Gaussian (SVG) with model parameters Θ_SVG = (μ,σ²)
    Reference: https://www.mdpi.com/1424-8220/16/11/1904/htm
    """
    def __init__(self):
        AnomalyModelBase.__init__(self)
        self._var       = None # Variance σ²
        self._mean      = None # Mean μ
    
    def classify(self, patch, threshold=None):
        """The anomaly measure is defined as the Mahalanobis distance between a feature sample
        and the single variate Gaussian distribution along each dimension.
        """
        return self.__mahalanobis_distance__(patch) > threshold
    
    def __mahalanobis_distance__(self, patch):
        """Calculate the Mahalanobis distance between the input and the model"""
        assert not self._var is None and not self._mean is None, \
            "You need to load a model before computing a Mahalanobis distance"
            
        feature = patch.features
        assert feature.shape == self._var.shape == self._mean.shape, \
            "Shapes don't match (x: %s, μ: %s, σ²: %s)" % (feature.shape, self._mean.shape, self._var.shape)
        
        # TODO: This is a hack for collapsed SVGs. Should normally not happen
        if not self._var.any(): # var contains only zeros
            if (feature == self._mean).all():
                return 0.0
            else:
                return np.nan

        res = np.sqrt(np.sum(np.divide((feature - self._mean) **2, self._var, out=np.zeros_like(self._var), where=self._var!=0)))
        # res2 = distance.mahalanobis(feature, self._mean, np.diag(self._varI))
        return res
        
        ### scipy implementation is way slower
        # if self._varI is None:
        #     self._varI = np.linalg.inv(np.diag(self._var))
        # return distance.mahalanobis(feature, self._mean, self._varI)

    def __generate_model__(self, patches, silent=False):
        if not silent: logger.info("Generating SVG from %i feature vectors of length %i" % (len(patches.ravel()), patches.features.shape[-1]))

        if not silent and patches.size == 1:
            logger.warning("Trying to generate SVG from a single value.")

        # Get the variance
        if not silent: logger.info("Calculating the variance")
        self._var = patches.var()
        # d = np.diag(self._var)
        # self._varI = np.linalg.pinv(d)
        # self._varI = np.divide(np.ones_like(self._var), self._var, out=np.zeros_like(self._var), where=self._var!=0)
        # --> one variance per feature dimension

        # Get the mean
        if not silent: logger.info("Calculating the mean")
        self._mean = patches.mean()
        # --> one mean per feature dimension

        return True

    def __load_model_from_file__(self, h5file):
        """Load a SVG model from file"""
        if not "var" in h5file.keys() or not "mean" in h5file.keys():
            return False
        self._var  = np.array(h5file["var"])
        # self._varI = np.array(h5file["varI"])#np.linalg.pinv(np.diag(self._var))
        self._mean = np.array(h5file["mean"])
        assert len(self._var) == len(self._mean), "Dimensions of variance and mean do not match!"
        return True
    
    def __save_model_to_file__(self, h5file):
        """Save the model to disk"""
        h5file.create_dataset("var",  data=self._var)
        # h5file.create_dataset("varI", data=self._varI)
        h5file.create_dataset("mean", data=self._mean)
        return True

# Only for tests
if __name__ == "__main__":
    model = AnomalyModelSVG()
    if model.load_or_generate(load_patches=True):
        model.visualize(threshold=200)