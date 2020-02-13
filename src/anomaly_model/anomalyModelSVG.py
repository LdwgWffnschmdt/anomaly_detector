# -*- coding: utf-8 -*-

import os

import logging
import h5py
import numpy as np
import tensorflow_probability as tfp
from scipy.spatial import distance

from anomalyModelBase import AnomalyModelBase
import feature_extractor.utils as utils

import matplotlib.pyplot as plt

class AnomalyModelSVG(AnomalyModelBase):
    """Anomaly model formed by a Single Variate Gaussian (SVG) with model parameters Θ_SVG = (μ,σ²)
    Reference: https://www.mdpi.com/1424-8220/16/11/1904/htm
    """
    def __init__(self):
        AnomalyModelBase.__init__(self)
        self.NAME       = "SVG"
        self._var        = None # Variance σ²
        self._mean       = None # Mean μ
        self.threshold  = None # Threshold for classification
    
    def classify(self, feature_vector, threshold=None):
        """The anomaly measure is defined as the Mahalanobis distance between a feature sample
        and the single variate Gaussian distribution along each dimension.
        """
        if threshold is None:
            threshold = self.threshold
        return self._mahalanobis_distance(feature_vector) > threshold
    
    def _mahalanobis_distance(self, feature_vector):
        """Calculate the Mahalanobis distance between the input and the model"""
        assert not self._var is None and not self._mean is None, \
            "You need to load a model before computing a Mahalanobis distance"
        assert feature_vector.shape == self._var.shape == self._mean.shape, \
            "Shapes don't match (x: %s, μ: %s, σ²: %s)" % (feature_vector.shape, self._mean.shape, self._var.shape)
        
        return np.sqrt(np.sum((feature_vector - self._mean) **2 / self._var))
        
        ### scipy implementation is way slower
        # if self._varI is None:
        #     self._varI = np.linalg.inv(np.diag(self._var))
        # return distance.mahalanobis(feature_vector, self._mean, self._varI)

    def generate_model(self, metadata, features):
        # Reduce features to simple list
        features_flat = self.reduce_feature_array(features)

        logging.info("Generating a Single Variate Gaussian (SVG) from %i feature vectors of length %i" % (features_flat.shape[0], features_flat.shape[1]))

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
    model = AnomalyModelSVG()

    features_file = "/home/ludwig/ros/src/ROS-kate_bag/bags/real/TFRecord/Features/MobileNetV2_Block6.h5"
    # features_file = "/home/ludwig/ros/src/ROS-kate_bag/bags/real/TFRecord/Features/hard_2020-02-06-17-20-22.MobileNetV2_Block6.h5"

    # Read file
    metadata, features = utils.read_features_file(features_file)

    metadata_anomaly = metadata[[m["label"] == 2 for m in metadata]]
    features_anomaly = features[[m["label"] == 2 for m in metadata]]

    # Only take feature vectors of images labeled as anomaly free (label == 1)
    metadata_no_anomaly = metadata[[m["label"] == 1 for m in metadata]]
    features_no_anomaly = features[[m["label"] == 1 for m in metadata]]
    
    # Generate model
    if model.generate_model(metadata_no_anomaly, features_no_anomaly) == False:
        logging.info("Could not generate model.")

    features_flat = model.reduce_feature_array(features)
    dists = np.array(list(map(model._mahalanobis_distance, features_flat)))
    thresh = np.amax(dists)
    print thresh

    def _feature_to_color(feature):
        b = 0
        g = 0
        r = model._mahalanobis_distance(feature) * (255 / thresh)
        return (b, g, r)

    model.visualize(metadata, features, _feature_to_color)

    # Save model
    # model.save_model_to_file(os.path.abspath(features_file.replace(".h5", "")) + "." + model.NAME + ".h5")
    
    
    # features_flat = model.reduce_feature_array(features)
    # features_anomaly_flat = model.reduce_feature_array(features_anomaly)

    # dists = np.array(list(map(model._mahalanobis_distance, features_flat)))
    # dists_anomaly = np.array(list(map(model._mahalanobis_distance, features_anomaly_flat)))

    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # ax1.set_title("No anomaly (%i)" % len(features_flat))
    # ax1.hist(dists, bins=50)

    # ax2.set_title("Anomaly (%i)" % len(features_anomaly_flat))
    # ax2.hist(dists_anomaly, bins=50)

    # fig.suptitle("Mahalanobis distances")

    # plt.show()