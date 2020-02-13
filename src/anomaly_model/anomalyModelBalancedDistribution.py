# -*- coding: utf-8 -*-

import os
import logging
import time
import h5py
import numpy as np
import tensorflow_probability as tfp
from scipy.spatial import distance

from anomalyModelBase import AnomalyModelBase
import feature_extractor.utils as utils

class AnomalyModelBalancedDistribution(AnomalyModelBase):
    """Anomaly model formed by a Balanced Distribution of feature vectors
    Reference: https://www.mdpi.com/2076-3417/9/4/757
    """
    def __init__(self, initial_normal_features=1000, threshold_learning=20, threshold_classification=5, pruning_parameter=0.5):
        AnomalyModelBase.__init__(self)
        self.NAME                       = "BalancedDistribution"
        
        assert 0 < pruning_parameter < 1, "Pruning parameter out of range (0 < η < 1)"

        self.initial_normal_features    = initial_normal_features   # See reference algorithm variable N
        self.threshold_learning         = threshold_learning        # See reference algorithm variable α
        self.threshold_classification   = threshold_classification  # See reference algorithm variable β
        self.pruning_parameter          = pruning_parameter         # See reference algorithm variable η

        self.normal_distribution        = None          # Array containing all the "normal" samples

        self._mean = None   # Mean
        self._covI = None   # Inverse of covariance matrix

    
    def classify(self, feature_vector, threshold_classification=None):
        """The anomaly measure is defined as the Mahalanobis distance between a feature sample
        and the Balanced Distribution.
        """
        if threshold_classification is None:
            threshold_classification = self.threshold_classification
        
        if self._mean is None or self._covI is None:
            self._calculate_mean_and_covariance()

        return dist > threshold_classification
    
    def _calculate_mean_and_covariance(self):
        """Calculate mean and inverse of covariance of the "normal" distribution"""
        assert not self.normal_distribution is None and len(self.normal_distribution) > 0, \
            "Can't calculate mean or covariance of nothing!"
        
        self._mean = np.mean(self.normal_distribution, axis=0, dtype=np.float64)    # Mean
        cov = np.cov(self.normal_distribution, rowvar=False)                        # Covariance matrix
        self._covI = np.linalg.pinv(cov)                                            # Inverse of covariance matrix
    
    def _mahalanobis_distance(self, feature_vector):
        """Calculate the Mahalanobis distance between the input and the model"""
        assert not self._covI is None and not self._mean is None, \
            "You need to load a model before computing a Mahalanobis distance"

        assert feature_vector.shape[0] == self._mean.shape[0] == self._covI.shape[0] == self._covI.shape[1], \
            "Shapes don't match (x: %s, μ: %s, Σ¯¹: %s)" % (feature_vector.shape, self._mean.shape, self._covI.shape)
        
        return distance.mahalanobis(feature_vector, self._mean, self._covI)


    def generate_model(self, metadata, features):
        # Reduce features to simple list
        features_flat = self.reduce_feature_array(features)

        logging.info("Generating a Balanced Distribution from %i feature vectors of length %i" % (features_flat.shape[0], features_flat.shape[1]))

        assert features_flat.shape[0] > self.initial_normal_features, \
            "Not enough initial features provided. Please decrease initial_normal_features (%i)" % self.initial_normal_features

        with utils.GracefulInterruptHandler() as h:
            # Create initial set of "normal" vectors
            self.normal_distribution = features_flat[:self.initial_normal_features]

            start = time.time()

            self._calculate_mean_and_covariance()
            
            # loggin.info(np.mean(np.array([self._mahalanobis_distance(f) for f in features_flat])))

            utils.print_progress(0, 1, prefix = "%i / %i" % (self.initial_normal_features, features_flat.shape[0]))
            
            # Loop over the remaining feature vectors
            for index, feature_vector in enumerate(features_flat[self.initial_normal_features:]):
                if h.interrupted:
                    logging.warning("Interrupted!")
                    self.normal_distribution = None
                    return False

                # Calculate the Mahalanobis distance to the "normal" distribution
                dist = self._mahalanobis_distance(feature_vector)
                if dist > self.threshold_learning:
                    # Add the vector to the "normal" distribution
                    self.normal_distribution = np.append(self.normal_distribution, [feature_vector], axis=0)

                    # Recalculate mean and covariance
                    self._calculate_mean_and_covariance()
                
                # Print progress
                utils.print_progress(index + self.initial_normal_features + 1,
                                     features_flat.shape[0],
                                     prefix = "%i / %i" % (index + self.initial_normal_features + 1, features_flat.shape[0]),
                                     suffix = "%i vectors in Balanced Distribution" % len(self.normal_distribution),
                                     time_start = start)

            # Prune the distribution
            
            logging.info(np.mean(np.array([self._mahalanobis_distance(f) for f in self.normal_distribution])))

            prune_filter = []
            pruned = 0
            logging.info("Pruning Balanced Distribution")
            utils.print_progress(0, 1, prefix = "%i / %i" % (self.initial_normal_features, features_flat.shape[0]))
            start = time.time()

            for index, feature_vector in enumerate(self.normal_distribution):
                if h.interrupted:
                    logging.warning("Interrupted!")
                    self.normal_distribution = None
                    return False

                prune = self._mahalanobis_distance(feature_vector) < self.threshold_learning * self.pruning_parameter
                prune_filter.append(prune)

                if prune:
                    pruned += 1

                # Print progress
                utils.print_progress(index + 1,
                                     len(self.normal_distribution),
                                     prefix = "%i / %i" % (index + 1, len(self.normal_distribution)),
                                     suffix = "%i vectors pruned" % pruned,
                                     time_start = start)

            self.normal_distribution = self.normal_distribution[prune_filter]

            logging.info("Generated Balanced Distribution with %i entries" % len(self.normal_distribution))
        
            self._calculate_mean_and_covariance()
            return True

        
    def load_model_from_file(self, model_file):
        """Load a Balanced Distribution from file"""
        logging.info("Reading model parameters from: %s" % model_file)
        with h5py.File(model_file, "r") as hf:
            self.normal_distribution        = np.array(hf["normal_distribution"])
            self.initial_normal_features    = np.array(hf["initial_normal_features"])
            self.threshold_learning         = np.array(hf["threshold_learning"])
            self.threshold_classification   = np.array(hf["threshold_classification"])
            self.pruning_parameter          = np.array(hf["pruning_parameter"])
        assert 0 < self.pruning_parameter < 1, "Pruning parameter out of range (0 < η < 1)"
        self._calculate_mean_and_covariance()    
        logging.info("Successfully loaded Balanced Distribution with %i entries" % len(self.normal_distribution))
    
    def save_model_to_file(self, output_file = ""):
        """Save the model to disk"""
        logging.info("Writing model parameters to: %s" % output_file)
        with h5py.File(output_file, "w") as hf:
            hf.create_dataset("normal_distribution",        data=self.normal_distribution, dtype=np.float64)
            hf.create_dataset("initial_normal_features",    data=self.initial_normal_features, dtype=np.float64)
            hf.create_dataset("threshold_learning",         data=self.threshold_learning, dtype=np.float64)
            hf.create_dataset("threshold_classification",   data=self.threshold_classification, dtype=np.float64)
            hf.create_dataset("pruning_parameter",          data=self.pruning_parameter, dtype=np.float64)
        logging.info("Successfully written model parameters to: %s" % output_file)

# Only for tests
if __name__ == "__main__":
    model = AnomalyModelBalancedDistribution()
    model.load_model_from_file("/home/ludwig/ros/src/ROS-kate_bag/bags/real/TFRecord/Features/MobileNetV2_Block6.BalancedDistribution.h5")

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
    # if model.generate_model(metadata_no_anomaly, features_no_anomaly) == False:
    #     logging.info("Could not generate model.")

    # # Save model
    # model.save_model_to_file(os.path.abspath(features_file.replace(".h5", "")) + "." + model.NAME + ".h5")
    
    # features_flat = model.reduce_feature_array(features)
    # dists = np.array(list(map(model._mahalanobis_distance, features_flat)))
    thresh = 66#np.amax(dists)
    # print thresh

    def _feature_to_color(feature):
        b = 0#255 if feature in model.normal_distribution else 0
        g = 0
        # r = model._mahalanobis_distance(feature) * (255 / thresh)
        r = 100 if model._mahalanobis_distance(feature) > 25 else 0
        return (b, g, r)

    def _feature_to_text(feature):
        return round(model._mahalanobis_distance(feature), 2)

    def _pause(feature):
        return feature in model.normal_distribution

    model.visualize(metadata, features, _feature_to_color, feature_to_text_func=_feature_to_text)
    
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