import os
import time
import logging

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import anomaly_detector.consts as consts
from common import FeatureArray
import common.utils as utils

class AnomalyModelBase(object):
    
    def __init__(self):
        self.NAME = self.__class__.__name__.replace("AnomalyModel", "")
        self.features = None
    
    def __generate_model__(self, features):
        """Generate a model based on the features and metadata
        
        Args:
            features (FeatureArray): Array of features as extracted by a FeatureExtractor

        Returns:
            h5py.group that will be saved to the features file
        """
        raise NotImplementedError()
        
    def classify(self, feature):
        """Classify a single feature based on the loaded model
        
        Args:
            feature (Feature): A single feature

        Returns:
            A label
        """
        raise NotImplementedError()
    
    def __save_model_to_file__(self, h5file):
        """ Internal method that should be implemented by subclasses and save the
        necessary information to the file so that the model can be reloaded later
        """
        raise NotImplementedError()

    def __load_model_from_file__(self, h5file):
        """ Internal method that should be implemented by subclasses and load the
        necessary information from the file
        """
        raise NotImplementedError()
    
    ########################
    # Common functionality #
    ########################
    
    def load_or_generate(self, features=None,
                               load_features=False, load_mahalanobis_distances=False):
        """Load a model from file or generate it based on the features
        
        Args:
            features (str, FeatureArray) : HDF5 file containing metadata and features (see feature_extractor for details)
        """
        # Get default Features file if none is given
        if features is None:
            features = consts.FEATURES_FILE

        # Load features if necessary
        if isinstance(features, basestring):
            if features == "" or not os.path.exists(features) or not os.path.isfile(features):
                raise ValueError("Specified file does not exist (%s)" % features)
            
            # Try loading
            loaded = self.load_from_file(features, load_features=load_features, load_mahalanobis_distances=load_mahalanobis_distances)
            if loaded:
                return True

            # Read file
            if not isinstance(self.features, FeatureArray):
                features = FeatureArray(features)
        elif isinstance(features, FeatureArray):
            # Try loading
            loaded = self.load_from_file(features.filename, load_features=load_features, load_mahalanobis_distances=load_mahalanobis_distances)
            if loaded:
                return True
        else:
            raise ValueError("features must be a path to a file or a FeatureArray.")
        
        self.features = features

        start = time.time()

        # Generate model
        if self.__generate_model__(self.features.no_anomaly) == False:
            logging.info("Could not generate model.")
            return False

        end = time.time()

        logging.info("Writing model to: %s" % self.features.filename)
        with h5py.File(self.features.filename, "a") as hf:
            g = hf.get(self.NAME)

            if g is not None:
                del hf[self.NAME]
            
            g = hf.create_group(self.NAME)

            # Add metadata to the output file
            g.attrs["Number of features used"]   = len(self.features.flatten())

            computer_info = utils.getComputerInfo()
            for key, value in computer_info.items():
                g.attrs[key] = value

            g.attrs["Start"] = start
            g.attrs["End"] = end
            g.attrs["Duration"] = end - start
            g.attrs["Duration (formatted)"] = utils.format_duration(end - start)

            self.__save_model_to_file__(g)
        logging.info("Successfully written model to: %s" % self.features.filename)

        if load_mahalanobis_distances:
            self.calculate_mahalobis_distances()

        return True

    def load_from_file(self, model_file, load_features=False, load_mahalanobis_distances=False):
        """ Load a model from file """
        logging.info("Reading model from: %s" % model_file)
        with h5py.File(model_file, "r") as hf:
            g = hf.get(self.NAME)
            
            if g is None:
                return False

            if load_features:
                self.features = FeatureArray(model_file)

            if load_mahalanobis_distances:
                self.load_mahalanobis_distances()
                
            return self.__load_model_from_file__(g)
    
    def calculate_mahalobis_distances(self):
        """ Calculate all the Mahalanobis distances """
        if self.features is None:
            raise ValueError("No features loaded.")
        
        # Try loading
        if self.load_mahalanobis_distances():
            return True

        logging.info("Calculating Mahalanobis distances of %i features with and %i features without anomalies" % \
            (len(self.features.no_anomaly.flatten()), len(self.features.anomaly.flatten())))

        self.mahalanobis_no_anomaly = np.array(list(map(self._mahalanobis_distance, tqdm(self.features.no_anomaly.flatten(), desc="No anomaly  ")))) # 75.49480115577167
        self.mahalanobis_anomaly    = np.array(list(map(self._mahalanobis_distance, tqdm(self.features.anomaly.flatten(), desc="With anomaly"))))    # 76.93620254133627

        self.mahalanobis_no_anomaly_max = np.nanmax(self.mahalanobis_no_anomaly) if len(self.mahalanobis_no_anomaly) > 0 else np.nan
        self.mahalanobis_anomaly_max    = np.nanmax(self.mahalanobis_anomaly) if len(self.mahalanobis_anomaly) > 0 else np.nan
        
        self.mahalanobis_max = np.nanmax([self.mahalanobis_no_anomaly_max, self.mahalanobis_anomaly_max])

        logging.info("Maximum Mahalanobis distance (no anomaly): %f" % self.mahalanobis_no_anomaly_max)
        logging.info("Maximum Mahalanobis distance (anomaly)   : %f" % self.mahalanobis_anomaly_max)
        
        self.save_mahalanobis_distances()

    def load_mahalanobis_distances(self):
        """ Load the mahalanobis distances from file """
        with h5py.File(self.features.filename, "r") as hf:
            g = hf.get(self.NAME)
            
            if g is None:
                return False
            
            if not "mahalanobis_no_anomaly" in g.keys():
                return False

            self.mahalanobis_no_anomaly = np.array(g["mahalanobis_no_anomaly"])
            self.mahalanobis_anomaly    = np.array(g["mahalanobis_anomaly"])
            
            self.mahalanobis_no_anomaly_max = g["mahalanobis_no_anomaly"].attrs["max"]
            self.mahalanobis_anomaly_max    = g["mahalanobis_anomaly"].attrs["max"]
            self.mahalanobis_max = np.nanmax((self.mahalanobis_no_anomaly_max, self.mahalanobis_anomaly_max))

            logging.info("Maximum Mahalanobis distance (no anomaly): %f" % self.mahalanobis_no_anomaly_max)
            logging.info("Maximum Mahalanobis distance (anomaly)   : %f" % self.mahalanobis_anomaly_max)
            return True

    def save_mahalanobis_distances(self):
        """ Save the mahalanobis distances from file """
        with h5py.File(self.features.filename, "r+") as hf:
            g = hf.get(self.NAME)

            if g is None:
                g = hf.create_group(self.NAME)
            
            no_anomaly = g.create_dataset("mahalanobis_no_anomaly",  data=self.mahalanobis_no_anomaly)
            anomaly    = g.create_dataset("mahalanobis_anomaly",     data=self.mahalanobis_anomaly)
            
            no_anomaly.attrs["max"] = self.mahalanobis_no_anomaly_max
            anomaly.attrs["max"]    = self.mahalanobis_anomaly_max

            logging.info("Saved Mahalanobis distances to file")
            return True

    def show_mahalanobis_distribution(self):
        """ Plot the distribution of all Mahalanobis distances """
        logging.info("Showing Mahalanobis distance distribution")
        if self.mahalanobis_no_anomaly is None:
            self.calculate_mahalobis_distances()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.set_title("No anomaly (%i)" % len(self.features.no_anomaly.flatten()))
        ax1.hist(self.mahalanobis_no_anomaly, bins=50)

        ax2.set_title("Anomaly (%i)" % len(self.features.anomaly.flatten()))
        ax2.hist(self.mahalanobis_anomaly, bins=50)

        fig.suptitle("Mahalanobis distances")

        plt.show()

    def visualize(self, images_path=None, threshold=None, feature_to_color_func=None, feature_to_text_func=None, pause_func=None):
        """ Visualize the result of a anomaly model """
        if threshold is None:
            if not hasattr(self, "mahalanobis_max") or self.mahalanobis_max is None:
                self.calculate_mahalobis_distances()
            threshold = self.mahalanobis_max * 0.9
        
        def _default_feature_to_color(feature, t, show_thresh):
            b = 0#100 if feature in self.normal_distribution else 0
            g = 0
            if show_thresh:
                r = 100 if self._mahalanobis_distance(feature) > t else 0
            elif t == 0:
                r = 0
            else:
                r = min(255, int(self._mahalanobis_distance(feature) * (255 / t)))
            return (b, g, r)

        def _default_feature_to_text(feature, t):
            return round(self._mahalanobis_distance(feature), 2)

        if feature_to_color_func is None:
            feature_to_color_func = _default_feature_to_color

        if feature_to_text_func is None:
            feature_to_text_func = _default_feature_to_text

        utils.visualize(self.features,
                        images_path=images_path,
                        threshold=threshold,
                        feature_to_color_func=feature_to_color_func,
                        feature_to_text_func=feature_to_text_func,
                        pause_func=pause_func)