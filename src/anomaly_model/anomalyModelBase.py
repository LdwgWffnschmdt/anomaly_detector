import os
import logging
import h5py
import cv2

import numpy as np
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import feature_extractor.utils as utils
from IPM import IPM, _DictObjHolder

class AnomalyModelBase(object):
    
    def __init__(self):
        self.NAME = ""          # Should be set by the implementing class
    
    def generate_model(self, metadata, features):
        """Generate a model based on the features and metadata
        
        Args:
            metadata (list): Array of metadata for the features
            features (list): Array of features as extracted by a FeatureExtractor
        """
        raise NotImplementedError
        
    def classify(self, feature_vector):
        """ Classify a single feature vector based on the loaded model """
        raise NotImplementedError
    
    def load_model_from_file(self, model_file):
        """ Load a model from file """
        raise NotImplementedError

    def save_model_to_file(self, output_file):
        """Save the model to output_file
        
        Args:
            output_file (str): Output path for the model file
        """
        raise NotImplementedError

    ########################
    # Common functionality #
    ########################

    def generate_model_from_file(self, features_file, output_file = ""):
        """Generate a model based on the features in features_file and save it to output_file
        
        Args:
            features_file (str) : HDF5 or TFRecord file containing metadata and features (see feature_extractor for details)
            output_file (str): Output path for the model file (same path as features_file if not specified)
        """
        # Check parameters
        if features_file == "" or not os.path.exists(features_file) or not os.path.isfile(features_file):
            raise ValueError("Specified model file does not exist (%s)" % features_file)
        
        if output_file == "":
            output_file = os.path.abspath(features_file.replace(".h5", "")) + "." + self.NAME + ".h5"
            logging.info("Output file set to %s" % output_file)
        
        # Read file
        file_content = utils.read_features_file(features_file)

        # Generate model
        if self.generate_model(file_content["no_anomaly"].metadata,
                               file_content["no_anomaly"].features) == False:
            logging.info("Could not generate model.")
            return False

        # Save model
        self.save_model_to_file(output_file)
        
        return True