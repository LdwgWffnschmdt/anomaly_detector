import os
import time
import logging

import h5py

from common import FeatureArray
import common.utils as utils

class AnomalyModelBase(object):
    
    def __init__(self):
        self.NAME = self.__class__.__name__.replace("AnomalyModel", "")
    
    def generate_model(self, features):
        """Generate a model based on the features and metadata
        
        Args:
            features (FeaturesArray): Array of features as extracted by a FeatureExtractor
        """
        self.features = features
        
    def classify(self, feature):
        """Classify a single feature based on the loaded model
        
        Args:
            feature (Feature): A single feature

        Returns:
            A label
        """
        raise NotImplementedError
    
    def __save_model_to_file__(self, h5file):
        """ Internal method that should be implemented by subclasses and save the
        necessary information to the file so that the model can be reloaded later
        """
        raise NotImplementedError

    def __load_model_from_file__(self, h5file):
        """ Internal method that should be implemented by subclasses and load the
        necessary information from the file
        """
        raise NotImplementedError
    
    ########################
    # Common functionality #
    ########################

    def generate_model_from_file(self, features_file, output_file = ""):
        """Generate a model based on the features in features_file and save it to output_file
        
        Args:
            features_file (str) : HDF5 file containing metadata and features (see feature_extractor for details)
            output_file (str): Output path for the model file (same path as features_file if not specified)
        """
        # Check parameters
        if features_file == "" or not os.path.exists(features_file) or not os.path.isfile(features_file):
            raise ValueError("Specified model file does not exist (%s)" % features_file)
        
        # Read file
        features = FeatureArray(features_file)

        if output_file == "":
            output_dir = os.path.join(os.path.abspath(os.path.dirname(features_file)), self.NAME)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, os.path.basename(features_file))
            logging.info("Output file set to %s" % output_file)
        
        # Generate model
        if self.generate_model(features.no_anomaly) == False:
            logging.info("Could not generate model.")
            return False

        # Save model
        self.save_model_to_file(output_file)
        
        return True
        
    def save_to_file(self, output_file):
        """Save the model to output_file
        
        Args:
            output_file (str): Output path for the model file
        """
        
        logging.info("Writing model to: %s" % output_file)
        with h5py.File(output_file, "w") as hf:
            # Add metadata to the output file
            hf.attrs["Anomaly model name"]        = self.NAME
            if self.features is None:
                logging.warn("You are saving a model seems to have been loaded"
                             "from file and is not freshly generated from features")
                hf.attrs["No features loaded"]   = True
            else:
                hf.attrs["Number of features used"]   = len(self.features.flatten())
                hf.attrs["Features shape"]            = self.features.shape
                hf.attrs["Features file"]             = self.features.filename
                hf.attrs["Feature extractor"]         = self.features.extractor
                hf.attrs["Feature batch size"]        = self.features.batch_size
                hf.attrs["Original files"]            = self.features.files

            computer_info = utils.getComputerInfo()
            for key, value in computer_info.items():
                hf.attrs[key] = value

            hf.attrs["Created"] = time.time()

            self.__save_model_to_file__(hf)
        logging.info("Successfully written model to: %s" % output_file)


    def load_from_file(self, model_file, load_features=False):
        """ Load a model from file """
        logging.info("Reading model from: %s" % model_file)
        with h5py.File(model_file, "r") as hf:
            if hf.attrs["Anomaly model name"] != self.NAME:
                logging.warn("The model you are trying to load does not seem to have"
                             "been created by this anomaly detector (%s vs %s)" % (hf.attrs["Anomaly model name"], self.NAME))
            if load_features:
                features_file = hf.attrs["Features file"]
                if os.path.exists(features_file):
                    self.features = FeaturesArray(features_file)
                else:
                    raise ValueError("Can't find features file (%s)" % features_file)
            self.__load_model_from_file__(hf)