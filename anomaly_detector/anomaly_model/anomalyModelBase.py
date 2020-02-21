import os
import logging

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
        raise NotImplementedError
        
    def classify(self, feature):
        """Classify a single feature based on the loaded model
        
        Args:
            feature (Feature): A single feature

        Returns:
            A label
        """
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
            features_file (str) : HDF5 file containing metadata and features (see feature_extractor for details)
            output_file (str): Output path for the model file (same path as features_file if not specified)
        """
        # Check parameters
        if features_file == "" or not os.path.exists(features_file) or not os.path.isfile(features_file):
            raise ValueError("Specified model file does not exist (%s)" % features_file)
        
        if output_file == "":
            output_file = os.path.abspath(features_file.replace(".h5", "")) + "." + self.__name__ + ".h5"
            logging.info("Output file set to %s" % output_file)
        
        # Read file
        features = FeatureArray(features_file)

        # Generate model
        if self.generate_model(features.no_anomaly) == False:
            logging.info("Could not generate model.")
            return False

        # Save model
        self.save_model_to_file(output_file)
        
        return True