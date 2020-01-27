import os
import h5py

import feature_extractor.utils as utils

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
            features_file (str) : HDF5 file containing metadata and features (see feature_extractor for details)
            output_file (str): Output path for the model file (same path as features_file if not specified)
        """
        # Check parameters
        if features_file == "" or not os.path.exists(features_file) or not os.path.isfile(features_file):
            rospy.logerr("Specified model file does not exist (%s)" % features_file)
            return
        
        if output_file == "":
            output_file = os.path.abspath(features_file.replace(".h5", "")) + "." + self.NAME + ".h5"
            print("Output file set to %s" % output_file)
        
        # Read file
        metadata, features = utils.read_hdf5(features_file)

        # Only take feature vectors of images labeled as anomaly free (label == 1)
        features = features[[m["label"] == 1 for m in metadata]]
        
        # Generate model
        if self.generate_model(metadata, features) == False:
            print("Could not generate model.")
            return False

        # Save model
        self.save_model_to_file(output_file)
        
        return True

    def reduce_feature_array(self, features_vector_array):
        """Reduce an array of feature vectors of shape to a simple list
        
        Args:
            features_vector_array (object[]): feature vectors array
        """
        # Create an array of only the feature vectors, eg. (25000, 1280)
        return features_vector_array.reshape(-1, features_vector_array.shape[-1])
