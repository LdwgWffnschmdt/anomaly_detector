import os
import h5py

import feature_extractor.utils as utils

class AnomalyModelBase(object):
    
    def __init__(self):
        self.name = ""          # Should be set by the implementing class
    
    def classify(self, feature_vector):
        """ Classify a single feature vector based on the loaded model """
        pass
    
    def load_model_from_file(self, model_file):
        """ Load a model from file """
        pass

    def generate_model(self, locations, features):
        """
        Generate a model based on the features and locations
        @params:
            locations   - Required  : Array of locations for the features
            features    - Required  : Array of features as extracted by a FeatureExtractor
        """
        pass
        
    def save_model_to_file(self, output_file):
        """
        Save the model to output_file
        @params:
            output_file     - Required  : Output path for the model file
        """
        pass

    ########################
    # Common functionality #
    ########################

    def generate_model_from_file(self, features_file, output_file = ""):
        """
        Generate a model based on the features in features_file and save it to output_file
        @params:
            features_file   - Required  : HDF5 file containing locations and features (see feature_extractor for details)
            output_file     - Optional  : Output path for the model file (same path as features_file if not specified)
        """
        # Check parameters
        if features_file == "" or not os.path.exists(features_file) or not os.path.isfile(features_file):
            rospy.logerr("Specified model file does not exist (%s)" % features_file)
            return
        
        if output_file == "":
            output_file = os.path.abspath(features_file.replace(".h5", "")) + "AnomalyModel" + self.name + ".h5"
            print("Output file set to %s" % output_file)
        
        # Read file
        locations, features = utils.read_hdf5(features_file)

        # Generate model
        self.generate_model(locations, features)

        # Save model
        self.save_model_to_file(output_file)

    def reduce_feature_array(self, features_vector_array):
        """
        Reduce an array of feature vectors of shape to a simple list
        @params:
            features_vector_array   - Required  : feature vectors array
        """
        # Create an array of only the feature vectors, eg. (25000, 1280)
        return features_vector_array.reshape(-1, features_vector_array.shape[-1])
