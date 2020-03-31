import os
import time
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import consts
from common import PatchArray, Visualize, utils, logger

class AnomalyModelBase(object):
    
    def __init__(self):
        self.NAME = self.__class__.__name__.replace("AnomalyModel", "")
        self.patches = None
    
    def __generate_model__(self, patches):
        """Generate a model based on the features and metadata
        
        Args:
            patches (PatchArray): Array of patches with features as extracted by a FeatureExtractor

        Returns:
            h5py.group that will be saved to the features file
        """
        raise NotImplementedError()
        
    def __mahalanobis_distance__(self, patch):
        """Calculate the Mahalanobis distance between the input and the model"""
        raise NotImplementedError()
        
    def classify(self, patch):
        """Classify a single feature based on the loaded model
        
        Args:
            patch (np.record): A single patch (with feature)

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
    
    def load_or_generate(self, patches=consts.FEATURES_FILE,
                               load_patches=False, load_mahalanobis_distances=False):
        """Load a model from file or generate it based on the features
        
        Args:
            patches (str, PatchArray) : HDF5 file containing features (see feature_extractor for details)
        """
        
        # Load patches if necessary
        if isinstance(patches, basestring):
            if patches == "" or not os.path.exists(patches) or not os.path.isfile(patches):
                raise ValueError("Specified file does not exist (%s)" % patches)
            
            # Try loading
            loaded = self.load_from_file(patches, load_patches=load_patches, load_mahalanobis_distances=load_mahalanobis_distances)
            if loaded:
                return True

            # Read file
            if not isinstance(self.patches, PatchArray):
                patches = PatchArray(patches)
        elif isinstance(patches, PatchArray):
            # Try loading
            loaded = self.load_from_file(patches.filename, load_patches=load_patches, load_mahalanobis_distances=load_mahalanobis_distances)
            if loaded:
                return True
        else:
            raise ValueError("patches must be a path to a file or a PatchArray.")
        
        assert patches.contains_features, "patches must contain features to calculate an anomaly model."

        self.patches = patches

        start = time.time()

        # Generate model
        if self.__generate_model__(self.patches.no_anomaly) == False:
            logger.info("Could not generate model.")
            return False

        end = time.time()

        logger.info("Writing model to: %s" % self.patches.filename)
        with h5py.File(self.patches.filename, "a") as hf:
            g = hf.get(self.NAME)

            if g is not None:
                del hf[self.NAME]
            
            g = hf.create_group(self.NAME)

            # Add metadata to the output file
            g.attrs["Number of features used"]   = len(self.patches.ravel())

            computer_info = utils.getComputerInfo()
            for key, value in computer_info.items():
                g.attrs[key] = value

            g.attrs["Start"] = start
            g.attrs["End"] = end
            g.attrs["Duration"] = end - start
            g.attrs["Duration (formatted)"] = utils.format_duration(end - start)

            self.__save_model_to_file__(g)
        logger.info("Successfully written model to: %s" % self.patches.filename)

        if load_mahalanobis_distances:
            self.calculate_mahalobis_distances()

        return True

    def load_from_file(self, model_file, load_patches=False, load_mahalanobis_distances=False):
        """ Load a model from file """
        logger.info("Reading model from: %s" % model_file)
        with h5py.File(model_file, "r") as hf:
            g = hf.get(self.NAME)
            
            if g is None:
                return False

            if load_patches:
                self.patches = PatchArray(model_file)

            if load_mahalanobis_distances:
                self.load_mahalanobis_distances()
                
            return self.__load_model_from_file__(g)
    
    def calculate_mahalobis_distances(self):
        """ Calculate all the Mahalanobis distances """
        if self.patches is None:
            raise ValueError("No patches loaded.")
        
        # Try loading
        if self.load_mahalanobis_distances():
            return True

        logger.info("Calculating Mahalanobis distances of %i features with and %i features without anomalies" % \
            (len(self.patches.no_anomaly.ravel()), len(self.patches.anomaly.ravel())))

        self.mahalanobis_no_anomaly = np.array(list(map(self.__mahalanobis_distance__, tqdm(np.nditer(self.patches.no_anomaly), desc="No anomaly  ", file=sys.stderr)))) # 75.49480115577167
        self.mahalanobis_anomaly    = np.array(list(map(self.__mahalanobis_distance__, tqdm(np.nditer(self.patches.anomaly), desc="With anomaly", file=sys.stderr))))    # 76.93620254133627

        self.mahalanobis_no_anomaly_max = np.nanmax(self.mahalanobis_no_anomaly) if len(self.mahalanobis_no_anomaly) > 0 else np.nan
        self.mahalanobis_anomaly_max    = np.nanmax(self.mahalanobis_anomaly) if len(self.mahalanobis_anomaly) > 0 else np.nan
        
        self.mahalanobis_max = np.nanmax([self.mahalanobis_no_anomaly_max, self.mahalanobis_anomaly_max])

        logger.info("Maximum Mahalanobis distance (no anomaly): %f" % self.mahalanobis_no_anomaly_max)
        logger.info("Maximum Mahalanobis distance (anomaly)   : %f" % self.mahalanobis_anomaly_max)
        
        self.save_mahalanobis_distances()

    def load_mahalanobis_distances(self):
        """ Load the mahalanobis distances from file """
        with h5py.File(self.patches.filename, "r") as hf:
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

            logger.info("Maximum Mahalanobis distance (no anomaly): %f" % self.mahalanobis_no_anomaly_max)
            logger.info("Maximum Mahalanobis distance (anomaly)   : %f" % self.mahalanobis_anomaly_max)
            return True

    def save_mahalanobis_distances(self):
        """ Save the mahalanobis distances from file """
        with h5py.File(self.patches.filename, "r+") as hf:
            g = hf.get(self.NAME)

            if g is None:
                g = hf.create_group(self.NAME)
            
            no_anomaly = g.create_dataset("mahalanobis_no_anomaly",  data=self.mahalanobis_no_anomaly)
            anomaly    = g.create_dataset("mahalanobis_anomaly",     data=self.mahalanobis_anomaly)
            
            no_anomaly.attrs["max"] = self.mahalanobis_no_anomaly_max
            anomaly.attrs["max"]    = self.mahalanobis_anomaly_max

            logger.info("Saved Mahalanobis distances to file")
            return True

    def show_mahalanobis_distribution(self):
        """ Plot the distribution of all Mahalanobis distances """
        logger.info("Showing Mahalanobis distance distribution")
        if self.mahalanobis_no_anomaly is None:
            self.calculate_mahalobis_distances()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.set_title("No anomaly (%i)" % len(self.patches.no_anomaly.ravel()))
        ax1.hist(self.mahalanobis_no_anomaly, bins=50)

        ax2.set_title("Anomaly (%i)" % len(self.patches.anomaly.ravel()))
        ax2.hist(self.mahalanobis_anomaly, bins=50)

        fig.suptitle("Mahalanobis distances")

        plt.show()

    def visualize(self, **kwargs):
        """ Visualize the result of a anomaly model """

        if "threshold" not in kwargs:
            if not hasattr(self, "mahalanobis_max") or self.mahalanobis_max is None:
                self.calculate_mahalobis_distances()
            kwargs["threshold"] = self.mahalanobis_max * 0.9
        
        if "patch_to_color_func" not in kwargs:
            def _default_patch_to_color(v, feature):
                b = 0#100 if feature in self.normal_distribution else 0
                g = 0
                threshold = v.get_trackbar("threshold")
                if v.get_trackbar("show_thresh"):
                    r = 100 if self.__mahalanobis_distance__(feature) > threshold else 0
                elif threshold == 0:
                    r = 0
                else:
                    r = min(255, int(self.__mahalanobis_distance__(feature) * (255 / threshold)))
                return (b, g, r)
            kwargs["patch_to_color_func"] = _default_patch_to_color

        if "patch_to_text_func" not in kwargs:
            def _default_patch_to_text(v, feature):
                return round(self.__mahalanobis_distance__(feature), 2)
            kwargs["patch_to_text_func"] = _default_patch_to_text

        vis = Visualize(self.patches, **kwargs)

        vis.create_trackbar("threshold", kwargs["threshold"], kwargs["threshold"] * 3)
        vis.create_trackbar("show_thresh", 1, 1)
        
        vis.show()