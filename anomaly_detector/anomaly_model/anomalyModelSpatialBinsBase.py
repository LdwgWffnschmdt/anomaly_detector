# -*- coding: utf-8 -*-

import os
import time
import logging
import traceback

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from anomalyModelBase import AnomalyModelBase
import common.utils as utils

class AnomalyModelSpatialBinsBase(AnomalyModelBase):
    """ Base for anomaly models that create one model per spatial bin (grid cell) """
    def __init__(self, create_anomaly_model_func, cell_size=0.2):
        """ Create a new spatial bin anomaly model

        Args:
            create_anomaly_model_func (func): Method that returns an anomaly model
            cell_size (float): Width and height of spatial bin in meter
        """
        AnomalyModelBase.__init__(self)
        self.CELL_SIZE = cell_size
        self.CREATE_ANOMALY_MODEL_FUNC = create_anomaly_model_func
        
        m = create_anomaly_model_func()
        self.NAME = "SpatialBin/%s/%.2f" % (m.__class__.__name__.replace("AnomalyModel", ""), cell_size)
    
    def classify(self, metadata, feature_vector, threshold=None):
        """The anomaly measure is defined as the Mahalanobis distance between a feature sample
        and the single variate Gaussian distribution along each dimension.
        """
        assert feature.location is not None, "The feature needs a location to find its bins"

        self.get_bin(feature)
        
        model = self.get_model(feature.bin)
        if model is None:
            logging.warning("No model available for this bin (%i, %i)" % feature.bin)
            return 0 # Unknown
        
        return model.classify(feature)
    
    def _mahalanobis_distance(self, feature):
        """Calculate the Mahalanobis distance between the input and the model"""
        assert feature.location is not None, "The feature needs a location to find its bins"
        
        self.get_bin(feature)
        
        model = self.get_model(feature.bin)
        if model is None:
            logging.warning("No model available for this bin (%i, %i)" % feature.bin)
            return -1 # TODO: What should we do?
        
        return model._mahalanobis_distance(feature)

    def __generate_model__(self, features):
        # Get location for features
        features.calculate_locations()
        
        # Reduce features to simple list
        features_flat = features.flatten()
        
        # Get extent
        x_min = features[0,0,0].location[0]
        y_min = features[0,0,0].location[1]
        x_max = features[0,0,0].location[0]
        y_max = features[0,0,0].location[1]

        for f in features_flat:
            if f.location[0] > x_max:
                x_max = f.location[0]
            if f.location[0] < x_min:
                x_min = f.location[0]

            if f.location[1] > y_max:
                y_max = f.location[1]
            if f.location[1] < y_min:
                y_min = f.location[1]
        
        # Round to cell size (increases bounds to fit next cell size)
        x_min -= x_min % self.CELL_SIZE
        y_min -= y_min % self.CELL_SIZE
        x_max += self.CELL_SIZE - (x_max % self.CELL_SIZE)
        y_max += self.CELL_SIZE - (y_max % self.CELL_SIZE)

        # Create the bins (Tuple(bins_x, bins_y))
        self.bins = (np.arange(x_min, x_max, self.CELL_SIZE),
                     np.arange(y_min, y_max, self.CELL_SIZE))
        
        shape = (len(self.bins[0]) + 1, len(self.bins[1]) + 1)

        logging.info("Extent of features: (%i, %i)/(%i, %i)." % (x_min, y_min, x_max, y_max))
        logging.info("Results in %i bins in x and %i bins in y direction (with cell size %.2f)" % (len(self.bins[0]), len(self.bins[1]), self.CELL_SIZE))

        logging.info("Calculating corresponding bin for every feature")
        self.feature_indices = np.empty(shape=shape, dtype=object)
        self.feature_indices_count = np.zeros(shape=shape, dtype=np.uint32)
        # Get the corresponding bin for every feature
        for i, f in enumerate(tqdm(features_flat, desc="Calculating bins")):
            self.get_bin(f)
            if self.feature_indices[f.bin] is None:
                self.feature_indices[f.bin] = list()
            self.feature_indices[f.bin].append(i)
            self.feature_indices_count[f.bin] += 1
        
        # Empty grid that will contain the model for each bin
        self.models = np.empty(shape=shape, dtype=object)
        total = self.models.shape[0] * self.models.shape[1]

        with utils.GracefulInterruptHandler() as h:
            start = time.time()
            utils.print_progress(0, total, prefix="Generating models:",
                                           suffix = "%i / %i" % (0, total))
            
            i = 0
            try:
                for u in range(shape[0]):
                    for v in range(shape[1]):
                        i += 1

                        if h.interrupted:
                            logging.warning("Interrupted!")
                            raise KeyboardInterrupt()
                        
                        if self.feature_indices[u, v] is not None:
                            # Filter by bin
                            bin_features_flat = features_flat[self.feature_indices[u, v]]
                            
                            if len(bin_features_flat) > 0:
                                # Create a new model
                                model = self.CREATE_ANOMALY_MODEL_FUNC() # Instantiate a new model
                                model.__generate_model__(bin_features_flat)  # The model only gets flattened features
                                self.models[u, v] = model                # Store the model
                        
                        # Print progress
                        utils.print_progress(i, total,
                                                prefix = "Generating models:",
                                                suffix = "%i / %i" % (i, total),
                                                time_start = start)
            except:
                utils.print_progress(i, total,
                                        prefix = "Error:",
                                        suffix = "%i / %i" % (i, total),
                                        time_start = start)
                traceback.print_exc()
                return False
        return True
    
    def get_bin(self, feature):
        """Gets the indices for the bin the given feature belongs to

        Args:
            feature (Feature): Feature
        
        Returns:
            Tuple containing the bin indices (u, v)
        """
        assert self.bins is not None, "Can't calculate bin index when no bins were computed before."
        assert feature.location is not None, "Feature locations need to be computed before computing bins"

        if hasattr(feature, "bin") and feature.bin is not None:
            return feature.bin
        else:
            feature.bin = (np.digitize(feature.location[0], self.bins[0]),
                           np.digitize(feature.location[1], self.bins[1]))
            return feature.bin

    def get_model(self, bin):
        """ Return the respective model for the given bin """
        assert self.models is not None, "Please generate the models first."
        return self.models[bin[0], bin[1]]

    def __load_model_from_file__(self, h5file):
        """Load a SVG model from file"""
        if not "bins_x" in h5file.keys() or \
           not "bins_y" in h5file.keys() or \
           not "feature_indices_count" in h5file.keys() or \
           not "Cell size" in h5file.attrs.keys():
            return False
        
        self.CELL_SIZE = h5file.attrs["Cell size"]
        
        self.bins = (np.array(h5file["bins_x"]), np.array(h5file["bins_y"]))
        self.feature_indices_count = np.array(h5file["feature_indices_count"])
        
        self.models = np.empty(shape=(len(self.bins[0]) + 1, len(self.bins[1]) + 1), dtype=object)

        def _add_model(name, g):
            if "u" in g.attrs.keys() and "v" in g.attrs.keys():
                u = g.attrs["u"]
                v = g.attrs["v"]
                self.models[u, v] = self.CREATE_ANOMALY_MODEL_FUNC()
                self.models[u, v].__load_model_from_file__(g)

        h5file.visititems(_add_model)
        return True
    
    def __save_model_to_file__(self, h5file):
        """Save the model to disk"""
        h5file.attrs["Cell size"] = self.CELL_SIZE
        h5file.attrs["Num models"] = sum(x is not None for x in self.models.flatten())
        h5file.attrs["Models shape"] = self.models.shape
        h5file.create_dataset("bins_x", data=self.bins[0])
        h5file.create_dataset("bins_y", data=self.bins[1])
        h5file.create_dataset("feature_indices_count", data=self.feature_indices_count)
        
        for u, v in tqdm(np.ndindex(self.models.shape), desc="Saving models"):
            model = self.get_model((u, v))
            if model is not None:
                g = h5file.create_group("%i/%i" % (u, v))
                g.attrs["u"] = u
                g.attrs["v"] = v
                model.__save_model_to_file__(g)
        return True
    	
    def show_spatial_histogram(self):
        """ Show the spatial histogram of features (like a map) """
        plt.imshow(self.feature_indices_count, vmin=1)
        plt.show()
        
# Only for tests
if __name__ == "__main__":
    from anomalyModelSVG import AnomalyModelSVG
    
    model = AnomalyModelSpatialBinsBase(AnomalyModelSVG)
    if model.load_or_generate(load_features=True):
        model.show_spatial_histogram()

        model.calculate_mahalobis_distances()
        model.show_mahalanobis_distribution()

        model.visualize()