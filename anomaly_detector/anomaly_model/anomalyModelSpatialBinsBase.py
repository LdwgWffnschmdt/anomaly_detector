# -*- coding: utf-8 -*-

import os
import time
import traceback
import sys

import h5py
import numpy as np
from tqdm import tqdm

from anomalyModelBase import AnomalyModelBase
from common import utils, logger
from common import FeatureArray

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
    
    def classify(self, feature, threshold=None):
        """The anomaly measure is defined as the Mahalanobis distance between a feature sample
        and the single variate Gaussian distribution along each dimension.
        """
        assert feature.location is not None, "The feature needs a location to find its bins"

        model = self.get_model(feature.get_bin(self.CELL_SIZE))
        if model is None:
            logger.warning("No model available for this bin (%i, %i)" % feature.get_bin(self.CELL_SIZE))
            return 0 # Unknown
        
        return model.classify(feature)
    
    def _mahalanobis_distance(self, feature):
        """Calculate the Mahalanobis distance between the input and the model"""
        assert feature.location is not None, "The feature needs a location to find its bins"
        
        model = self.get_model(feature.get_bin(self.CELL_SIZE))
        if model is None:
            logger.warning("No model available for this bin (%i, %i)" % feature.get_bin(self.CELL_SIZE))
            return -1 # TODO: What should we do?
        
        return model._mahalanobis_distance(feature)

    def __generate_model__(self, features):
        # Ensure locations are calculated
        features.ensure_locations()
        
        shape = features.calculate_rasterization(self.CELL_SIZE)

        # Empty grid that will contain the model for each bin
        self.models = np.empty(shape=shape, dtype=object)

        for bin in tqdm(np.ndindex(shape), desc="Generating models", total=np.prod(shape), file=sys.stderr):
            bin_features_flat = features.bin(bin, self.CELL_SIZE)
            
            if len(bin_features_flat) > 0:
                # Create a new model
                model = self.CREATE_ANOMALY_MODEL_FUNC() # Instantiate a new model
                model.__generate_model__(bin_features_flat)  # The model only gets flattened features
                self.models[bin] = model                # Store the model
        return True
    
    def get_model(self, bin):
        """ Return the respective model for the given bin """
        assert self.models is not None, "Please generate the models first."
        return self.models[bin[0], bin[1]]

    def __load_model_from_file__(self, h5file):
        """Load a SVG model from file"""
        if not "Models shape" in h5file.attrs.keys() or \
           not "Num models" in h5file.attrs.keys() or \
           not "Cell size" in h5file.attrs.keys():
            return False
        
        self.CELL_SIZE = h5file.attrs["Cell size"]
        
        self.models = np.empty(shape=h5file.attrs["Models shape"], dtype=object)

        with tqdm(desc="Loading models", total=h5file.attrs["Num models"], file=sys.stderr) as pbar:
            def _add_model(name, g):
                if "u" in g.attrs.keys() and "v" in g.attrs.keys():
                    u = g.attrs["u"]
                    v = g.attrs["v"]
                    self.models[u, v] = self.CREATE_ANOMALY_MODEL_FUNC()
                    self.models[u, v].__load_model_from_file__(g)
                    pbar.update()

            h5file.visititems(_add_model)
        
        if isinstance(self.features, FeatureArray):
            self.features.calculate_rasterization(self.CELL_SIZE)

        return True
    
    def __save_model_to_file__(self, h5file):
        """Save the model to disk"""
        h5file.attrs["Cell size"] = self.CELL_SIZE
        h5file.attrs["Num models"] = sum(x is not None for x in self.models.flatten())
        h5file.attrs["Models shape"] = self.models.shape
        
        for u, v in tqdm(np.ndindex(self.models.shape), desc="Saving models", total=np.prod(self.models.shape), file=sys.stderr):
            model = self.get_model((u, v))
            if model is not None:
                g = h5file.create_group("%i/%i" % (u, v))
                g.attrs["u"] = u
                g.attrs["v"] = v
                model.__save_model_to_file__(g)
        return True
    	
        
# Only for tests
if __name__ == "__main__":
    from anomalyModelSVG import AnomalyModelSVG
    
    model = AnomalyModelSpatialBinsBase(AnomalyModelSVG)
    if model.load_or_generate(load_features=True):
        # model.show_spatial_histogram()

        # model.calculate_mahalobis_distances()
        # model.show_mahalanobis_distribution()

        model.visualize()