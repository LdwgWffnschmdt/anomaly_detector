# -*- coding: utf-8 -*-

import os
import time
import traceback
import sys

import h5py
import numpy as np
from tqdm import tqdm

from anomalyModelBase import AnomalyModelBase
from common import utils, logger, PatchArray
import consts

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
        self.KEY = "%.2f" % self.CELL_SIZE
        if consts.FAKE_RF: self.KEY = "fake_" + self.KEY
        self.CREATE_ANOMALY_MODEL_FUNC = create_anomaly_model_func
        
        m = create_anomaly_model_func()
        self.NAME = "SpatialBin/%s/%s" % (m.__class__.__name__.replace("AnomalyModel", ""), self.KEY)
    
    def classify(self, patch, threshold=None):
        """The anomaly measure is defined as the Mahalanobis distance between a feature sample
        and the single variate Gaussian distribution along each dimension.
        """
        model = self.models.flat[patch["bins_" + self.KEY]]
        if model is None:
            # logger.warning("No model available for this bin (%i, %i)" % (patch.bins.v, patch.bins.u))
            return 0 # Unknown
        
        return model.classify(patch)
    
    def __mahalanobis_distance__(self, patch):
        """Calculate the Mahalanobis distance between the input and the model"""
        
        model = self.models.flat[patch["bins_" + self.KEY]]
        if np.all(model == None):
            # logger.warning("No model available for this bin (%i, %i)" % (patch.bins.v, patch.bins.u))
            return np.nan # TODO: What should we do?
        
        return np.mean([m.__mahalanobis_distance__(patch) for m in model if m is not None])

    def filter_training(self, patches):
        return patches

    def __generate_model__(self, patches, silent=False):
        # Ensure locations are calculated
        assert patches.contains_features, "Can only compute patch locations if there are patches"
        assert patches.contains_locations, "Can only compute patch locations if there are locations calculated"
        
        # Check if cell size rasterization is already calculated
        if not self.KEY in patches.contains_bins.keys() or not patches.contains_bins[self.KEY]:
            patches.calculate_rasterization(self.CELL_SIZE)
        
        patches_flat = patches.ravel()
        
        raster = patches.rasterizations[self.KEY]

        # Empty grid that will contain the model for each bin
        self.models = np.empty(shape=raster.shape, dtype=object)
        models_created = 0

        with tqdm(desc="Generating models", total=self.models.size, file=sys.stderr) as pbar:
            for bin in np.ndindex(raster.shape):
                indices = raster[bin]

                if len(indices) > 0:
                    model_input = AnomalyModelBase.filter_training(self, patches_flat[indices])
                    if model_input.size > 0:
                        # Create a new model
                        model = self.CREATE_ANOMALY_MODEL_FUNC()    # Instantiate a new model
                        model.__generate_model__(model_input, silent=silent) # The model only gets flattened features
                        self.models[bin] = model                    # Store the model
                        models_created += 1
                        pbar.set_postfix({"Models": models_created})
                pbar.update()
        return True
    
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
                if "v" in g.attrs.keys() and "u" in g.attrs.keys():
                    v = g.attrs["v"]
                    u = g.attrs["u"]
                    self.models[v, u] = self.CREATE_ANOMALY_MODEL_FUNC()
                    self.models[v, u].__load_model_from_file__(g)
                    pbar.update()

            h5file.visititems(_add_model)
        
        if isinstance(self.patches, PatchArray):
            self.patches.calculate_rasterization(self.CELL_SIZE)

        return True
    
    def __save_model_to_file__(self, h5file):
        """Save the model to disk"""
        h5file.attrs["Cell size"] = self.CELL_SIZE
        h5file.attrs["Models shape"] = self.models.shape
        
        models_count = 0
        for v, u in tqdm(np.ndindex(self.models.shape), desc="Saving models", total=self.models.size, file=sys.stderr):
            model = self.models[v, u]
            if model is not None:
                g = h5file.create_group("%i/%i" % (v, u))
                g.attrs["v"] = v
                g.attrs["u"] = u
                model.__save_model_to_file__(g)
                models_count += 1
        h5file.attrs["Num models"] = models_count
        return True
    	
        
# Only for tests
if __name__ == "__main__":
    from anomalyModelSVG import AnomalyModelSVG
    import consts

    patches = PatchArray(consts.FEATURES_FILE)

    model = AnomalyModelSpatialBinsBase(AnomalyModelSVG, cell_size=2.0)
    
    if model.load_or_generate(patches):
        # patches.show_spatial_histogram(model.CELL_SIZE)
        model.visualize()