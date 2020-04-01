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
    
    def classify(self, patch, threshold=None):
        """The anomaly measure is defined as the Mahalanobis distance between a feature sample
        and the single variate Gaussian distribution along each dimension.
        """
        if patch.cell_size.flat[0] != self.CELL_SIZE:
            patch.base.calculate_rasterization(self.CELL_SIZE)

        model = self.get_model(patch.bins)
        if model is None:
            logger.warning("No model available for this bin (%i, %i)" % (patch.bins.v, patch.bins.u))
            return 0 # Unknown
        
        return model.classify(patch)
    
    def __mahalanobis_distance__(self, patch):
        """Calculate the Mahalanobis distance between the input and the model"""
        if patch.cell_size.flat[0] != self.CELL_SIZE:
            patch.base.calculate_rasterization(self.CELL_SIZE)

        model = self.get_model(patch.bins)
        if model is None:
            logger.warning("No model available for this bin (%i, %i)" % (patch.bins.v, patch.bins.u))
            return -1 # TODO: What should we do?
        
        return model.__mahalanobis_distance__(patch)

    def __generate_model__(self, patches):
        # Ensure locations are calculated
        patches.ensure_locations()
        
        shape = patches.calculate_rasterization(self.CELL_SIZE)

        # Empty grid that will contain the model for each bin
        self.models = np.empty(shape=shape, dtype=object)
        models_created = 0

        with tqdm(desc="Generating models", total=np.prod(shape), file=sys.stderr) as pbar:
            for bin in np.ndindex(shape):
                bin_features_flat = patches.bin(bin, self.CELL_SIZE)
                
                if len(bin_features_flat) > 0:
                    # Create a new model
                    model = self.CREATE_ANOMALY_MODEL_FUNC()    # Instantiate a new model
                    model.__generate_model__(bin_features_flat) # The model only gets flattened features
                    self.models[bin] = model                    # Store the model
                    models_created += 1
                    pbar.set_postfix({"Models": models_created})
                pbar.update()
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
        h5file.attrs["Num models"] = sum(x is not None for x in self.models.flatten())
        h5file.attrs["Models shape"] = self.models.shape
        
        for v, u in tqdm(np.ndindex(self.models.shape), desc="Saving models", total=np.prod(self.models.shape), file=sys.stderr):
            model = self.get_model((v, u))
            if model is not None:
                g = h5file.create_group("%i/%i" % (v, u))
                g.attrs["v"] = v
                g.attrs["u"] = u
                model.__save_model_to_file__(g)
        return True
    	
        
# Only for tests
if __name__ == "__main__":
    from anomalyModelSVG import AnomalyModelSVG
    import consts

    patches = PatchArray(consts.FEATURES_FILE)

    model = AnomalyModelSpatialBinsBase(AnomalyModelSVG, cell_size=1.0)
    if model.load_or_generate(patches):
        patches.show_spatial_histogram(model.CELL_SIZE)

        model.calculate_mahalobis_distances()
        model.show_mahalanobis_distribution()

        model.visualize()