# -*- coding: utf-8 -*-

import os
import time
import traceback
import sys

import h5py
import numpy as np
from tqdm import tqdm

from shapely.strtree import STRtree
from shapely.geometry import Polygon, box, Point
from shapely.prepared import prep

from anomalyModelBase import AnomalyModelBase
from common import utils, logger, PatchArray
import consts

class AnomalyModelSpatialBinsBase(AnomalyModelBase):
    """ Base for anomaly models that create one model per spatial bin (grid cell) """
    def __init__(self, create_anomaly_model_func, patches, cell_size=0.2, fake=False):
        """ Create a new spatial bin anomaly model

        Args:
            create_anomaly_model_func (func): Method that returns an anomaly model
            patches (PatchArray): The patches are needed to get the rasterization
            cell_size (float): Width and height of spatial bin in meter
        """
        AnomalyModelBase.__init__(self)
        self.CELL_SIZE = cell_size
        self.KEY = "%.2f" % self.CELL_SIZE
        if fake: self.KEY = "fake_" + self.KEY
        self.CREATE_ANOMALY_MODEL_FUNC = create_anomaly_model_func
        self.FAKE = fake
        
        # Get extent
        x_min, y_min, x_max, y_max = patches.get_extent(cell_size, fake=fake)

        # Create the bins
        bins_y = np.arange(y_min, y_max, cell_size)
        bins_x = np.arange(x_min, x_max, cell_size)
        
        shape = (len(bins_y), len(bins_x))

        # Create the grid
        raster = np.zeros(shape, dtype=object)
        
        for v, y in enumerate(bins_y):
            for u, x in enumerate(bins_x):
                b = box(x, y, x + cell_size, y + cell_size)#Point(x + cell_size / 2, y + cell_size / 2)# 
                b.u = u
                b.v = v
                b.patches = list()
                raster[v, u] = b
            
        # Create a search tree of spatial boxes
        self._grid = STRtree(raster.ravel().tolist())
        
        m = create_anomaly_model_func()
        self.NAME = "SpatialBin/%s/%s" % (m.__class__.__name__.replace("AnomalyModel", ""), self.KEY)
    
    def classify(self, patch, threshold=None):
        """The anomaly measure is defined as the Mahalanobis distance"""
        model = self.models.flat[patch["bins_" + self.KEY]]
        if model is None:
            # logger.warning("No model available for this bin (%i, %i)" % (patch.bins.v, patch.bins.u))
            return 0 # Unknown
        
        return model.classify(patch)
    
    def __mahalanobis_distance__(self, patch):
        """Calculate the Mahalanobis distance between the input and the models
        in each bin that intersects the receptive field """
        
        model = self.models.flat[patch["bins_" + self.KEY]]
        if np.all(model == None):
            # logger.warning("No model available for this bin (%i, %i)" % (patch.bins.v, patch.bins.u))
            return np.nan # TODO: What should we do?
        
        # Use the mean of Mahalanobis distances to each model
        return np.mean([m.__mahalanobis_distance__(patch) for m in model if m is not None])

    def __mahalanobis_distance_single__(self, patch):
        """Calculate the Mahalanobis distance between the input and the model in the closest bin"""
        
        poly = Polygon([(patch.locations.tl.x, patch.locations.tl.y),
                        (patch.locations.tr.x, patch.locations.tr.y),
                        (patch.locations.br.x, patch.locations.br.y),
                        (patch.locations.bl.x, patch.locations.bl.y)])
        
        bin = self._grid.query(poly.centroid)

        # # pr = prep(poly)
        bin = filter(poly.centroid.intersects, bin)
        
        if len(bin) == 0:
            bin = [self._grid.nearest(poly.centroid)]

        model = self.models[bin[0].v, bin[0].u]
        if model == None:
            # logger.warning("No model available for this bin (%i, %i)" % (patch.bins.v, patch.bins.u))
            return np.nan # TODO: What should we do?
        
        return model.__mahalanobis_distance__(patch)
    
    def filter_training(self, patches):
        return patches

    def __generate_model__(self, patches, silent=False):
        # Ensure locations are calculated
        assert patches.contains_features, "Can only compute patch locations if there are patches"
        assert patches.contains_locations, "Can only compute patch locations if there are locations calculated"
        
        # Check if cell size rasterization is already calculated
        if not self.KEY in patches.contains_bins.keys() or not patches.contains_bins[self.KEY]:
            patches.calculate_rasterization(self.CELL_SIZE, self.FAKE)
        
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
    	
    def calculate_mahalanobis_distances(self):
        """ Calculate all the Mahalanobis distances and save them to the file """
        with h5py.File(self.patches.filename, "r+") as hf:
            ### Calculate Mahalanobis distances based on a single bin
            g = hf.get(self.NAME)

            if g is None:
                raise ValueError("The model needs to be saved first")
            
            maha = np.zeros(self.patches.shape, dtype=np.float64)
            
            for i in tqdm(np.ndindex(self.patches.shape), desc="Calculating mahalanobis distances (mean)", total=self.patches.size, file=sys.stderr):
                maha[i] = self.__mahalanobis_distance__(self.patches[i])

            no_anomaly = maha[self.patches.labels == 1]
            anomaly = maha[self.patches.labels == 2]

            if g.get("mahalanobis_distances") is not None: del g["mahalanobis_distances"]
            m = g.create_dataset("mahalanobis_distances", data=maha)
            m.attrs["max_no_anomaly"] = np.nanmax(no_anomaly) if no_anomaly.size > 0 else np.NaN
            m.attrs["max_anomaly"]    = np.nanmax(anomaly) if anomaly.size > 0 else np.NaN

            logger.info("Saved Mahalanobis distances to file")
            
            ### Calculate Mahalanobis distances based on a single bin
            maha = np.zeros(self.patches.shape, dtype=np.float64)
            
            for i in tqdm(np.ndindex(self.patches.shape), desc="Calculating mahalanobis distances (single)", total=self.patches.size, file=sys.stderr):
                maha[i] = self.__mahalanobis_distance_single__(self.patches[i])

            no_anomaly = maha[self.patches.labels == 1]
            anomaly = maha[self.patches.labels == 2]

            if g.get("Single/mahalanobis_distances") is not None: del g["Single/mahalanobis_distances"]
            m = g.create_dataset("Single/mahalanobis_distances", data=maha)
            m.attrs["max_no_anomaly"] = np.nanmax(no_anomaly) if no_anomaly.size > 0 else np.NaN
            m.attrs["max_anomaly"]    = np.nanmax(anomaly) if anomaly.size > 0 else np.NaN

            logger.info("Saved Mahalanobis distances to file")

            return True

        
# Only for tests
if __name__ == "__main__":
    from anomalyModelSVG import AnomalyModelSVG
    import consts

    patches = PatchArray(consts.FEATURES_FILE)

    model = AnomalyModelSpatialBinsBase(AnomalyModelSVG, patches, cell_size=0.2)
    
    if model.load_or_generate(patches):
        # patches.show_spatial_histogram(model.CELL_SIZE)

        def patch_to_color(patch):
            b = 0
            g = 0
            r = min(255, int(model.__mahalanobis_distance__(patch) * (255 / 50)))
            return (b, g, r)

        def patch_to_text(patch):
            return round(model.__mahalanobis_distance__(patch), 2)

        def click(frame, y, x):
            print "%i, %i" % (y, x)

        model.visualize(click=click, patch_to_color=patch_to_color, patch_to_text=patch_to_text)