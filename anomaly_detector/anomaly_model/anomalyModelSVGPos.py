# -*- coding: utf-8 -*-

from anomaly_model import AnomalyModelSVG
from common import utils, logger

class AnomalyModelSVGPos(AnomalyModelSVG):
    """Anomaly model formed by a Single Variate Gaussian (SVG) with model parameters Θ_SVG = (μ,σ²)
    Contains the location of each patch as two additional feature vector dimensions
    Reference: https://www.mdpi.com/1424-8220/16/11/1904/htm
    """
    
    def __generate_model__(self, patches, silent=False):
        # Add location to features
        AnomalyModelSVG.__generate_model__(self, patches.add_location_as_feature_dimension(), silent=silent)

# Only for tests
if __name__ == "__main__":
    model = AnomalyModelSVGPos()
    if model.load_or_generate(load_patches=True):
        model.visualize(threshold=200)