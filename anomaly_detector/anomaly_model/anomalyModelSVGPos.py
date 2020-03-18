# -*- coding: utf-8 -*-

from anomaly_model import AnomalyModelSVG
from common import utils, logger

class AnomalyModelSVGPos(AnomalyModelSVG):
    """Anomaly model formed by a Single Variate Gaussian (SVG) with model parameters Θ_SVG = (μ,σ²)
    Contains the location of each patch as two additional feature vector dimensions
    Reference: https://www.mdpi.com/1424-8220/16/11/1904/htm
    """
    
    def __generate_model__(self, features):
        # Add location to features
        features.add_location_as_feature_dimension()
        AnomalyModelSVG.__generate_model__(self, features)

# Only for tests
if __name__ == "__main__":
    model = AnomalyModelSVGPos()
    if model.load_or_generate(load_features=True):
        # model.calculate_mahalobis_distances()
        # model.show_mahalanobis_distribution()
        model.visualize(threshold=200)