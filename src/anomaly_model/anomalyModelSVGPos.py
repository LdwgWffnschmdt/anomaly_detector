# -*- coding: utf-8 -*-

from anomaly_model import AnomalyModelSVG
import feature_extractor.utils as utils
from common.imageLocationUtility import ImageLocationUtility

class AnomalyModelSVGPos(AnomalyModelSVG):
    """Anomaly model formed by a Single Variate Gaussian (SVG) with model parameters Θ_SVG = (μ,σ²)
    Contains the location of each patch as two additional feature vector dimensions
    Reference: https://www.mdpi.com/1424-8220/16/11/1904/htm
    """
    
    def generate_model(self, features):
        # Add location to features
        features.calculate_locations()
        features.add_location_as_feature_dimension()
        AnomalyModelSVG.generate_model(self, features)

# Only for tests
if __name__ == "__main__":
    from anomalyModelTest import AnomalyModelTest
    test = AnomalyModelTest(AnomalyModelSVGPos(), add_locations_to_features=True)

    test.calculateMahalobisDistances()