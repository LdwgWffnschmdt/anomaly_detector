# -*- coding: utf-8 -*-

from anomaly_model import AnomalyModelBalancedDistribution
import feature_extractor.utils as utils
from common.imageLocationUtility import ImageLocationUtility

class AnomalyModelBalancedDistributionPos(AnomalyModelBalancedDistribution):
    """Anomaly model formed by a Balanced Distribution of feature vectors
    Contains the location of each patch as two additional feature vector dimensions
    Reference: https://www.mdpi.com/2076-3417/9/4/757
    """
    
    def generate_model(self, features):
        # Add location to features
        features.calculate_locations()
        features.add_location_as_feature_dimension()
        AnomalyModelBalancedDistribution.generate_model(self, features)

# Only for tests
if __name__ == "__main__":
    from anomalyModelTest import AnomalyModelTest
    test = AnomalyModelTest(AnomalyModelBalancedDistributionPos(), add_locations_to_features=True)

    test.calculateMahalobisDistances()

    # def _pause(feature):
    #     return feature in test.model.normal_distribution

    # test.visualize(pause_func=_pause)