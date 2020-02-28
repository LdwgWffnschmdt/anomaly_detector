# -*- coding: utf-8 -*-

from anomaly_model import AnomalyModelBalancedDistribution
import common.utils as utils
from common.imageLocationUtility import ImageLocationUtility

class AnomalyModelBalancedDistributionPos(AnomalyModelBalancedDistribution):
    """Anomaly model formed by a Balanced Distribution of feature vectors
    Contains the location of each patch as two additional feature vector dimensions
    Reference: https://www.mdpi.com/2076-3417/9/4/757
    """
    
    def __generate_model__(self, features):
        # Add location to features
        features.calculate_locations()
        features.add_location_as_feature_dimension()
        AnomalyModelBalancedDistribution.__generate_model__(self, features)

# Only for tests
if __name__ == "__main__":
    model = AnomalyModelBalancedDistributionPos()
    model.load_or_generate()

    model.calculate_mahalobis_distances()

    # def _pause(feature):
    #     return feature in test.model.normal_distribution

    # model.visualize(pause_func=_pause)