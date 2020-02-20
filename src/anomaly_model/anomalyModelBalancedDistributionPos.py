# -*- coding: utf-8 -*-

from anomaly_model import AnomalyModelBalancedDistribution
import feature_extractor.utils as utils
from common.imageLocationUtility import ImageLocationUtility

class AnomalyModelBalancedDistributionPos(AnomalyModelBalancedDistribution):
    """Anomaly model formed by a Balanced Distribution of feature vectors
    Contains the location of each patch as two additional feature vector dimensions
    Reference: https://www.mdpi.com/2076-3417/9/4/757
    """
    def __init__(self, initial_normal_features=1000, threshold_learning=20, threshold_classification=5, pruning_parameter=0.5):
        AnomalyModelBalancedDistribution.__init__(self, initial_normal_features=initial_normal_features,
                                                        threshold_learning=threshold_learning,
                                                        threshold_classification=threshold_classification,
                                                        pruning_parameter=pruning_parameter)
        self.ilu = ImageLocationUtility()

    def generate_model(self, metadata, features):
        # Add location to features
        features = self.ilu.add_location_to_features(metadata, features)
        AnomalyModelBalancedDistribution.generate_model(self, metadata, features)

# Only for tests
if __name__ == "__main__":
    from anomalyModelTest import AnomalyModelTest
    test = AnomalyModelTest(AnomalyModelBalancedDistributionPos(), add_locations_to_features=True)

    test.calculateMahalobisDistances()

    # def _pause(feature):
    #     return feature in test.model.normal_distribution

    # test.visualize(pause_func=_pause)