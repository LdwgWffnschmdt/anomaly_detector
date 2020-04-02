# -*- coding: utf-8 -*-

from anomaly_model import AnomalyModelBalancedDistribution
from common import utils, ImageLocationUtility

class AnomalyModelBalancedDistributionPos(AnomalyModelBalancedDistribution):
    """Anomaly model formed by a Balanced Distribution of feature vectors
    Contains the location of each patch as two additional feature vector dimensions
    Reference: https://www.mdpi.com/2076-3417/9/4/757
    """
    
    def __generate_model__(self, patches, silent=False):
        # Add location to features
        AnomalyModelBalancedDistribution.__generate_model__(self, patches.add_location_as_feature_dimension(), silent=silent)

# Only for tests
if __name__ == "__main__":
    model = AnomalyModelBalancedDistributionPos()
    if model.load_or_generate(load_patches=True):
        model.visualize(threshold=200)