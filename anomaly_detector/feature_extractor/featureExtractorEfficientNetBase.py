from featureExtractorBase import FeatureExtractorBase

class FeatureExtractorEfficientNetBase(FeatureExtractorBase):
    """Base class for feature extractors based on EfficientNet"""
    __handle__     = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"
    __output_key__ = "default"
    IMG_SIZE   = 528
    BATCH_SIZE = 8      # 16 might also fit

    def __init__(self):
        # Create the base model from the pre-trained model
        self.model = self.load_model(self.__handle__, output_key=self.__output_key__)
    
    def extract_batch(self, batch):
        return self.model(batch)