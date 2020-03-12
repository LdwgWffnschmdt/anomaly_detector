from featureExtractorBase import FeatureExtractorBase

class FeatureExtractorEfficientNetB0_Block10(FeatureExtractorBase):
    """Feature extractor based on EfficientNetB0 (trained on ImageNet).
    Output layer: block_10
    Generates 14x14x112 feature vectors per image
    """

    def __init__(self):
        FeatureExtractorBase.__init__(self)

        self.IMG_SIZE = 224 # All images will be resized to 224x224

        # Create the base model from the pre-trained model
        self.model = self.load_model("https://tfhub.dev/google/efficientnet/b0/feature-vector/1",
                                     output_key="block_10")
                       
    def extract_batch(self, batch):
        return self.model(batch)

# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorEfficientNetB0_Block10()
    extractor.plot_model(extractor.model)
    extractor.extract_files()