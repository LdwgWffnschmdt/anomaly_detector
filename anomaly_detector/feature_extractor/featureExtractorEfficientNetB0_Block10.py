from featureExtractorEfficientNetBase import FeatureExtractorEfficientNetBase

class FeatureExtractorEfficientNetB0_Block10(FeatureExtractorEfficientNetBase):
    """Feature extractor based on EfficientNetB0 (trained on ImageNet).
    Output layer: block_10
    Generates 14x14x112 feature vectors per image
    """
    __handle__     = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"
    __output_key__ ="block_10"
    IMG_SIZE   = 224
    BATCH_SIZE = 128

# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorEfficientNetB0_Block10()
    extractor.plot_model(extractor.model)
    extractor.extract_files()