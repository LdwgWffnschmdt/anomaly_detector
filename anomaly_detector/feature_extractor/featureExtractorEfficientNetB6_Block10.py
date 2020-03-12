from featureExtractorEfficientNetBase import FeatureExtractorEfficientNetBase

class FeatureExtractorEfficientNetB6_Block10(FeatureExtractorEfficientNetBase):
    """Feature extractor based on EfficientNetB6 (trained on ImageNet).
    Output layer: block_10
    Generates 66x66x72 feature vectors per image
    """
    __handle__     = "https://tfhub.dev/google/efficientnet/b6/feature-vector/1"
    __output_key__ ="block_10"
    IMG_SIZE   = 528
    BATCH_SIZE = 8

# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorEfficientNetB6_Block10()
    extractor.plot_model(extractor.model, 600)
    extractor.extract_files()