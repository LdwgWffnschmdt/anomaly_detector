from featureExtractorEfficientNetBase import FeatureExtractorEfficientNetBase

class FeatureExtractorEfficientNetB0(FeatureExtractorEfficientNetBase):
    """Feature extractor based on EfficientNetB0 (trained on ImageNet).
    Generates 7x7x1280 feature vectors per image
    """
    __handle__     = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"
    __output_key__ = "default"
    IMG_SIZE   = 224
    BATCH_SIZE = 128

# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorEfficientNetB0()
    extractor.plot_model(extractor.model)
    extractor.extract_files()