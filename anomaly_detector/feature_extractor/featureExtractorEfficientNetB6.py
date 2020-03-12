from featureExtractorEfficientNetBase import FeatureExtractorEfficientNetBase

class FeatureExtractorEfficientNetB6(FeatureExtractorEfficientNetBase):
    """Feature extractor based on EfficientNetB6 (trained on ImageNet).
    Generates 1x1x2304 feature vectors per image
    """
    __handle__     = "https://tfhub.dev/google/efficientnet/b6/feature-vector/1"
    __output_key__ = "default"
    IMG_SIZE   = 528
    BATCH_SIZE = 8      # 16 might also fit

# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorEfficientNetB6()
    extractor.plot_model(extractor.model, 600)
    extractor.extract_files()