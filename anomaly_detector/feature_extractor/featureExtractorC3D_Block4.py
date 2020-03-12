from feature_extractor import FeatureExtractorC3D

class FeatureExtractorC3D_Block4(FeatureExtractorC3D):
    """Feature extractor based on C3D (trained on sports1M).
    Output layer: conv4b + MaxPooling3D to reduce frames
    Generates 14x14x512 feature vectors per temporal image batch
    """
    __layer__ = "conv4b"

# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorC3D_Block4()
    extractor.plot_model(extractor.model)
    extractor.extract_files()