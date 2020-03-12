from feature_extractor import FeatureExtractorC3D

class FeatureExtractorC3D_Block3(FeatureExtractorC3D):
    """Feature extractor based on C3D (trained on sports1M).
    Output layer: conv3b + MaxPooling3D to reduce frames
    Generates 28x28x256 feature vectors per temporal image batch
    """
    __layer__ = "conv3b"

# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorC3D_Block3()
    extractor.plot_model(extractor.model)
    extractor.extract_files()