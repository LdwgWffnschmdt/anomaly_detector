from featureExtractorVGG16 import FeatureExtractorVGG16

class FeatureExtractorVGG16_Block4(FeatureExtractorVGG16):
    """Feature extractor based on VGG16 at Bock 4 without the last max pooling layer (trained on ImageNet).
    Generates 28x28x512 feature vectors per image
    """
    __layer__  = "block4_conv3"
    IMG_SIZE   = 224
    BATCH_SIZE = 32

# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorVGG16_Block4()
    extractor.plot_model(extractor.model)
    extractor.extract_files()