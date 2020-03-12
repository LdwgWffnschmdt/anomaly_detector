import tensorflow as tf

from featureExtractorMobileNetV2 import FeatureExtractorMobileNetV2

class FeatureExtractorMobileNetV2_Block6(FeatureExtractorMobileNetV2):
    """Feature extractor based on MobileNetV2 (trained on ImageNet).
    Output layer: block_6_project_BN
    Generates 14x14x64 feature vectors per image
    """
    IMG_SIZE   = 224
    BATCH_SIZE = 64

    def __init__(self):
        FeatureExtractorMobileNetV2.__init__(self)
        self.model = tf.keras.Model(self.model.inputs, self.model.get_layer("block_6_project_BN").output)   
        self.model.trainable = False

# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorMobileNetV2_Block6()
    extractor.plot_model(extractor.model)
    extractor.extract_files()