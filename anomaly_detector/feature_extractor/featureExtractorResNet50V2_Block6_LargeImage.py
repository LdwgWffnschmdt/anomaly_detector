import tensorflow as tf

from featureExtractorResNet50V2 import FeatureExtractorResNet50V2

class FeatureExtractorResNet50V2_Block6_LargeImage(FeatureExtractorResNet50V2):
    """Feature extractor based on ResNet50V2 (trained on ImageNet).
    Output layer: conv4_block6_out
    Generates 15x15x1024 feature vectors per image
    """
    # More info on image size: https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py#L128
    # TODO: Maybe increase image size to also increase spatial output resolution
    IMG_SIZE   = 449
    BATCH_SIZE = 8

    def __init__(self):
        FeatureExtractorResNet50V2.__init__(self)
        self.model = tf.keras.Model(self.model.inputs, self.model.get_layer("conv4_block6_out").output)   
        self.model.trainable = False

# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorResNet50V2_Block6_LargeImage()
    extractor.plot_model(extractor.model)
    extractor.extract_files()