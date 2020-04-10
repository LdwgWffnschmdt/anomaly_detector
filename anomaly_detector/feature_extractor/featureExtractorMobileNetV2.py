import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from featureExtractorBase import FeatureExtractorBase

class FeatureExtractorMobileNetV2(FeatureExtractorBase):
    """Feature extractor based on MobileNetV2 (trained on ImageNet)."""
    IMG_SIZE        = 224
    BATCH_SIZE      = 64
    LAYER_NAME      = "out_relu"
    OUTPUT_SHAPE    = (7, 7, 1280)
    RECEPTIVE_FIELD = {'stride': (32.0, 32.0), 'size': (491, 491)}

    def __init__(self):
        # Create the base model from the pre-trained model MobileNetV2
        model_full = tf.keras.applications.MobileNetV2(input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
                                                       include_top=False,
                                                       weights="imagenet")
        model_full.trainable = False
        self.model = tf.keras.Model(model_full.inputs, model_full.get_layer(self.LAYER_NAME).output)   
        self.model.trainable = False
    
    def format_image(self, image):
        """Resize the images to a fixed input size, and
        rescale the input channels to a range of [-1, 1].
        (According to https://www.tensorflow.org/tutorials/images/transfer_learning)
        """
        image = tf.cast(image, tf.float32)
        #       \/ does the same #  image = (image / 127.5) - 1
        image = preprocess_input(image) # https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L152
        image = tf.image.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
        return image

    def extract_batch(self, batch):
        return self.model(batch)

class FeatureExtractorMobileNetV2_Block16(FeatureExtractorMobileNetV2):
    """Feature extractor based on MobileNetV2 (trained on ImageNet)."""
    IMG_SIZE        = 224
    BATCH_SIZE      = 64
    LAYER_NAME      = "block_16_project_BN"
    OUTPUT_SHAPE    = (7, 7, 320)
    RECEPTIVE_FIELD = {'stride': (32.0, 32.0), 'size': (491, 491)}

class FeatureExtractorMobileNetV2_Block14(FeatureExtractorMobileNetV2):
    """Feature extractor based on MobileNetV2 (trained on ImageNet)."""
    IMG_SIZE        = 224
    BATCH_SIZE      = 64
    LAYER_NAME      = "block_14_add"
    OUTPUT_SHAPE    = (7, 7, 160)
    RECEPTIVE_FIELD = {'stride': (32.0, 32.0), 'size': (363, 363)}

class FeatureExtractorMobileNetV2_Block12(FeatureExtractorMobileNetV2):
    """Feature extractor based on MobileNetV2 (trained on ImageNet)."""
    IMG_SIZE        = 224
    BATCH_SIZE      = 64
    LAYER_NAME      = "block_12_add"
    OUTPUT_SHAPE    = (14, 14, 96)
    RECEPTIVE_FIELD = {'stride': (16.0, 16.0), 'size': (267, 267)}

class FeatureExtractorMobileNetV2_Block9(FeatureExtractorMobileNetV2):
    """Feature extractor based on MobileNetV2 (trained on ImageNet)."""
    IMG_SIZE        = 224
    BATCH_SIZE      = 64
    LAYER_NAME      = "block_9_add"
    OUTPUT_SHAPE    = (14, 14, 64)
    RECEPTIVE_FIELD = {'stride': (16.0, 16.0), 'size': (171, 171)}

class FeatureExtractorMobileNetV2_Block6(FeatureExtractorMobileNetV2):
    """Feature extractor based on MobileNetV2 (trained on ImageNet)."""
    IMG_SIZE        = 224
    BATCH_SIZE      = 64
    LAYER_NAME      = "block_6_project_BN"
    OUTPUT_SHAPE    = (14, 14, 64)
    RECEPTIVE_FIELD = {'stride': (16.0, 16.0), 'size': (75, 75)}

class FeatureExtractorMobileNetV2_Block3(FeatureExtractorMobileNetV2):
    """Feature extractor based on MobileNetV2 (trained on ImageNet)."""
    IMG_SIZE        = 224
    BATCH_SIZE      = 64
    LAYER_NAME      = "block_3_project_BN"
    OUTPUT_SHAPE    = (28, 28, 32)
    RECEPTIVE_FIELD = {'stride': (8.0, 8.0),   'size': (27, 27)}


# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorMobileNetV2()
    extractor.plot_model(extractor.model)
    extractor.extract_files()