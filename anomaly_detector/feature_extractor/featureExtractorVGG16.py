import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

from featureExtractorBase import FeatureExtractorBase

class FeatureExtractorVGG16_Block5(FeatureExtractorBase):
    """Feature extractor based on VGG16 at block 5 without the last max pooling layer (trained on ImageNet)."""
    IMG_SIZE        = 224
    BATCH_SIZE      = 32
    LAYER_NAME      = "block5_conv3"
    OUTPUT_SHAPE    = (14, 14, 512)
    RECEPTIVE_FIELD = {'stride': (16.0, 16.0), 'size': (181, 181)}

    def __init__(self):
        # Create the base model from the pre-trained model MobileNet V2
        model_full = tf.keras.applications.VGG16(input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
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

class FeatureExtractorVGG16_Block4(FeatureExtractorVGG16_Block5):
    """Feature extractor based on VGG16 at block 4 without the last max pooling layer (trained on ImageNet)."""
    IMG_SIZE   = 224
    BATCH_SIZE = 32
    LAYER_NAME  = "block4_conv3"
    OUTPUT_SHAPE = (28, 28, 512)
    RECEPTIVE_FIELD = {'stride': (8.0, 8.0),   'size': (85, 85)}

class FeatureExtractorVGG16_Block3(FeatureExtractorVGG16_Block5):
    """Feature extractor based on VGG16 at block 3 without the last max pooling layer (trained on ImageNet)."""
    IMG_SIZE   = 224
    BATCH_SIZE = 32
    LAYER_NAME  = "block3_conv3"
    OUTPUT_SHAPE = (56, 56, 512)
    RECEPTIVE_FIELD = {'stride': (4.0, 4.0),   'size': (37, 37)}

# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorVGG16_Block5()
    extractor.plot_model(extractor.model)
    extractor.extract_files()