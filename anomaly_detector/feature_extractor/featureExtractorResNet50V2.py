from featureExtractorBase import FeatureExtractorBase

import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input

class FeatureExtractorResNet50V2_Block5(FeatureExtractorBase):
    """Feature extractor based on ResNet50V2 (trained on ImageNet)."""
    IMG_SIZE        = 224
    BATCH_SIZE      = 32
    LAYER_NAME      = "post_relu"
    OUTPUT_SHAPE    = (7, 7, 2048)
    RECEPTIVE_FIELD = {'stride': (32.0, 32.0), 'size': (479, 479)}

    def __init__(self):
        # Create the base model from the pre-trained model ResNet50V2
        model_full = tf.keras.applications.ResNet50V2(input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
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

class FeatureExtractorResNet50V2_Block4(FeatureExtractorResNet50V2_Block5):
    OUTPUT_SHAPE    = (7, 7, 1024)
    IMG_SIZE        = 224
    BATCH_SIZE      = 32
    LAYER_NAME      = "conv4_block6_out"
    RECEPTIVE_FIELD = {"stride": (32.0, 32.0), 'size': (287, 287)}

class FeatureExtractorResNet50V2_Block3(FeatureExtractorResNet50V2_Block5):
    OUTPUT_SHAPE    = (14, 14, 512)
    IMG_SIZE        = 224
    BATCH_SIZE      = 32
    LAYER_NAME      = "conv3_block4_out"
    RECEPTIVE_FIELD = {"stride": (16.0, 16.0), "size":  (95, 95)}

# More info on image size: https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py#L128
class FeatureExtractorResNet50V2_LargeImage_Block5(FeatureExtractorResNet50V2_Block5):
    OUTPUT_SHAPE    = (15, 15, 2048)
    IMG_SIZE        = 449
    BATCH_SIZE      = 4

class FeatureExtractorResNet50V2_LargeImage_Block4(FeatureExtractorResNet50V2_LargeImage_Block5):
    OUTPUT_SHAPE    = (15, 15, 1024)
    IMG_SIZE        = 449
    BATCH_SIZE      = 4

class FeatureExtractorResNet50V2_LargeImage_Block3(FeatureExtractorResNet50V2_LargeImage_Block5):
    OUTPUT_SHAPE    = (29, 29, 512)
    IMG_SIZE        = 449
    BATCH_SIZE      = 8

# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorResNet50V2_Block5()
    extractor.plot_model(extractor.model)
    extractor.extract_files()