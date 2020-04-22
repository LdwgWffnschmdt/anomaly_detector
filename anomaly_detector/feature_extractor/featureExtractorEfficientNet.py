import tensorflow as tf
import efficientnet.tfkeras as efn
from efficientnet.tfkeras import preprocess_input

from featureExtractorBase import FeatureExtractorBase

######
# B0 #
######

class FeatureExtractorEfficientNetB0(FeatureExtractorBase):
    """Feature extractors based on EfficientNetB0 (trained on noisy-student)."""
    IMG_SIZE        = 224
    BATCH_SIZE      = 128
    LAYER_NAME      = "top_conv"
    OUTPUT_SHAPE    = (7, 7, 1280)
    RECEPTIVE_FIELD = {'stride': (32.0, 32.0), 'size': (851, 851)}

    def __init__(self):
        # Create the base model from the pre-trained EfficientNet
        model_full = efn.EfficientNetB0(input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
                                        include_top=False,
                                        weights="noisy-student")
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

class FeatureExtractorEfficientNetB0_Block3(FeatureExtractorEfficientNetB0):
    """Feature extractor based on EfficientNetB0 (trained on noisy-student)."""
    LAYER_NAME      = "block3b_add"
    OUTPUT_SHAPE    = (28, 28, 40)
    RECEPTIVE_FIELD = {'stride': (8.0, 8.0),   'size': (67, 67)}

class FeatureExtractorEfficientNetB0_Block4(FeatureExtractorEfficientNetB0):
    """Feature extractor based on EfficientNetB0 (trained on noisy-student)."""
    LAYER_NAME      = "block4c_add"
    OUTPUT_SHAPE    = (14, 14, 80)
    RECEPTIVE_FIELD = {'stride': (16.0, 16.0), 'size': (147, 147)}

class FeatureExtractorEfficientNetB0_Block5(FeatureExtractorEfficientNetB0):
    """Feature extractor based on EfficientNetB0 (trained on noisy-student)."""
    LAYER_NAME      = "block5c_add"
    OUTPUT_SHAPE    = (14, 14, 112)
    RECEPTIVE_FIELD = {'stride': (16.0, 16.0), 'size': (339, 339)}

class FeatureExtractorEfficientNetB0_Block6(FeatureExtractorEfficientNetB0):
    """Feature extractor based on EfficientNetB0 (trained on noisy-student)."""
    LAYER_NAME      = "block6d_add"
    OUTPUT_SHAPE    = (7, 7, 192)
    RECEPTIVE_FIELD = {'stride': (32.0, 32.0), 'size': (787, 787)}

######
# B6 #
######

class FeatureExtractorEfficientNetB6(FeatureExtractorEfficientNetB0):
    """Feature extractors based on EfficientNetB6 (trained on noisy-student)."""
    IMG_SIZE        = 528
    BATCH_SIZE      = 64
    LAYER_NAME      = "top_conv"
    OUTPUT_SHAPE    = (17, 17, 2304)
    RECEPTIVE_FIELD = {'stride': (32.0, 32.0), 'size': (1056, 1056)}

    def __init__(self):
        # Create the base model from the pre-trained EfficientNet
        model_full = efn.EfficientNetB6(input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
                                        include_top=False,
                                        weights="noisy-student")
        model_full.trainable = False

        self.model = tf.keras.Model(model_full.inputs, model_full.get_layer(self.LAYER_NAME).output)   
        self.model.trainable = False

class FeatureExtractorEfficientNetB6_Block3(FeatureExtractorEfficientNetB6):
    """Feature extractor based on EfficientNetB6 (trained on noisy-student)."""
    LAYER_NAME      = "block3f_add"
    OUTPUT_SHAPE    = (66, 66, 72)
    RECEPTIVE_FIELD = {'stride': (8.0, 8.0),   'size': (235, 235)}

class FeatureExtractorEfficientNetB6_Block4(FeatureExtractorEfficientNetB6):
    """Feature extractor based on EfficientNetB6 (trained on noisy-student)."""
    LAYER_NAME      = "block4h_add"
    OUTPUT_SHAPE    = (33, 33, 144)
    RECEPTIVE_FIELD = {'stride': (16.0, 16.0), 'size': (475, 475)}

class FeatureExtractorEfficientNetB6_Block5(FeatureExtractorEfficientNetB6):
    """Feature extractor based on EfficientNetB6 (trained on noisy-student)."""
    LAYER_NAME      = "block5h_add"
    OUTPUT_SHAPE    = (33, 33, 200)
    RECEPTIVE_FIELD = {'stride': (16.0, 16.0), 'size': (987, 987)}

class FeatureExtractorEfficientNetB6_Block6(FeatureExtractorEfficientNetB6):
    """Feature extractor based on EfficientNetB6 (trained on noisy-student)."""
    LAYER_NAME      = "block6k_add"
    OUTPUT_SHAPE    = (17, 17, 344)
    RECEPTIVE_FIELD = {'stride': (32.0, 32.0), 'size': (1056, 1056)}

# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorEfficientNetB0()
    extractor.plot_model(extractor.model)
    extractor.extract_files()