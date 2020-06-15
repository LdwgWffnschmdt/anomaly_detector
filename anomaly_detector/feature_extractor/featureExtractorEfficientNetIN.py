"""
Feature extractors using EfficientNet (https://arxiv.org/abs/1905.11946)
pre-trained using imagenet.
Different levels align with the levels stated in the paper.
"""

import tensorflow as tf
import efficientnet.tfkeras as efn
from efficientnet.tfkeras import preprocess_input

from featureExtractorBase import FeatureExtractorBase
from featureExtractorEfficientNet import __FeatureExtractorEfficientNetBase__

######
# B0 #
######

class FeatureExtractorEfficientNetINB0_Level9(__FeatureExtractorEfficientNetBase__):
    """Feature extractors based on EfficientNetB0 (trained on imagenet)."""
    IMG_SIZE        = 224
    BATCH_SIZE      = 32
    LAYER_NAME      = "top_conv"
    OUTPUT_SHAPE    = (7, 7, 1280)
    RECEPTIVE_FIELD = {'stride': (32.0, 32.0), 'size': (851, 851)}

    def __init__(self):
        # Create the base model from the pre-trained EfficientNet
        model_full = efn.EfficientNetB0(input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
                                        include_top=False,
                                        weights="imagenet")
        model_full.trainable = False

        self.model = tf.keras.Model(model_full.inputs, model_full.get_layer(self.LAYER_NAME).output)   
        self.model.trainable = False
    
# class FeatureExtractorEfficientNetINB0_Level4(FeatureExtractorEfficientNetINB0_Level9):
#     """Feature extractor based on EfficientNetB0 (trained on imagenet)."""
#     LAYER_NAME      = "block3b_add"
#     OUTPUT_SHAPE    = (28, 28, 40)
#     RECEPTIVE_FIELD = {'stride': (8.0, 8.0),   'size': (67, 67)}

# class FeatureExtractorEfficientNetINB0_Level5(FeatureExtractorEfficientNetINB0_Level9):
#     """Feature extractor based on EfficientNetB0 (trained on imagenet)."""
#     LAYER_NAME      = "block4c_add"
#     OUTPUT_SHAPE    = (14, 14, 80)
#     RECEPTIVE_FIELD = {'stride': (16.0, 16.0), 'size': (147, 147)}

class FeatureExtractorEfficientNetINB0_Level6(FeatureExtractorEfficientNetINB0_Level9):
    """Feature extractor based on EfficientNetB0 (trained on imagenet)."""
    LAYER_NAME      = "block5c_add"
    OUTPUT_SHAPE    = (14, 14, 112)
    RECEPTIVE_FIELD = {'stride': (16.0, 16.0), 'size': (339, 339)}

class FeatureExtractorEfficientNetINB0_Level7(FeatureExtractorEfficientNetINB0_Level9):
    """Feature extractor based on EfficientNetB0 (trained on imagenet)."""
    LAYER_NAME      = "block6d_add"
    OUTPUT_SHAPE    = (7, 7, 192)
    RECEPTIVE_FIELD = {'stride': (32.0, 32.0), 'size': (787, 787)}

class FeatureExtractorEfficientNetINB0_Level8(FeatureExtractorEfficientNetINB0_Level9):
    """Feature extractor based on EfficientNetB0 (trained on imagenet)."""
    LAYER_NAME      = "block7a_project_bn"
    OUTPUT_SHAPE    = (7, 7, 320)
    RECEPTIVE_FIELD = {'stride': (32.0, 32.0), 'size': (819, 819)}

######
# B3 #
######

class FeatureExtractorEfficientNetINB3_Level9(__FeatureExtractorEfficientNetBase__):
    """Feature extractors based on EfficientNetB3 (trained on imagenet)."""
    IMG_SIZE        = 300
    BATCH_SIZE      = 16
    LAYER_NAME      = "top_conv"
    OUTPUT_SHAPE    = (10, 10, 1536)
    RECEPTIVE_FIELD = {'stride': (32.0, 32.0), 'size': (1200, 1200)}
    
    def __init__(self):
        # Create the base model from the pre-trained EfficientNet
        model_full = efn.EfficientNetB3(input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
                                        include_top=False,
                                        weights="imagenet")
        model_full.trainable = False

        self.model = tf.keras.Model(model_full.inputs, model_full.get_layer(self.LAYER_NAME).output)   
        self.model.trainable = False

# class FeatureExtractorEfficientNetINB3_Level4(FeatureExtractorEfficientNetINB3_Level9):
#     """Feature extractor based on EfficientNetB3 (trained on imagenet)."""
#     LAYER_NAME      = "block3c_add"
#     OUTPUT_SHAPE    = (38, 38, 48)
#     RECEPTIVE_FIELD = {'stride': (8.0, 8.0),   'size': (111, 111)}

# class FeatureExtractorEfficientNetINB3_Level5(FeatureExtractorEfficientNetINB3_Level9):
#     """Feature extractor based on EfficientNetB3 (trained on imagenet)."""
#     LAYER_NAME      = "block4e_add"
#     OUTPUT_SHAPE    = (19, 19, 96)
#     RECEPTIVE_FIELD = {'stride': (16.0, 16.0), 'size': (255, 255)}

class FeatureExtractorEfficientNetINB3_Level6(FeatureExtractorEfficientNetINB3_Level9):
    """Feature extractor based on EfficientNetB3 (trained on imagenet)."""
    LAYER_NAME      = "block5e_add"
    OUTPUT_SHAPE    = (19, 19, 136)
    RECEPTIVE_FIELD = {'stride': (16.0, 16.0), 'size': (575, 575)}

class FeatureExtractorEfficientNetINB3_Level7(FeatureExtractorEfficientNetINB3_Level9):
    """Feature extractor based on EfficientNetB3 (trained on imagenet)."""
    LAYER_NAME      = "block6f_add"
    OUTPUT_SHAPE    = (10, 10, 232)
    RECEPTIVE_FIELD = {'stride': (32.0, 32.0), 'size': (1200, 1200)}

class FeatureExtractorEfficientNetINB3_Level8(FeatureExtractorEfficientNetINB3_Level9):
    """Feature extractor based on EfficientNetB3 (trained on imagenet)."""
    LAYER_NAME      = "block7b_add"
    OUTPUT_SHAPE    = (10, 10, 384)
    RECEPTIVE_FIELD = {'stride': (32.0, 32.0), 'size': (1200, 1200)}

######
# B6 #
######

class FeatureExtractorEfficientNetINB6_Level9(__FeatureExtractorEfficientNetBase__):
    """Feature extractors based on EfficientNetB6 (trained on imagenet)."""
    IMG_SIZE        = 528
    BATCH_SIZE      = 2
    LAYER_NAME      = "top_conv"
    OUTPUT_SHAPE    = (17, 17, 2304)
    RECEPTIVE_FIELD = {'stride': (32.0, 32.0), 'size': (1056, 1056)}
    
    def __init__(self):
        # Create the base model from the pre-trained EfficientNet
        model_full = efn.EfficientNetB6(input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
                                        include_top=False,
                                        weights="imagenet")
        model_full.trainable = False

        self.model = tf.keras.Model(model_full.inputs, model_full.get_layer(self.LAYER_NAME).output)   
        self.model.trainable = False

# class FeatureExtractorEfficientNetINB6_Level4(FeatureExtractorEfficientNetINB6_Level9):
#     """Feature extractor based on EfficientNetB6 (trained on imagenet)."""
#     LAYER_NAME      = "block3f_add"
#     OUTPUT_SHAPE    = (66, 66, 72)
#     RECEPTIVE_FIELD = {'stride': (8.0, 8.0),   'size': (235, 235)}

# class FeatureExtractorEfficientNetINB6_Level5(FeatureExtractorEfficientNetINB6_Level9):
#     """Feature extractor based on EfficientNetB6 (trained on imagenet)."""
#     LAYER_NAME      = "block4h_add"
#     OUTPUT_SHAPE    = (33, 33, 144)
#     RECEPTIVE_FIELD = {'stride': (16.0, 16.0), 'size': (475, 475)}

class FeatureExtractorEfficientNetINB6_Level6(FeatureExtractorEfficientNetINB6_Level9):
    """Feature extractor based on EfficientNetB6 (trained on imagenet)."""
    LAYER_NAME      = "block5h_add"
    OUTPUT_SHAPE    = (33, 33, 200)
    RECEPTIVE_FIELD = {'stride': (16.0, 16.0), 'size': (987, 987)}

class FeatureExtractorEfficientNetINB6_Level7(FeatureExtractorEfficientNetINB6_Level9):
    """Feature extractor based on EfficientNetB6 (trained on imagenet)."""
    LAYER_NAME      = "block6k_add"
    OUTPUT_SHAPE    = (17, 17, 344)
    RECEPTIVE_FIELD = {'stride': (32.0, 32.0), 'size': (1056, 1056)}

class FeatureExtractorEfficientNetINB6_Level8(FeatureExtractorEfficientNetINB6_Level9):
    """Feature extractor based on EfficientNetB6 (trained on imagenet)."""
    LAYER_NAME      = "block7c_add"
    OUTPUT_SHAPE    = (17, 17, 576)
    RECEPTIVE_FIELD = {'stride': (32.0, 32.0), 'size': (1056, 1056)}

# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorEfficientNetINB0_Level9()
    extractor.plot_model(extractor.model)
    extractor.extract_files()