import os
import time

import tensorflow as tf

from featureExtractorBase import FeatureExtractorBase
import feature_extractor.utils as utils

class FeatureExtractorMobileNetV2(FeatureExtractorBase):
    
    def __init__(self):
        FeatureExtractorBase.__init__(self)
        self.name = "MobileNetV2"

        self.IMG_SIZE = 160 # All images will be resized to 160x160
        self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)

        # Create the base model from the pre-trained model MobileNet V2
        self.model = tf.keras.applications.MobileNetV2(input_shape=self.IMG_SHAPE,
                                                       include_top=False,
                                                       weights="imagenet")
    
    def format_image(self, image):  # Resize the images to a fixed input size, and rescale the input channels to a range of [-1,1]
        image = tf.cast(image, tf.float32)
        image = (image/127.5) - 1
        image = tf.image.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
        return image

    def extract(self, image):
        example = tf.reshape(image, (1, self.IMG_SIZE, self.IMG_SIZE, 3))
        return self.model(example)
        
    def extract_batch(self, batch):
        return self.model(batch)