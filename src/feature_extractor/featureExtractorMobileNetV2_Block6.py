import os

import tensorflow as tf

from featureExtractorBase import FeatureExtractorBase
import feature_extractor.utils as utils

class FeatureExtractorMobileNetV2_Block6(FeatureExtractorBase):
    """Feature extractor based on MobileNetV2 (trained on ImageNet).
    Output layer: block_6_project_BN
    Generates 14x14x64 feature vectors per image
    """

    def __init__(self):
        FeatureExtractorBase.__init__(self)

        self.IMG_SIZE  = 224 # All images will be resized to 224x224
        self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)

        # Create the base model from the pre-trained model MobileNet V2
        model_full = tf.keras.applications.MobileNetV2(input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
                                                       include_top=False,
                                                       weights="imagenet")
        model_full.trainable = False

        self.model = tf.keras.Model(model_full.inputs, model_full.get_layer("block_6_project_BN").output)   
        self.model.trainable = False
    
    def format_image(self, image):
        """Resize the images to a fixed input size, and rescale the input channels to a range of [-1,1]"""
        image = tf.cast(image, tf.float32)
        image = (image/127.5) - 1
        image = tf.image.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
        return image

    def extract_batch(self, batch):
        return self.model(batch)

# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorMobileNetV2_Block6()
    
    # tf.keras.utils.plot_model(
    #     extractor.model,
    #     to_file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "%s.png" % extractor.NAME),
    #     show_shapes=True,
    #     show_layer_names=True,
    #     rankdir="TB",   # "TB" creates a vertical plot; "LR" creates a horizontal plot
    #     expand_nested=True,
    #     dpi=96
    # )

    extractor.extract_files("/home/ludwig/ros/src/ROS-kate_bag/bags/real/TFRecord/*.tfrecord")