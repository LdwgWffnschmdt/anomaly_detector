import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

from featureExtractorBase import FeatureExtractorBase

class FeatureExtractorResNet50V2(FeatureExtractorBase):
    """Feature extractor based on ResNet50V2 (trained on ImageNet).
    Generates 7x7x2048 feature vectors per image
    """

    def __init__(self):
        FeatureExtractorBase.__init__(self)

        # More info on image size: https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py#L128
        # TODO: Maybe increase image size to also increase spatial output resolution
        self.IMG_SIZE = 224 # All images will be resized to 224x224

        # Create the base model from the pre-trained model ResNet50V2
        self.model = tf.keras.applications.ResNet50(input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
                                                    include_top=False,
                                                    weights="imagenet")
        self.model.trainable = False
    
    def format_image(self, image):
        """Resize the images to a fixed input size, and
        rescale the input channels to a range of [-1, 1].
        (According to https://www.tensorflow.org/tutorials/images/transfer_learning)
        """
        image = tf.cast(image, tf.float32)
        #       \/ does the same #  image = (image / 127.5) - 1
        image = preprocess_input(image) # https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L152
        image = tf.image.resize_images(image, (self.IMG_SIZE, self.IMG_SIZE))
        return image

    def extract_batch(self, batch):
        return self.model(batch)

# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorResNet50V2()
    # extractor.plot_model(extractor.model)
    extractor.extract_files("/home/ldwg/data/CCW/2020-02-06-17-11-37.tfrecord")