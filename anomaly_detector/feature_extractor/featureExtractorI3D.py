import tensorflow as tf
import tensorflow_hub as hub

from featureExtractorBase import FeatureExtractorBase

class FeatureExtractorI3D(FeatureExtractorBase):
    """Feature extractor based on I3D (trained on Kinetics-600).
    Generates 1x1x600 feature vectors per temporal image batch
    """
    IMG_SIZE            = 224
    BATCH_SIZE          = 0
    TEMPORAL_BATCH_SIZE = 16

    def __init__(self):

        # Create the base model from the pre-trained model
        inputs = tf.keras.Input(shape=(self.TEMPORAL_BATCH_SIZE, self.IMG_SIZE, self.IMG_SIZE, 3))
        layer = hub.KerasLayer("https://tfhub.dev/deepmind/i3d-kinetics-600/1", trainable=False)(inputs)

        self.model = tf.keras.Model(inputs=inputs, outputs=layer)
    
    def __transform_dataset__(self, dataset, total):
        total = total - self.TEMPORAL_BATCH_SIZE + 1

        temporal_image_windows = dataset.map(lambda image, *args: image).window(self.TEMPORAL_BATCH_SIZE, 1, 1, True)
        matching_meta_stuff    = dataset.map(lambda image, *args: args).skip(self.TEMPORAL_BATCH_SIZE - 1)
        return tf.data.Dataset.zip((temporal_image_windows, matching_meta_stuff)).map(lambda image, meta: (image,) + meta), total

    def extract_batch(self, batch):
        tensor = tf.constant(list(batch.as_numpy_iterator()))
        return self.model(tensor)

# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorI3D()
    extractor.plot_model(extractor.model)
    extractor.extract_files()