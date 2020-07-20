import tensorflow as tf
import numpy as np

from featureExtractorBase import FeatureExtractorBase

from Models.C3D.c3d import C3D
from Models.C3D.sports1M_utils import preprocess_input

class FeatureExtractorC3D_Block5(FeatureExtractorBase):
    """Feature extractor based on C3D (trained on sports1M).
    Output layer: conv5b + MaxPooling3D to reduce frames"""
    IMG_SIZE            = 112
    BATCH_SIZE          = 8
    LAYER_NAME          = "conv5b"
    OUTPUT_SHAPE        = (7, 7, 512)
    TEMPORAL_BATCH_SIZE = 16   # Fixed for C3D
    RECEPTIVE_FIELD     = {'stride': (16.0, 16.0), 'size': (119, 119)}

    def __init__(self):
        # Create the base model from the pre-trained C3D
        model_full = C3D(weights='sports1M')
        model_full.trainable = False

        output = model_full.get_layer(self.LAYER_NAME).output
        pool_size = (output.shape[1], 1, 1)
        output = tf.keras.layers.MaxPooling3D(pool_size=pool_size, strides=pool_size, padding='valid', name='reduce_frames')(output)

        self.model = tf.keras.Model(model_full.inputs, output)
        self.model.trainable = False
    
    def format_image(self, image):
        # image = tf.cast(image, tf.float32)
        image = preprocess_input(image)
        return image

    def __transform_dataset__(self, dataset, total):
        total = total - self.TEMPORAL_BATCH_SIZE + 1

        temporal_image_windows = dataset.map(lambda image, *args: image).window(self.TEMPORAL_BATCH_SIZE, 1, 1, True)
        temporal_image_windows = temporal_image_windows.flat_map(lambda window: window.batch(self.TEMPORAL_BATCH_SIZE))

        matching_meta_stuff    = dataset.map(lambda image, *args: args).skip(self.TEMPORAL_BATCH_SIZE - 1)
        return tf.data.Dataset.zip((temporal_image_windows, matching_meta_stuff)).map(lambda image, meta: (image,) + meta), total

    def extract_batch(self, batch):
        if batch.ndim == 4:
            batch = np.expand_dims(batch, axis=0)
        return tf.squeeze(self.model(batch))

class FeatureExtractorC3D_Block4(FeatureExtractorC3D_Block5):
    """Feature extractor based on C3D (trained on sports1M).
    Output layer: conv4b + MaxPooling3D to reduce frames"""
    BATCH_SIZE      = 32
    LAYER_NAME      = "conv4b"
    OUTPUT_SHAPE    = (14, 14, 512)
    RECEPTIVE_FIELD = {'stride': (8.0, 8.0),   'size': (55, 55)}

class FeatureExtractorC3D_Block3(FeatureExtractorC3D_Block5):
    """Feature extractor based on C3D (trained on sports1M).
    Output layer: conv3b + MaxPooling3D to reduce frames"""
    BATCH_SIZE      = 32
    LAYER_NAME      = "conv3b"
    OUTPUT_SHAPE    = (28, 28, 256)
    RECEPTIVE_FIELD = {'stride': (4.0, 4.0),   'size': (23, 23)}

# Only for tests
if __name__ == "__main__":
    from common import PatchArray
    extractor = FeatureExtractorC3D_Block5()
    # extractor.plot_model(extractor.model)
    patches = PatchArray()

    p = patches[:, 0, 0]

    f = np.zeros(p.shape, dtype=np.bool)
    f[:] = np.logical_and(p.directions == 1,                                   # CCW and
                            np.logical_or(p.labels == 2,                         #   Anomaly or
                                        np.logical_and(p.round_numbers >= 7,   #     Round between 2 and 5
                                                        p.round_numbers <= 9)))

    # Let's make contiguous blocks of at least 10, so
    # we can do some meaningful temporal smoothing afterwards
    for i, b in enumerate(f):
        if b and i - 10 >= 0:
            f[i - 10:i] = True

    patches = patches[f]

    extractor.extract_dataset(patches.to_temporal_dataset(), patches.shape[0])