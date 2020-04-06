""" Abstract base class for all feature extractors """
import os
from common import utils, logger
import sys
import time
import traceback

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import h5py
from tqdm import tqdm

import consts

class FeatureExtractorBase(object):
    NAME       = property(lambda self: self.__class__.__name__.replace("FeatureExtractor", ""))
    
    # OVERRIDE THESE WITH THE RESPECTIVE IMPLEMENTATION
    IMG_SIZE   = 224
    BATCH_SIZE = consts.DEFAULT_BATCH_SIZE # Change this per network so it best utilizes resources
    TEMPORAL_BATCH_SIZE = 1

    def extract_batch(self, batch): # Should be implemented by child class
        """Extract the features of batch of images"""
        pass  
    
    def format_image(self, image):  # Can be overridden by child class
        """Format an image to be compliant with extractor (NN) input"""
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)   # Converts to float and scales to [0,1]
        image = tf.image.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
        return image

    def __transform_dataset__(self, dataset, total):
        """Do any transformations necessary (eg. temporal windowing for 3D networks)"""
        return dataset, total # Nothing by default

    ########################
    # Common functionality #
    ########################

    def extract(self, image):
        """Extract the features of a single image"""
        # A single image is of shape (w, h, 3), but the network wants (None, w, h, 3) as input
        batch = tf.expand_dims(image, 0) # Expand dimension so single image is a "batch" of one image
        return tf.squeeze(self.extract_batch(batch)) # Remove unnecessary output dimension
    
    def extract_frame_array(self, patches, **kwargs):
        """Loads a set of files, extracts the features and saves them to file
        Args:
            patches (PatchArray): PatchArray
            For **kwargs see extract_dataset

        Returns:
            success (bool)
        """

        dataset = patches.to_dataset()
        total = patches.shape[0]

        return self.extract_dataset(dataset, total, **kwargs)

    def extract_files(self, files=consts.EXTRACT_FILES, **kwargs):
        """Loads a set of files, extracts the features and saves them to file
        Args:
            files (str / str[]): TFRecord file(s) extracted by rosbag_to_tfrecord
            For **kwargs see extract_dataset

        Returns:
            success (bool)
        """
        dataset, total = utils.load_dataset(files)
        return self.extract_dataset(dataset, total, **kwargs)
    
    def extract_dataset(self, dataset, total, output_file="", batch_size=None, compression=None, compression_opts=None, **kwargs):
        """Loads a set of files, extracts the features and saves them to file
        Args:
            dataset (tf.data.Dataset): Dataset containing the input data
            totla (int): Number of items in Dataset
            output_file (str): Filename and path of the output file
            batch_size (str): Size of image batches fed to the extractor. Set to 0 for no batching. (Default: self.BATCH_SIZE)
            compression (str): Output file compression, set to None for no compression (Default: None), lzf is feasable, gzip can be extremely slow combined with HDF5
            compression_opts (str): Compression level, set to None for no compression (Default: None)
            **kwargs: Additional arguments will be saved to the output file as h5 attributes

        Returns:
            success (bool)
        """

        if output_file == "":
            output_dir = consts.FEATURES_PATH
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, self.NAME + ".h5")
            logger.info("Output file set to %s" % output_file)
        
        if batch_size is None:
            batch_size = self.BATCH_SIZE

        # Preprocess images
        dataset = dataset.map(lambda image, time: (self.format_image(image), time),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Call internal transformations (eg. temporal windowing for 3D networks)
        # dataset, total = self.__transform_dataset__(dataset, total)

        # Get batches (seems to be better performance wise than extracting individual images)
        if batch_size > 0:
            dataset = dataset.batch(batch_size)

        # IO stuff
        hf = h5py.File(output_file, "w")
    
        try:
            # Add metadata to the output file
            hf.attrs["Extractor"]           = self.NAME
            hf.attrs["Batch size"]          = batch_size
            hf.attrs["Compression"]         = str(compression)
            hf.attrs["Compression options"] = str(compression_opts)
            hf.attrs["Temporal batch size"] = self.TEMPORAL_BATCH_SIZE

            for key, value in kwargs.items():
                if value is not None:
                    hf.attrs[key] = value
            
            computer_info = utils.getComputerInfo()
            for key, value in computer_info.items():
                hf.attrs[key] = value
            
            start = time.time()
            counter = 0

            hf.attrs["Start"] = start
            
            # Create arrays to store output
            feature_dataset = None # We don't know the feature shape yet
            time_dataset    = hf.create_dataset("times",
                                                shape=(total,),
                                                dtype=np.uint64,
                                                compression=compression,
                                                compression_opts=compression_opts)
            
            # Loop over the dataset
            with tqdm(desc="Extracting features (batch size: %i)" % batch_size, total=total, file=sys.stderr) as pbar:
                for batch in dataset:
                    # Extract features
                    feature_batch = self.extract_batch(batch[0]) # This is where the magic happens

                    current_batch_size = len(feature_batch)

                    if feature_dataset is None:
                        # Create the array to store the features now
                        feature_dataset = hf.create_dataset("features",
                                                            shape=(total,) + tuple(feature_batch[0].shape),
                                                            chunks=(1,) + tuple(feature_batch[0].shape),
                                                            dtype=np.float32,
                                                            compression=compression,
                                                            compression_opts=compression_opts)

                    # Save the features and their metadata to the arrays
                    feature_dataset[counter : counter + current_batch_size] = feature_batch.numpy()
                    time_dataset[counter : counter + current_batch_size]    = batch[1].numpy()

                    # Count and update progress bar
                    counter += current_batch_size
                    pbar.update(n=current_batch_size)

            ## Variant where we first store everything in RAM
            # feature_dataset = None # We don't know the feature shape yet
            # time_dataset    = np.empty((total,),   dtype=np.uint64)
            
            # # Loop over the dataset
            # with tqdm(desc="Extracting features (batch size: %i)" % batch_size, total=total, file=sys.stderr) as pbar:
            #     for batch in dataset:
            #         # Extract features
            #         feature_batch = self.extract_batch(batch[0]) # This is where the magic happens

            #         current_batch_size = len(feature_batch)

            #         if feature_dataset is None:
            #             # Create the array to store the features now
            #             feature_dataset = np.empty((total,) + feature_batch[0].shape, dtype=np.float32)

            #         # Save the features and their metadata to the arrays
            #         feature_dataset[counter : counter + current_batch_size] = feature_batch.numpy()
            #         time_dataset[counter : counter + current_batch_size]    = batch[1].numpy()

            #         # Count and update progress bar
            #         counter += current_batch_size
            #         pbar.update(n=current_batch_size)

            # # Save arrays to file
            # hf.create_dataset("features",
            #                   data=feature_dataset,
            #                   dtype=np.float32,
            #                   compression=compression,
            #                   compression_opts=compression_opts)
            # hf.create_dataset("times",
            #                   data=time_dataset,
            #                   chunks=(1,) + tuple(feature_batch[0].shape),
            #                   dtype=np.uint64,
            #                   compression=compression,
            #                   compression_opts=compression_opts)
        except:
            exc = traceback.format_exc()
            logger.error(exc)
            hf.attrs["Exception"] = exc
            return False
        finally:
            end = time.time()
            hf.attrs["End"] = end
            hf.attrs["Duration"] = end - start
            hf.attrs["Duration (formatted)"] = utils.format_duration(end - start)
            hf.attrs["Number of frames extracted"] = counter
            hf.attrs["Number of total frames"] = total
            hf.close()

        return True

    ########################
    #      Utilities       #
    ########################

    def load_model(self, handle, signature="image_feature_vector", output_key="default"):
        """Load a pretrained model from TensorFlow Hub

        Args:
            handle: a callable object (subject to the conventions above), or a Python string for which hub.load() returns such a callable. A string is required to save the Keras config of this Layer.
        """
        inputs = tf.keras.Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 3))
        layer = hub.KerasLayer(handle,
                               trainable=False,
                               signature=signature,
                               output_key=output_key)(inputs)

        return tf.keras.Model(inputs=inputs, outputs=layer)

    def print_outputs(self, handle, signature="image_feature_vector"):
        """ Print possible outputs and their shapes """
        image = tf.cast(np.random.rand(1, self.IMG_SIZE, self.IMG_SIZE, 3), tf.float32)
        model = hub.load(handle).signatures[signature]
        out = model(image)
        logger.info("Outputs for model at %s with signature %s" % (handle, signature))
        for s in map(lambda y: "%-40s | %s" % (y, str(out[y].shape)), sorted(list(out), key=lambda x:out[x].shape[1])):
            logger.info(s)

    def plot_model(self, model, dpi=300, to_file=None):
        """ Plot a model to an image file """
        # Set the default file location and name
        if to_file is None:
            to_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Models", "%s.png" % self.NAME)

        # Make sure the output directory exists
        output_dir = os.path.dirname(to_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info("Creating plot of model %s: %s" % (self.NAME, to_file))
        
        tf.keras.utils.plot_model(
            model,
            to_file=to_file,
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",   # "TB" creates a vertical plot; "LR" creates a horizontal plot
            expand_nested=True,
            dpi=dpi
        )