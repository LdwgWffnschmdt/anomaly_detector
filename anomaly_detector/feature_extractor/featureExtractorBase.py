""" Abstract base class for all feature extractors """
import os
import logging
import time
import traceback
from glob import glob

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import h5py

import common.utils as utils

class FeatureExtractorBase(object):
    
    def __init__(self):
        self.NAME = self.__class__.__name__.replace("FeatureExtractor", "")
        self.IMG_SIZE = 260
    
    def extract_batch(self, batch): # Should be implemented by child class
        """Extract the features of batch of images"""
        pass  
    
    def format_image(self, image):  # Can be overridden by child class
        """Format an image to be compliant with extractor (NN) input"""
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)   # Converts to float and scales to [0,1]
        image = tf.image.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
        return image

    ########################
    # Common functionality #
    ########################

    def extract(self, image):
        """Extract the features of a single image"""
        # A single image is of shape (w, h, 3), but the network wants (None, w, h, 3) as input
        batch = tf.expand_dims(image, 0) # Expand dimension so single image is a "batch" of one image
        return tf.squeeze(self.extract_batch(batch)) # Remove unnecessary output dimension
    
    def extract_files(self, files, output_file="", batch_size=128, compression="lzf", compression_opts=None):
        """Loads a set of files, extracts the features and saves them to file
        Args:
            files (str / str[]): TFRecord file(s) extracted by rosbag_to_tfrecord
            output_file (str): Filename and path of the output file
            batch_size (str): Size of image batches fed to the extractor (Defaults: 32)
            compression (str): Output file compression, set to None for no compression (Default: "lzf"), gzip can be extremely slow combined with HDF5
            compression_opts (str): Compression level, set to None for no compression (Default: None)

        Returns:
            success (bool)
        """
        
        if isinstance(files, basestring):
            files = [files]
            
        # Expand wildcards
        files_expanded = []
        for s in files:
            files_expanded += glob(s)
        files = list(set(files_expanded)) # Remove duplicates

        # Check parameters
        if not files or len(files) < 1 or files[0] == "":
            raise ValueError("Please specify at least one filename (%s)" % files)
            
        if output_file == "":
            output_dir = os.path.join(os.path.abspath(os.path.dirname(files[0])), "Features")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, self.NAME + ".h5")
            logging.info("Output file set to %s" % output_file)

        logging.info("Loading dataset")
        parsed_dataset = utils.load_tfrecords(files, batch_size)

        # Format images to match network input
        def _format_image(image, example):
            location = [example["location/translation/x"],
                        example["location/translation/y"],
                        example["location/translation/z"],
                        example["location/rotation/x"],
                        example["location/rotation/y"],
                        example["location/rotation/z"]]
            time      = example["time"]
            label     = example["label"]
            rosbag    = example["rosbag"]
            tfrecord  = example["tfrecord"]
            return self.format_image(image), location, time, label, rosbag, tfrecord
        parsed_dataset = parsed_dataset.map(_format_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        # Get number of examples in dataset
        total = sum(1 for record in parsed_dataset)

        with utils.GracefulInterruptHandler() as h:
            ### Extract features
            utils.print_progress(0, 1, prefix = "Extracting features:")
            
            # Get batches (seems to be better performance wise than extracting individual images)
            batches = parsed_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

            # IO stuff
            h5Writer = h5py.File(output_file, "w")
        
            # Add metadata to the output file
            h5Writer.attrs["Extractor"]                 = self.NAME
            h5Writer.attrs["Files"]                     = files
            h5Writer.attrs["Batch size"]                = batch_size
            h5Writer.attrs["Compression"]               = compression
            if compression_opts is not None:
                h5Writer.attrs["Compression options"]   = compression_opts

            computer_info = utils.getComputerInfo()
            for key, value in computer_info.items():
                h5Writer.attrs[key] = value
            h5Writer.attrs["Start"] = time.time()

            # For the progress bar
            counter = 0
            start = time.time()
            
            dt_str = h5py.string_dtype(encoding='ascii')

            feature_dataset = None
            locations_dataset   = np.empty((total, 6), dtype=np.float32)
            time_dataset        = np.empty((total,),   dtype=np.uint64)
            label_dataset       = np.empty((total,),   dtype=np.int8)
            rosbag_dataset      = np.empty((total,),   dtype=object)
            tfrecord_dataset    = np.empty((total,),   dtype=object)
            
            try:
                for batch in batches:
                    if h.interrupted:
                        logging.warning("Interrupted!")
                        raise KeyboardInterrupt()
                    
                    # Extract features
                    feature_batch = self.extract_batch(batch[0])

                    current_batch_size = len(feature_batch)

                    # Add features to list
                    if feature_dataset is None:
                        feature_dataset = np.empty((total,) + feature_batch[0].shape)

                    feature_dataset[counter : counter + current_batch_size]  = feature_batch.numpy()

                    locations_dataset[counter : counter + current_batch_size] = batch[1].numpy()
                    time_dataset[counter : counter + current_batch_size]      = batch[2].numpy()
                    label_dataset[counter : counter + current_batch_size]     = batch[3].numpy()

                    n = batch[4].numpy()

                    rosbag_dataset[counter : counter + current_batch_size]    = batch[4].numpy()
                    tfrecord_dataset[counter : counter + current_batch_size]  = batch[5].numpy()

                    counter += current_batch_size

                    # Print progress
                    utils.print_progress(counter,
                                         total,
                                         prefix = "Extracting features:",
                                         suffix = "(%i / %i)" % (counter, total),
                                         time_start = start)

                h5Writer.create_dataset("features", data=feature_dataset, dtype=np.float32,
                                                    compression=compression,
                                                    compression_opts=compression_opts)
                if len(locations_dataset) > 0:  h5Writer.create_dataset("locations_dataset" , data=locations_dataset , dtype=np.float32, compression=compression, compression_opts=compression_opts)
                if len(time_dataset) > 0:       h5Writer.create_dataset("time_dataset"      , data=time_dataset      , dtype=np.uint64, compression=compression, compression_opts=compression_opts)
                if len(label_dataset) > 0:      h5Writer.create_dataset("label_dataset"     , data=label_dataset     , dtype=np.int8, compression=compression, compression_opts=compression_opts)
                if len(rosbag_dataset) > 0:     h5Writer.create_dataset("rosbag_dataset"    , data=rosbag_dataset    , dtype=dt_str)
                if len(tfrecord_dataset) > 0:   h5Writer.create_dataset("tfrecord_dataset"  , data=tfrecord_dataset  , dtype=dt_str)
            except:
                utils.print_progress(counter, total,
                                              prefix = "Cancelled:",
                                              suffix = "(%i / %i)" % (counter, total),
                                              time_start = start)
                exc = traceback.format_exc()
                logging.error(exc)
                h5Writer.attrs["Exception"] = exc
                return False
            finally:
                end = time.time()
                h5Writer.attrs["End"] = end
                h5Writer.attrs["Duration"] = end - start
                h5Writer.attrs["Duration (formatted)"] = utils.format_duration(end - start)
                h5Writer.attrs["Number of frames extracted"] = counter
                h5Writer.attrs["Number of total frames"] = total
                h5Writer.close()

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
        logging.info("Outputs for model at %s with signature %s" % (handle, signature))
        for s in map(lambda y: "%-40s | %s" % (y, str(out[y].shape)), sorted(list(out), key=lambda x:out[x].shape[1])):
            print(s)

    def plot_model(self, model, dpi=300, to_file=None):
        """ Plot a model to an image file """
        # Set the default file location and name
        if to_file is None:
            to_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Models", "%s.png" % self.NAME)

        # Make sure the output directory exists
        output_dir = os.path.dirname(to_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logging.info("Creating plot of model %s: %s" % (self.NAME, to_file))
        
        tf.keras.utils.plot_model(
            model,
            to_file=to_file,
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",   # "TB" creates a vertical plot; "LR" creates a horizontal plot
            expand_nested=True,
            dpi=dpi
        )