""" Abstract base class for all feature extractors """
import os
import logging
import time

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import h5py

import feature_extractor.utils as utils

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
    
    def extract_files(self, filenames, output_format="h5", output_file="", batch_size=32, compression="lzf", compression_opts=None):
        """Loads a set of files, extracts the features and saves them to file
        Args:
            filenames (str / str[]): TFRecord file(s) extracted by rosbag_to_tfrecord
            output_format (str): Either "tfrecord" or "h5" (Default: "h5")
            output_file (str): Filename and path of the output file
            batch_size (str): Size of image batches fed to the extractor (Defaults: 32)
            compression (str): Output file compression, set to None for no compression (Default: "lzf"), gzip can be extremely slow combined with HDF5
            compression_opts (str): Compression level, set to None for no compression (Default: None)

        Returns:
            success (bool)
        """
        
        # Get all tfrecords in a folder if the file ends with *.tfrecord
        if isinstance(filenames, basestring) and filenames.endswith("*.tfrecord"):
            path = filenames.replace("*.tfrecord", "")
            filenames = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(".tfrecord")]
            if len(filenames) < 1:
                raise ValueError("There is no *.tfrecord file in %s." % path)

            if output_file == "":
                output_dir = os.path.join(os.path.abspath(os.path.dirname(filenames[0])), "Features")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_file = os.path.join(output_dir, self.NAME + "." + output_format)


        if not isinstance(filenames, list):
            filenames = [filenames]

        # Check parameters
        if not filenames or len(filenames) < 1 or filenames[0] == "":
            raise ValueError("Please specify at least one filename (%s)" % filenames)
        
        if output_format != "tfrecord" and output_format != "h5":
            raise ValueError("Supported output formats are 'tfrecord' and 'h5'")
            
        if output_file == "":
            output_file = os.path.join(os.path.abspath(os.path.dirname(filenames[0])), os.path.basename(filenames[0]).split(".")[0] + "." + self.NAME + "." + output_format)
            logging.info("Output file set to %s" % output_file)
            
        logging.info("Loading dataset")
        parsed_dataset = utils.load_dataset(filenames)

        # Format images to match network input
        def _format_image(image, example):
            return self.format_image(image), example
        parsed_dataset = parsed_dataset.map(_format_image)
        
        # Get number of examples in dataset
        total = sum(1 for record in parsed_dataset)

        with utils.GracefulInterruptHandler() as h:
            ### Extract features
            utils.print_progress(0, 1, prefix = "Extracting features:")
            
            # Get batches (seems to be better performance wise than extracting individual images)
            batches = parsed_dataset.batch(batch_size)

            # IO stuff
            if output_format == "tfrecord":
                tfOptions = tf.io.TFRecordOptions(compression_type=compression.upper(), compression_level=compression_opts)
                tfWriter = tf.io.TFRecordWriter(output_file, options=tfOptions)
            elif output_format == "h5":
                h5Writer = h5py.File(output_file, "w")
                metadata_dataset = h5Writer.create_dataset("metadata",
                                                           shape=(total,),
                                                           dtype=h5py.string_dtype(),
                                                           compression=compression,
                                                           compression_opts=compression_opts)
                feature_dataset  = None
        
            # For the progress bar
            counter = 0
            start = time.time()

            for batch in batches:
                if h.interrupted:
                    logging.warning("Interrupted!")
                    return False
                
                # Extract features
                feature_batch = self.extract_batch(batch[0])

                # Add features to list
                for index, feature_vector in enumerate(feature_batch):
                    counter += 1
                    if output_format == "tfrecord":
                        # Loop through the patches
                        for x, fv1 in enumerate(tf.unstack(feature_vector, axis=0)):
                            for y, fv in enumerate(tf.unstack(fv1, axis=0)):
                                # Add metadata and feature to TFRecord
                                feature_dict = {
                                    'metadata/location/translation/x'   : utils._float_feature(batch[1]["metadata/location/translation/x"][index].numpy()),
                                    'metadata/location/translation/y'   : utils._float_feature(batch[1]["metadata/location/translation/y"][index].numpy()),
                                    'metadata/location/translation/z'   : utils._float_feature(batch[1]["metadata/location/translation/z"][index].numpy()),
                                    'metadata/location/rotation/x'      : utils._float_feature(batch[1]["metadata/location/rotation/x"][index].numpy()),
                                    'metadata/location/rotation/y'      : utils._float_feature(batch[1]["metadata/location/rotation/y"][index].numpy()),
                                    'metadata/location/rotation/z'      : utils._float_feature(batch[1]["metadata/location/rotation/z"][index].numpy()),
                                    'metadata/time'                     : utils._int64_feature(batch[1]["metadata/time"][index].numpy()),
                                    'metadata/label'                    : utils._int64_feature(batch[1]["metadata/label"][index].numpy()),
                                    'metadata/rosbag'                   : utils._bytes_feature(batch[1]["metadata/rosbag"][index].numpy()),
                                    'metadata/tfrecord'                 : utils._bytes_feature(batch[1]["metadata/tfrecord"][index].numpy()),
                                    'metadata/feature_extractor'        : utils._bytes_feature(self.NAME),
                                    'metadata/patch/x'                  : utils._int64_feature(x),
                                    'metadata/patch/y'                  : utils._int64_feature(y),
                                    'feature'                           : utils._float_feature(list(fv.numpy()))
                                }
                                example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                                tfWriter.write(example.SerializeToString())
                    elif output_format == "h5":
                        if feature_dataset is None:
                            feature_dataset = h5Writer.create_dataset("features",
                                                                      shape=(total,
                                                                             feature_vector.shape[0],
                                                                             feature_vector.shape[1],
                                                                             feature_vector.shape[2]),
                                                                      dtype=np.float32,
                                                                      compression=compression,
                                                                      compression_opts=compression_opts)
                        feature_dataset[counter - 1] = feature_vector.numpy()
                        metadata_dataset[counter - 1] = str({
                            "location/translation/x": batch[1]["metadata/location/translation/x"][index].numpy(),
                            "location/translation/y": batch[1]["metadata/location/translation/y"][index].numpy(),
                            "location/translation/z": batch[1]["metadata/location/translation/z"][index].numpy(),
                            "location/rotation/x"   : batch[1]["metadata/location/rotation/x"][index].numpy(),
                            "location/rotation/y"   : batch[1]["metadata/location/rotation/y"][index].numpy(),
                            "location/rotation/z"   : batch[1]["metadata/location/rotation/z"][index].numpy(),
                            "time"                  : batch[1]["metadata/time"][index].numpy(),
                            "label"                 : batch[1]["metadata/label"][index].numpy(),
                            "rosbag"                : batch[1]["metadata/rosbag"][index].numpy(),
                            "tfrecord"              : batch[1]["metadata/tfrecord"][index].numpy(),
                            "feature_extractor"     : self.NAME
                        })
                
                # Print progress
                utils.print_progress(counter,
                                     total,
                                     prefix = "Extracting features:",
                                     suffix = "(%i / %i)" % (counter, total),
                                     time_start = start)

        if output_format == "tfrecord":
            tfWriter.close()
        elif output_format == "h5":
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

    def plot_model(self, model, dpi=96, to_file=None):
        """ Plot a model to an image file """
        if to_file is None:
            to_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Models", "%s.png" % self.NAME)
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