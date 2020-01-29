import os
import logging
import time

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py

import feature_extractor.utils as utils

class FeatureExtractorBase(object):
    
    def __init__(self):
        self.NAME = ""              # Should be set by the implementing class
        self.IMG_SIZE = 260
    
    def extract_batch(self, batch): # Should be implemented by child class
        """Extract the features of batch of images"""
        pass  
    
    def format_image(self, image):  # Can be overridden by child class
        """Format an image to be compliant with extractor (NN) input"""
        image = tf.cast(image, tf.float32)
        image = image/255.0
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
    
    def load_dataset(self, filenames):
        """Loads a set of TFRecord files
        Args:
            filenames (str / str[]): TFRecord file(s) extracted by rosbag_to_tfrecord

        Returns:
            tf.data.MapDataset
        """
        if not filenames or len(filenames) < 1 or filenames[0] == "":
            raise ValueError("Please specify at least one filename (%s)" % filenames)
        
        raw_dataset = tf.data.TFRecordDataset(filenames)

        # Create a dictionary describing the features.
        feature_description = {
            "metadata/location/translation/x"   : tf.io.FixedLenFeature([], tf.float32),
            "metadata/location/translation/y"   : tf.io.FixedLenFeature([], tf.float32),
            "metadata/location/translation/z"   : tf.io.FixedLenFeature([], tf.float32),
            "metadata/location/rotation/x"      : tf.io.FixedLenFeature([], tf.float32),
            "metadata/location/rotation/y"      : tf.io.FixedLenFeature([], tf.float32),
            "metadata/location/rotation/z"      : tf.io.FixedLenFeature([], tf.float32),
            "metadata/time"                     : tf.io.FixedLenFeature([], tf.float32),
            "metadata/label"                    : tf.io.FixedLenFeature([], tf.int64),   # 0: Unknown, 1: No anomaly, 2: Contains an anomaly
            "metadata/rosbag"                   : tf.io.FixedLenFeature([], tf.string),
            "metadata/tfrecord"                 : tf.io.FixedLenFeature([], tf.string),
            "image/height"      : tf.io.FixedLenFeature([], tf.int64),
            "image/width"       : tf.io.FixedLenFeature([], tf.int64),
            "image/channels"    : tf.io.FixedLenFeature([], tf.int64),
            "image/colorspace"  : tf.io.FixedLenFeature([], tf.string),
            "image/format"      : tf.io.FixedLenFeature([], tf.string),
            "image/encoded"     : tf.io.FixedLenFeature([], tf.string),
        }
        
        def _parse_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            example = tf.io.parse_single_example(example_proto, feature_description)
            image = tf.image.decode_jpeg(example["image/encoded"])
            return self.format_image(image), example

        return raw_dataset.map(_parse_function)

    def extract_files(self, filenames, output_format="tfrecord", output_file="", batch_size=32, compression="gzip", compression_opts=4):
        """Loads a set of files, extracts the features and saves them to file
        Args:
            filenames (str / str[]): TFRecord file(s) extracted by rosbag_to_tfrecord
            output_format (str): Either "tfrecord" or "h5" (Default: "tfrecord")
            output_file (str): Filename and path of the output file
            batch_size (str): Size of image batches fed to the extractor (Defaults: 32)
            compression (str): Output file compression, set to None for no compression (Default: "gzip"), can be extremely slow combined with HDF5
            compression_opts (str): Compression level, set to None for no compression (Default: 4)

        Returns:
            success (bool)
        """
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
        parsed_dataset = self.load_dataset(filenames)
        
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
                        # Add image and position to TFRecord
                        feature_dict = {
                            'metadata/location/translation/x'   : utils._float_feature(batch[1]["metadata/location/translation/x"][index].numpy()),
                            'metadata/location/translation/y'   : utils._float_feature(batch[1]["metadata/location/translation/y"][index].numpy()),
                            'metadata/location/translation/z'   : utils._float_feature(batch[1]["metadata/location/translation/z"][index].numpy()),
                            'metadata/location/rotation/x'      : utils._float_feature(batch[1]["metadata/location/rotation/x"][index].numpy()),
                            'metadata/location/rotation/y'      : utils._float_feature(batch[1]["metadata/location/rotation/y"][index].numpy()),
                            'metadata/location/rotation/z'      : utils._float_feature(batch[1]["metadata/location/rotation/z"][index].numpy()),
                            'metadata/time'                     : utils._float_feature(batch[1]["metadata/time"][index].numpy()),
                            'metadata/label'                    : utils._int64_feature(batch[1]["metadata/label"][index].numpy()),
                            'metadata/rosbag'                   : utils._bytes_feature(batch[1]["metadata/rosbag"][index].numpy()),
                            'metadata/tfrecord'                 : utils._bytes_feature(batch[1]["metadata/tfrecord"][index].numpy()),
                            'metadata/feature_extractor'        : utils._bytes_feature(self.NAME),
                            'feature/flat'                      : utils._float_feature(list(feature_vector.numpy().reshape(-1))),
                            'feature/shape'                     : utils._int64_feature(list(feature_vector.shape))
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
