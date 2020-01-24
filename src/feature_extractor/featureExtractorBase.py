import os
import time

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import feature_extractor.utils as utils

class FeatureExtractorBase(object):
    
    def __init__(self):
        self.NAME = ""              # Should be set by the implementing class
        self.IMG_SIZE = 260
    
    def extract_batch(self, batch): # Should be implemented by child class
        pass  
    
    def format_image(self, image):  # Can be overridden by child class
        image = tf.cast(image, tf.float32)
        image = image/255.0
        image = tf.image.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
        return image

    ########################
    # Common functionality #
    ########################

    def extract(self, image):
        # A single image is of shape (w, h, 3), but the network wants (None, w, h, 3) as input
        batch = tf.expand_dims(image, 0) # Expand dimension so single image is a "batch" of one image
        return tf.squeeze(self.extract_batch(batch)) # Remove unnecessary output dimension
    
    def extract_files(self, filenames, output_file="", batch_size=32):
        if not isinstance(filenames, list):
            filenames = [filenames]

        # Check parameters
        if not filenames or len(filenames) < 1 or filenames[0] == "":
            print("Please specify at least one filename (%s)" % filenames)
            return False
        
        if output_file == "":
            output_file = os.path.join(os.path.abspath(os.path.dirname(filenames[0])), os.path.basename(filenames[0]).split(".")[0] + "." + self.NAME + ".h5")
            print("Output file set to %s" % output_file)
            
        print ("Loading dataset")
        raw_dataset = tf.data.TFRecordDataset(filenames)

        # Get number of examples in dataset
        total = sum(1 for record in raw_dataset)

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

        print ("Parsing dataset")
        parsed_dataset = raw_dataset.map(_parse_function)
        
        with utils.GracefulInterruptHandler() as h:
            
            ### Extract metadata
            utils.print_progress(0, 1, prefix = "Extracting metadata:")
            metadata_dataset = []
            start = time.time()
            for example in parsed_dataset:
                if h.interrupted:
                    print "\nInterrupted!"
                    return
                
                metadata_dataset.append(str({
                    "location/translation/x": example[1]["metadata/location/translation/x"].numpy(),
                    "location/translation/y": example[1]["metadata/location/translation/y"].numpy(),
                    "location/translation/z": example[1]["metadata/location/translation/z"].numpy(),
                    "location/rotation/x"   : example[1]["metadata/location/rotation/x"].numpy(),
                    "location/rotation/y"   : example[1]["metadata/location/rotation/y"].numpy(),
                    "location/rotation/z"   : example[1]["metadata/location/rotation/z"].numpy(),
                    "time"                  : example[1]["metadata/time"].numpy(),
                    "label"                 : example[1]["metadata/label"].numpy(),
                    "rosbag"                : example[1]["metadata/rosbag"].numpy(),
                    "tfrecord"              : example[1]["metadata/tfrecord"].numpy(),
                    "feature_extractor"     : self.NAME
                }))
                # Print progress
                utils.print_progress(len(metadata_dataset),
                                        total,
                                        prefix = "Extracting metadata:",
                                        suffix = "(%i / %i)" % (len(metadata_dataset), total),
                                        time_start = start)

            ### Extract features
            utils.print_progress(0, 1, prefix = "Extracting features:")
            
            # Get batches (seems to be better performance wise than extracting individual images)
            batches = parsed_dataset.batch(batch_size)

            feature_dataset = []
            
            start = time.time()

            for batch in batches:
                if h.interrupted:
                    print "\nInterrupted!"
                    return
                
                # Extract features
                feature_batch = self.extract_batch(batch[0])

                # Add features to list
                for feature_vector in feature_batch:
                    feature_dataset.append(feature_vector.numpy())
                
                # Print progress
                utils.print_progress(len(feature_dataset),
                                     total,
                                     prefix = "Extracting features:",
                                     suffix = "(%i / %i)" % (len(feature_dataset), total),
                                     time_start = start)

        # Write metadata and features to disk as HDF5 file
        utils.write_hdf5(output_file, metadata_dataset, feature_dataset)

        return True
