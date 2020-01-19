import os
import time

import tensorflow as tf

import feature_extractor.utils as utils

class FeatureExtractorBase(object):
    
    def __init__(self):
        self.name = ""          # Should be set by the implementing class
    
    def extract(self, example):
        print example
        pass  

    def extract_batch(self, filenames, output_dir = ""):
        if not isinstance(filenames, list):
            filenames = [filenames]

        # Check parameters
        if not filenames or len(filenames) < 1 or filenames[0] == "":
            print("Please specify at least one filename (%s)" % filenames)
            return False
        
        if output_dir == "" or not os.path.exists(output_dir) or not os.path.isdir(output_dir):
            output_dir = os.path.abspath(filenames[0]) + "-Features" + self.name
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print("Output directory set to %s" % output_dir)
        
        print ("Loading dataset")
        raw_dataset = tf.data.TFRecordDataset(filenames)

        # Create a dictionary describing the features.
        feature_description = {
            'image/location/translation/x': tf.io.FixedLenFeature([], tf.float32),
            'image/location/translation/y': tf.io.FixedLenFeature([], tf.float32),
            'image/location/translation/z': tf.io.FixedLenFeature([], tf.float32),
            'image/location/rotation/x': tf.io.FixedLenFeature([], tf.float32),
            'image/location/rotation/y': tf.io.FixedLenFeature([], tf.float32),
            'image/location/rotation/z': tf.io.FixedLenFeature([], tf.float32),
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/channels': tf.io.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.io.FixedLenFeature([], tf.string),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
        }

        def _parse_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, feature_description)

        print ("Parsing dataset")

        parsed_dataset = raw_dataset.map(_parse_function)

        ### Extract features
        utils.print_progress(0, 1, prefix = 'Extracting features:')

        total = sum(1 for record in raw_dataset)
        index = 0
        start = time.time()

        # Iterate through dataset and extract features
        for example in parsed_dataset:
            index += 1
            feature_vector = self.extract(example)
            # TODO: Save feature_vector somehow to output_dir (h5?)

            # Print progress
            utils.print_progress(index,
                                 total,
                                 prefix = 'Extracting features:',
                                 suffix = '(%i / %i)' % (index, total),
                                 time_start = start)
            
            return False
        
        return True