import os
import time

import tensorflow as tf
import matplotlib.pyplot as plt

import feature_extractor.utils as utils

class FeatureExtractorBase(object):
    
    def __init__(self):
        self.name = ""              # Should be set by the implementing class
        self.IMG_SIZE = 260
    
    def extract(self, image):     # Should be implemented by child class
        pass  

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

    def extract_files(self, filenames, output_file = "", batch_size = 32):
        if not isinstance(filenames, list):
            filenames = [filenames]

        # Check parameters
        if not filenames or len(filenames) < 1 or filenames[0] == "":
            print("Please specify at least one filename (%s)" % filenames)
            return False
        
        if output_file == "":
            output_file = os.path.join(os.path.abspath(os.path.dirname(filenames[0])), "Features" + self.name + ".h5")
            print("Output file set to %s" % output_file)
        
        print ("Loading dataset")
        raw_dataset = tf.data.TFRecordDataset(filenames)

        # Get number of examples in dataset
        total = sum(1 for record in raw_dataset)

        # Create a dictionary describing the features.
        feature_description = {
            "image/location/translation/x": tf.io.FixedLenFeature([], tf.float32),
            "image/location/translation/y": tf.io.FixedLenFeature([], tf.float32),
            "image/location/translation/z": tf.io.FixedLenFeature([], tf.float32),
            "image/location/rotation/x": tf.io.FixedLenFeature([], tf.float32),
            "image/location/rotation/y": tf.io.FixedLenFeature([], tf.float32),
            "image/location/rotation/z": tf.io.FixedLenFeature([], tf.float32),
            "image/height": tf.io.FixedLenFeature([], tf.int64),
            "image/width": tf.io.FixedLenFeature([], tf.int64),
            "image/channels": tf.io.FixedLenFeature([], tf.int64),
            "image/colorspace": tf.io.FixedLenFeature([], tf.string),
            "image/format": tf.io.FixedLenFeature([], tf.string),
            "image/encoded": tf.io.FixedLenFeature([], tf.string),
        }

        # Get locations as list
        location_dataset = []

        def _parse_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            example = tf.io.parse_single_example(example_proto, feature_description)
            image = tf.image.decode_jpeg(example["image/encoded"])
            image = self.format_image(image)

            location = {
                "translation/x": example["image/location/translation/x"],
                "translation/y": example["image/location/translation/y"],
                "translation/z": example["image/location/translation/z"],
                "rotation/x":    example["image/location/rotation/x"],
                "rotation/y":    example["image/location/rotation/y"],
                "rotation/z":    example["image/location/rotation/z"]
            }
            location_dataset.append(str(location))

            return image

        print ("Parsing dataset")

        parsed_dataset = raw_dataset.map(_parse_function)
        
        ### Extract features
        utils.print_progress(0, 1, prefix = "Extracting features:")
        
        # Get batches (seems to be better performance wise than extracting individual images)
        batches = parsed_dataset.batch(batch_size)

        feature_dataset = []

        start = time.time()

        with utils.GracefulInterruptHandler() as h:
            for batch in batches:
                if h.interrupted:
                    print "Interrupted!"
                    return
                
                # Extract features
                feature_batch = self.extract_batch(batch)

                # Add features to list
                for feature_vector in feature_batch:
                    feature_dataset.append(feature_vector.numpy())
                
                # Print progress
                utils.print_progress(len(feature_dataset),
                                     total,
                                     prefix = "Extracting features:",
                                     suffix = "(%i / %i)" % (len(feature_dataset), total),
                                     time_start = start)

        # Write locations and features to disk as HDF5 file
        utils.write_hdf5(output_file, location_dataset, feature_dataset)
        print("Successfully written locations and features to: %s" % output_file)

        return True
