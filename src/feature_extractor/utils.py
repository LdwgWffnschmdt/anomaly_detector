# -*- coding: utf-8 -*-

import os
import logging
import sys
import time
import signal

import ast
import tensorflow as tf
import numpy as np
import h5py

import cv2

# Configure logging
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.INFO)

##############
# Meta + Feat#
##############

def getImageTransformationMatrix(w, h):
    """Calculate the matrix that will transform an input of
    given width and height to relative real world coordinates

    Args:
        w (int): Width of the input
        h (int): Height of the input

    Returns:
        A Matrix that will convert image coordinates to
        relative real world coordinates
    """
    # The transformation matrix is defined by the for corners of
    # the image and their real world coordinates relative to the camera
    src = np.float32([[ 0, 0], [w, 0], [ 0,   h], [w,   h]])
    dst = np.float32([[-3, 6], [3, 6], [-1, 0.7], [1, 0.7]])
    return cv2.getPerspectiveTransform(src, dst) # The transformation matrix

def getRelativeLocations(w, h):
    """Calculates an array of shape (w, h, 2) containing the
    respective relative real world coordinates

    Args:
        w (int): Width of the input
        h (int): Height of the input

    Returns:
        An array of shape (w, h, 2) containing the relative
        real world coordinates
    """
    P = getImageTransformationMatrix(w, h)
    
    relative_locations = np.zeros((w, h, 2))

    for x in range(w):
        for y in range(h):
            point = np.array([y + 0.5, x + 0.5, 1.]) # +0.5 so we take the center of each patch

            transformed_point = P.dot(point)
            transformed_point = transformed_point / transformed_point[2] # Normalize by third component
            relative_locations[x,y,0] = transformed_point[0]
            relative_locations[x,y,1] = transformed_point[1]
    
    return relative_locations

def getAbsoluteLocations(metadata, relative_locations):
    """Transform the relative locations to absolute locations

    Args:
        metadata (dict): A metadata dict containing information
                         about the camera location
        relative_locations (array): An array of shape (w, h, 2) with
                                    locations relative to the camera

    Returns:
        An array of shape (w, h, 2) containing absolute locations
    """

    camera_position = np.array([metadata["location/translation/x"],
                                metadata["location/translation/y"]])

    # Construct a 2D rotation matrix
    _s = np.sin(metadata["location/rotation/z"] - np.pi / 2.)
    _c = np.cos(metadata["location/rotation/z"] - np.pi / 2.)

    R = np.array([[_c, -_s],
                  [_s,  _c]])

    def _to_relative(p):
        return camera_position + R.dot(p)

    return np.apply_along_axis(_to_relative, 2, relative_locations)

def addLocationToFeatures(metadata, features):
    """Calculate the real world coordinates of every patch and add
    these to the feature vectors
    
    Args:
        metadata (list): Array of metadata for the features
        features (list): Array of features as extracted by a FeatureExtractor

    Returns:
        A new features list with the coordinates added as the
        last two feature dimensions
    """
    logging.info("Adding locations as feature dimensions")
    total, h, w, depth = features.shape
    
    relative_locations = getRelativeLocations(w, h)

    features_with_locations = np.zeros((total, h, w, depth + 2))

    for i, feature_3d in enumerate(features):
        meta = metadata[i]
        absolute_locations = getAbsoluteLocations(meta, relative_locations)
        features_with_locations[i] = np.concatenate((feature_3d, absolute_locations), axis=2)

    return features_with_locations

##############
# Images IO  #
##############

def load_dataset(filenames):
    """Loads a set of TFRecord files
    Args:
        filenames (str / str[]): TFRecord file(s) extracted
                                 by rosbag_to_tfrecord

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
        "metadata/time"                     : tf.io.FixedLenFeature([], tf.float32), # TODO: Change to int64 for production
        "metadata/label"                    : tf.io.FixedLenFeature([], tf.int64),   # 0: Unknown, 1: No anomaly, 2: Contains an anomaly
        "metadata/rosbag"                   : tf.io.FixedLenFeature([], tf.string),
        "metadata/tfrecord"                 : tf.io.FixedLenFeature([], tf.string),
        "image/height"      : tf.io.FixedLenFeature([], tf.int64),
        "image/width"       : tf.io.FixedLenFeature([], tf.int64),
        "image/channels"    : tf.io.FixedLenFeature([], tf.int64),
        "image/colorspace"  : tf.io.FixedLenFeature([], tf.string),
        "image/format"      : tf.io.FixedLenFeature([], tf.string),
        "image/encoded"     : tf.io.FixedLenFeature([], tf.string)
    }
    
    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        example = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.image.decode_jpeg(example["image/encoded"])
        return image, example

    return raw_dataset.map(_parse_function)


##############
# Feature IO #
##############

def read_features_file(filename):
    """Reads metadata and features from a HDF5 or TFRecords file.
    
    Args:
        filename (str): filename to read

    Returns:
        Tuple with metadata and features
    """
    logging.info("Reading metadata and features from: %s" % filename)
    fn, file_extension = os.path.splitext(filename)
    if file_extension == ".h5":
        with h5py.File(filename, "r") as hf:
            # Parse metadata object
            metadata = np.array([ast.literal_eval(m) for m in hf["metadata"]])
            return metadata, np.array(hf["features"])
    elif file_extension == ".tfrecord":
        raw_dataset = tf.data.TFRecordDataset(filename, compression_type="GZIP")
        
        # Create a dictionary describing the features.
        feature_description = {
            'metadata/location/translation/x'   : tf.io.FixedLenFeature([], tf.float32),
            'metadata/location/translation/y'   : tf.io.FixedLenFeature([], tf.float32),
            'metadata/location/translation/z'   : tf.io.FixedLenFeature([], tf.float32),
            'metadata/location/rotation/x'      : tf.io.FixedLenFeature([], tf.float32),
            'metadata/location/rotation/y'      : tf.io.FixedLenFeature([], tf.float32),
            'metadata/location/rotation/z'      : tf.io.FixedLenFeature([], tf.float32),
            'metadata/time'                     : tf.io.FixedLenFeature([], tf.float32), # TODO: Change to int64 for production
            'metadata/label'                    : tf.io.FixedLenFeature([], tf.int64),   # 0: Unknown, 1: No anomaly, 2: Contains an anomaly
            'metadata/rosbag'                   : tf.io.FixedLenFeature([], tf.string),
            'metadata/tfrecord'                 : tf.io.FixedLenFeature([], tf.string),
            'metadata/feature_extractor'        : tf.io.FixedLenFeature([], tf.string),
            'metadata/patch/x'                  : tf.io.FixedLenFeature([], tf.int64),
            'metadata/patch/y'                  : tf.io.FixedLenFeature([], tf.int64),
            'feature'                           : tf.io.FixedLenFeature([], tf.float32)
        }

        def _parse_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            example = tf.io.parse_single_example(example_proto, feature_description)
            metadata = {
                'location/translation/x'   : example['metadata/location/translation/x'],
                'location/translation/y'   : example['metadata/location/translation/y'],
                'location/translation/z'   : example['metadata/location/translation/z'],
                'location/rotation/x'      : example['metadata/location/rotation/x'   ],
                'location/rotation/y'      : example['metadata/location/rotation/y'   ],
                'location/rotation/z'      : example['metadata/location/rotation/z'   ],
                'time'                     : example['metadata/time'                  ],
                'label'                    : example['metadata/label'                 ],
                'rosbag'                   : example['metadata/rosbag'                ],
                'tfrecord'                 : example['metadata/tfrecord'              ],
                'feature_extractor'        : example['metadata/feature_extractor'     ],
                'patch/x'                  : example['metadata/patch/x'               ],
                'patch/y'                  : example['metadata/patch/y'               ]
            }

            feature = example['feature']
            return metadata, feature

        return raw_dataset.map(_parse_function)
    else:
        raise ValueError("Filename has to be *.h5 or *.tfrecord")

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# Can be used to store float64 values if necessary
# (http://jrmeyer.github.io/machinelearning/2019/05/29/tensorflow-dataset-estimator-api.html)
def _float64_feature(float64_value):
    float64_bytes = [str(float64_value).encode()]
    bytes_list = tf.train.BytesList(value=float64_bytes)
    bytes_list_feature = tf.train.Feature(bytes_list=bytes_list)
    return bytes_list_feature
    #    example['float_value'] = tf.strings.to_number(example['float_value'], out_type=tf.float64)


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#################
# Output helper #
#################

# Print iterations progress
# (https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a)
def print_progress(iteration, total, prefix="", suffix="", decimals=1, bar_length=40, time_start=None):
    """Call in a loop to create terminal progress bar

    Args:
        iteration (int): (Required) current iteration
        total (int): (Required) total iterations
        prefix (str): (Optional) prefix string
        suffix (str): (Optional) suffix string
        decimals (int): (Optional) positive number of decimals in percent complete
        bar_length (int): (Optional) character length of bar
    """
    iteration = min(iteration, total)

    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    t = ""
    if time_start:
        elapsed = time.time() - time_start
        fps = iteration / elapsed
        eta = (total - iteration) / fps
        t = "[Elapsed: %s, ETA: %s, FPS: %.2f]" % (format_duration(elapsed), format_duration(eta), fps)

    sys.stdout.write("\r%-25s |%s| %5s%s %-40s%-30s" % (prefix, bar, percents, "%", suffix, t)),

    if iteration == total:
        sys.stdout.write("\n")
    sys.stdout.flush()

def format_duration(t):
    """Format duration in seconds to a nice string (e.g. "1h 5m 20s")
    Args:
        t (int / float): Duration in seconds

    Returns:
        str
    """
    hours, remainder = divmod(t, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    output = "%is" % seconds
    
    if (minutes > 0):
        output = "%im %s" % (minutes, output)
        
    if (hours > 0):
        output = "%ih %s" % (hours, output)

    return output

def image_write_label(image, label):
    """Write the specified label on an image for debug purposes
    (0: Unknown, 1: No anomaly, 2: Contains an anomaly)
    
    Args:
        image (Image)
    """
    
    text = {
        0: "Unknown",
        1: "No anomaly",
        2: "Contains anomaly"
    }

    colors = {
        0: (255,255,255),   # Unknown
        1: (0, 204, 0),     # No anomaly
        2: (0, 0, 255)      # Contains anomaly
    }

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,50)
    fontScale              = 0.5
    thickness              = 1

    cv2.putText(image,'Label: ',
        bottomLeftCornerOfText,
        font,
        fontScale,
        (255,255,255),
        thickness)
    
    cv2.putText(image, text.get(label, 0),
        (bottomLeftCornerOfText[0] + 50, bottomLeftCornerOfText[1]),
        font,
        fontScale,
        colors.get(label, 0),
        thickness)

# https://gist.github.com/nonZero/2907502
class GracefulInterruptHandler(object):

    def __init__(self, sig=signal.SIGINT):
        self.sig = sig

    def __enter__(self):

        self.interrupted = False
        self.released = False

        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self.release()
            self.interrupted = True

        signal.signal(self.sig, handler)

        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):

        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)

        self.released = True

        return True