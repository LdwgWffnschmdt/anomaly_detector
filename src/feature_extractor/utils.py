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

def visualize(metadata, features, feature_to_color_func, feature_to_text_func=None, pause_func=None):
    """Visualize features on the source image

    Args:
        metadata (list): Array of metadata for the features
        features (list): Array of features as extracted by a FeatureExtractor
        feature_to_color_func (function): Function converting a feature to a color (b, g, r)
        feature_to_text_func (function): Function converting a feature to a string
        pause_func (function): Function converting a feature to a boolean that pauses the video
    """
    total, height, width, depth = features.shape
    
    cv2.namedWindow('image')

    def nothing(x):
        pass

    # create trackbars
    cv2.createTrackbar('delay',   'image', 1 , 1000, nothing)
    cv2.createTrackbar('overlay', 'image', 40, 100 , nothing)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.25
    thickness = 1

    tfrecord = None
    tfrecordCounter = 0

    pause = False

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_xlim([-10,10])
    # ax.set_ylim([-10,10])
    # plt.ion()

    # fig.show()
    # fig.canvas.draw()

    for i, feature_3d in enumerate(features):
        meta = metadata[i]
        
        if tfrecord != meta["tfrecord"]:
            if not "2020-02-06-17-17-25.tfrecord" in meta["tfrecord"]: # TODO: Remove for production
                continue
            tfrecord = meta["tfrecord"]
            tfrecordCounter = 0
            image_dataset = utils.load_dataset(meta["tfrecord"]).as_numpy_iterator()
        else:
            tfrecordCounter += 1

        image, example = next(image_dataset)
        # if meta["label"] == 1:
        #     continue

        overlay = image.copy()

        # if example["metadata/time"] != meta["time"]:
        #     logging.error("Times somehow don't match (%f)" % (example["metadata/time"]- meta["time"]))

        patch_size = (image.shape[1] / width, image.shape[0] / height)
        
        # ax.clear()
        # ax.plot(meta["location/translation/x"], 
        #         meta["location/translation/y"], 'bo')

        # print "%.2f / %.2f / %.2f       |       %.2f / %.2f / %.2f" % (meta["location/translation/x"],
        #                                                                meta["location/translation/y"],
        #                                                                meta["location/translation/z"],
        #                                                                meta["location/rotation/x"],
        #                                                                meta["location/rotation/y"],
        #                                                                meta["location/rotation/z"])

        for x in range(width):
            for y in range(height):
                feature = feature_3d[y,x,:]

                # ax.plot(feature[-2], feature[-1], 'r+')

                p1 = (x * patch_size[0], y * patch_size[1])
                p2 = (p1[0] + patch_size[0], p1[1] + patch_size[1])
                cv2.rectangle(overlay, p1, p2, feature_to_color_func(feature), -1)

                if feature_to_text_func is not None:
                    text = str(feature_to_text_func(feature))
                    cv2.putText(overlay, text,
                        (p1[0] + 2, p1[1] + patch_size[1] - 2),
                        font,
                        fontScale,
                        (0, 0, 255),
                        thickness)
                
                if pause_func is not None and pause_func(feature):
                    pause = True
        
        # fig.canvas.draw()
        
        alpha = cv2.getTrackbarPos('overlay','image') / 100.0  # Transparency factor.

        # Following line overlays transparent overlay over the image
        image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        utils.image_write_label(image_new, meta["label"])
        
        cv2.imshow("image", image_new)
        key = cv2.waitKey(0 if pause else cv2.getTrackbarPos('delay','image'))
        if key == 27:   # [esc] => Quit
            return None
        elif key != -1:
            pause = not pause
    
    cv2.destroyAllWindows()

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