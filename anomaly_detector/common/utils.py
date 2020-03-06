# -*- coding: utf-8 -*-

import os
import logging
import sys
import time
import signal

import tensorflow as tf
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt

from imageLocationUtility import ImageLocationUtility

# Configure logging
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.INFO)

###############
#  Images IO  #
###############

def load_tfrecords(filenames):
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
        "metadata/time"                     : tf.io.FixedLenFeature([], tf.int64), # TODO: Change to int64
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
        image = tf.image.decode_jpeg(example["image/encoded"], channels=3)
        return image, example

    return raw_dataset.map(_parse_function)

#################
# Output helper #
#################

def visualize(features, threshold, feature_to_color_func=None, feature_to_text_func=None, pause_func=None, show_grid=False, show_map=True):
    """Visualize features on the source image

    Args:
        features (FeatureArray): Array of features as extracted by a FeatureExtractor
        feature_to_color_func (function): Function converting a feature to a color (b, g, r)
        feature_to_text_func (function): Function converting a feature to a string
        pause_func (function): Function converting a feature to a boolean that pauses the video
        show_grid (bool): Overlay real world coordinate grid
    """
    image, example, feature_2d = (None, None, None)
    total, height, width = features.shape
    
    ### Set up window
    cv2.namedWindow('image')

    has_locations = features[0,0,0].location is not None

    if has_locations:
        x_min, y_min, x_max, y_max = features.get_extent()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

        plt.ion()
        fig.show()
    else:
        show_grid = False
        show_map = False

    def __draw__(x=None):
        overlay = image.copy()

        # if example["metadata/time"] != meta["time"]:
        #     logging.error("Times somehow don't match (%f)" % (example["metadata/time"]- meta["time"]))

        patch_size = (image.shape[1] / width, image.shape[0] / height)
        
        # print "%.2f / %.2f / %.2f       |       %.2f / %.2f / %.2f" % (meta["location/translation/x"],
        #                                                                meta["location/translation/y"],
        #                                                                meta["location/translation/z"],
        #                                                                meta["location/rotation/x"],
        #                                                                meta["location/rotation/y"],
        #                                                                meta["location/rotation/z"])

        threshold = cv2.getTrackbarPos('threshold', 'image')
        if has_locations:
            show_grid = bool(cv2.getTrackbarPos('show_grid', 'image'))
            show_map  = bool(cv2.getTrackbarPos('show_map', 'image'))
        else:
            show_grid = False
            show_map = False
        show_values = bool(cv2.getTrackbarPos('show_values', 'image'))
        show_thresh = bool(cv2.getTrackbarPos('show_thresh', 'image'))

        if show_map:
            ax.clear()
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])

            ax.fill([feature_2d[0 ,  0].location[0],
                     feature_2d[-1,  0].location[0],
                     feature_2d[-1, -1].location[0],
                     feature_2d[0 , -1].location[0]],
                    [feature_2d[0 ,  0].location[1],
                     feature_2d[-1,  0].location[1],
                     feature_2d[-1, -1].location[1],
                     feature_2d[0 , -1].location[1]])

            # ax.plot(feature_2d[0 ,  0].location[0], feature_2d[0 ,  0].location[1], 'r+', markersize=2, linewidth=2)
            # ax.plot(feature_2d[-1,  0].location[0], feature_2d[-1,  0].location[1], 'r+', markersize=2, linewidth=2)
            # ax.plot(feature_2d[0 , -1].location[0], feature_2d[0 , -1].location[1], 'r+', markersize=2, linewidth=2)
            # ax.plot(feature_2d[-1, -1].location[0], feature_2d[-1, -1].location[1], 'r+', markersize=2, linewidth=2)
            # # ax.plot(meta["location/translation/x"], 
            #         meta["location/translation/y"], 'bo')
            fig.canvas.draw()

        for x in range(width):
            for y in range(height):
                feature = feature_2d[y,x]

                p1 = (x * patch_size[0], y * patch_size[1])
                p2 = (p1[0] + patch_size[0], p1[1] + patch_size[1])
                
                if feature_to_color_func is not None:
                    cv2.rectangle(overlay, p1, p2, feature_to_color_func(feature, threshold, show_thresh), -1)

                if feature_to_text_func is not None and show_values:
                    text = str(feature_to_text_func(feature, threshold))
                    cv2.putText(overlay, text,
                        (p1[0] + 2, p1[1] + patch_size[1] - 2),
                        font,
                        fontScale,
                        (0, 0, 255),
                        thickness, lineType=cv2.LINE_AA)
                
                if pause_func is not None and pause_func(feature, threshold):
                    pause = True
        
        if show_grid:
            relative_grid = ilu.absolute_to_relative(absolute_locations, feature_2d[0,0].camera_position, feature_2d[0,0].camera_rotation)
            image_grid = ilu.relative_to_image(relative_grid, image.shape[1], image.shape[0])
        
            for a in range(image_grid.shape[0]):
                for b in range(image_grid.shape[1]):
                    pos = (int(image_grid[a][b][0]), int(image_grid[a][b][1]))
                    if pos[0] < 0 or pos[0] > image.shape[1] or pos[1] < 0 or pos[1] > image.shape[0]:
                        continue
                    cv2.circle(overlay, pos, 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                    
                    cv2.putText(overlay, "%.1f / %.1f" % (absolute_locations[a][b][0], absolute_locations[a][b][1]),
                        (pos[0] + 3, pos[1] + 2),
                        font,
                        fontScale,
                        (255, 255, 255),
                        thickness, lineType=cv2.LINE_AA)

        alpha = cv2.getTrackbarPos('overlay','image') / 100.0  # Transparency factor.

        # Following line overlays transparent overlay over the image
        image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        image_write_label(image_new, feature_2d[0,0].label)
        cv2.imshow("image", image_new)
    
    # create trackbars
    cv2.createTrackbar('threshold',   'image', int(threshold) , int(threshold) * 3, __draw__)
    cv2.createTrackbar('show_thresh', 'image', 1 ,              1,                  __draw__)
    if has_locations:
        cv2.createTrackbar('show_grid',   'image', int(show_grid) , 1,                  __draw__)
        cv2.createTrackbar('show_map',    'image', int(show_map)  , 1,                  __draw__)
    cv2.createTrackbar('show_values', 'image', 0 ,              1,                  __draw__)
    cv2.createTrackbar('delay',       'image', 1 ,              1000,               __draw__)
    cv2.createTrackbar('overlay',     'image', 40,              100 ,               __draw__)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.25
    thickness = 1

    ### Calculate grid overlay
    if has_locations:
        ilu = ImageLocationUtility()
        absolute_locations = ilu.span_grid(60, 60, 1, -30, -30)

    tfrecord = None
    tfrecordCounter = 0

    pause = False

    for i, feature_2d in enumerate(features):
        if tfrecord != feature_2d[0,0].tfrecord:
            # if not "2020-02-06-17-17-25.tfrecord" in feature_2d[0,0].tfrecord: # TODO: Remove for production
            #     continue
            tfrecord = feature_2d[0,0].tfrecord
            tfrecordCounter = 0
            image_dataset = load_tfrecords(feature_2d[0,0].tfrecord).as_numpy_iterator()
        else:
            tfrecordCounter += 1

        image, example = next(image_dataset)

        # if meta["label"] == 1:
        #     continue

        __draw__()
        
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
        eta = 0 if fps == 0 else (total - iteration) / fps
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
        thickness, lineType=cv2.LINE_AA)
    
    cv2.putText(image, text.get(label, 0),
        (bottomLeftCornerOfText[0] + 50, bottomLeftCornerOfText[1]),
        font,
        fontScale,
        colors.get(label, 0),
        thickness, lineType=cv2.LINE_AA)

#################
# Computer Info #
#################

import cpuinfo
import GPUtil
from psutil import virtual_memory

def getComputerInfo():

    # Get CPU info
    cpu = cpuinfo.get_cpu_info()

    result_dict = {
        "Python version": cpu["python_version"],
        "TensorFlow version": tf.version.VERSION,
        "CPU Description": cpu["brand"],
        "CPU Clock speed (advertised)": cpu["hz_advertised"],
        "CPU Clock speed (actual)": cpu["hz_actual"],
        "CPU Architecture": cpu["arch"]
    }

    # Get GPU info
    gpus_tf = tf.config.experimental.list_physical_devices("GPU")
    
    result_dict["Number of GPUs (tf)"] = len(gpus_tf)

    gpus = GPUtil.getGPUs()
    gpus_available = GPUtil.getAvailability(gpus)
    for i, gpu in enumerate(gpus):
        result_dict["GPU %i" % gpu.id] = gpu.name
        result_dict["GPU %i (driver)" % gpu.id] = gpu.driver
        result_dict["GPU %i (memory total)" % gpu.id] = gpu.memoryTotal
        result_dict["GPU %i (memory free)" % gpu.id] = gpu.memoryFree
        result_dict["GPU %i (available?)" % gpu.id] = gpus_available[i]

    # Get RAM info
    mem = virtual_memory()

    result_dict["RAM (total)"] = mem.total
    result_dict["RAM (available)"] = mem.available

    return result_dict

#################
#     Misc      #
#################

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

class _DictObjHolder(object):
    def __init__(self, dct):
        self.dct = dct

    def __getattr__(self, name):
        return self.dct[name]
