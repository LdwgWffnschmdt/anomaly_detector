# -*- coding: utf-8 -*-

import os
from common import logger
import sys
import time
import signal
import yaml

import tensorflow as tf
import numpy as np
import h5py
import cv2

###############
#  Images IO  #
###############

def load_tfrecords(filenames, batch_size=64, preprocess_function=None):
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
        return tf.io.parse_example(example_proto, feature_description)

    def _decode_function(example):
        image = tf.image.decode_jpeg(example["image/encoded"], channels=3)
        location = [example["metadata/location/translation/x"],
                    example["metadata/location/translation/y"],
                    example["metadata/location/translation/z"],
                    example["metadata/location/rotation/x"],
                    example["metadata/location/rotation/y"],
                    example["metadata/location/rotation/z"]]
        time      = example["metadata/time"]
        label     = example["metadata/label"]
        rosbag    = example["metadata/rosbag"]
        tfrecord  = example["metadata/tfrecord"]
        if preprocess_function is not None:
            image = preprocess_function(image)
        return image, location, time, label, rosbag, tfrecord

    return raw_dataset.batch(batch_size) \
                      .map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                      .unbatch() \
                      .map(_decode_function, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                      .cache() \
                      .prefetch(tf.data.experimental.AUTOTUNE)

def load_jpgs(filenames, preprocess_function=None):
    """Loads a set of TFRecord files
    Args:
        filenames (str / str[]): JPEG file(s) extracted
                                 by rosbag_to_images

    Returns:
        tf.data.MapDataset
    """
    if not filenames or len(filenames) < 1 or filenames[0] == "":
        raise ValueError("Please specify at least one filename (%s)" % filenames)
    
    raw_dataset = tf.data.Dataset.list_files(filenames)

    def _load_metadata(yml):
        # Load and decode metadata file
        with open(str(yml.numpy()), "r") as file:
            meta = yaml.safe_load(file)

            location = [meta["location/translation/x"],
                        meta["location/translation/y"],
                        meta["location/translation/z"],
                        meta["location/rotation/x"],
                        meta["location/rotation/y"],
                        meta["location/rotation/z"]]
            time      = meta["time"]
            label     = meta["label"]
            rosbag    = meta["rosbag"]
            tfrecord  = tf.constant("", dtype=tf.string) # Nuffin
        return location, time, label, rosbag, tfrecord

    def _decode_function(file_path):
        # Load and decode image
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)

        # Get the metadata file path
        yml = tf.strings.regex_replace(file_path, ".jpg", ".yml")

        location, time, label, rosbag, tfrecord = tf.py_function(_load_metadata, [yml], [tf.float32,
                                                                                         tf.int64,
                                                                                         tf.int8,
                                                                                         tf.string,
                                                                                         tf.string])

        if preprocess_function is not None:
            image = preprocess_function(image)
        return image, location, time, label, rosbag, tfrecord

    return raw_dataset.map(_decode_function, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                      .cache() \
                      .prefetch(tf.data.experimental.AUTOTUNE)

#################
# Output helper #
#################

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

#################
# Computer Info #
#################

import cpuinfo
import GPUtil
from psutil import virtual_memory

def getComputerInfo():

    # Get CPU info
    # cpu = cpuinfo.get_cpu_info()

    result_dict = {
        # "Python version": cpu["python_version"],
        "TensorFlow version": tf.version.VERSION,
        # "CPU Description": cpu["brand"],
        # "CPU Clock speed (advertised)": cpu["hz_advertised"],
        # "CPU Clock speed (actual)": cpu["hz_actual"],
        # "CPU Architecture": cpu["arch"]
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
