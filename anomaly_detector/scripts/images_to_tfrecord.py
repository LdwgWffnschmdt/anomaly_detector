#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser(description="Convert image files with metadata yaml to TensorFlow TFRecords.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("files", metavar="F", type=str, nargs='*',
                    help="The image file(s). Supports \"path/to/*.jpg\"")

parser.add_argument("--output_dir", metavar="OUT", dest="output_dir", type=str,
                    help="Output directory (default: {bag_file}/TFRecord)")

args = parser.parse_args()

import os
import sys
import time
import yaml
from glob import glob

from common import utils, logger

import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm

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

def images_to_tfrecord():
    files = args.files
    output_dir = args.output_dir

    # Expand wildcards
    files_expanded = []
    for s in files:
        files_expanded += glob(s)
    files = list(set(files_expanded)) # Remove duplicates

    # Check parameters
    if not files or len(files) < 1 or files[0] == "":
        logger.error("No input file specified.")
        return
    
    if output_dir is None or output_dir == "" or not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        output_dir = os.path.join(os.path.abspath(os.path.dirname(files[0])), "TFRecord")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger.info("Output directory set to %s" % output_dir)

    parsed_dataset = utils.load_jpgs(files)

    # Get number of examples in dataset
    total = sum(1 for record in tqdm(parsed_dataset, desc="Loading dataset"))

    tfWriter = None        

    colorspace = b"BGR"
    channels = 3

    for x in tqdm(parsed_dataset, desc="Writing TFRecord", total=total, file=sys.stderr):
        _, encoded = cv2.imencode(".jpeg", x[0].numpy())

        # Create a new writer if we need one
        if not tfWriter or per_bin_count >= images_per_bin:
            if tfWriter:
                tfWriter.close()
            
            if number_of_bins == 1:
                output_filename = "%s.tfrecord" % bag_file_name
            else:
                bin_number = total_saved_count // images_per_bin + 1
                output_filename = "%s.%.5d-of-%.5d.tfrecord" % (bag_file_name, bin_number, number_of_bins)
                
            output_file = os.path.join(output_dir, output_filename)
            tfWriter = tf.io.TFRecordWriter(output_file)
            per_bin_count = 0

        # Add image and position to TFRecord
        feature_dict = {
            "metadata/location/translation/x"   : _float_feature(float(x[1].numpy()[0])),
            "metadata/location/translation/y"   : _float_feature(float(x[1].numpy()[1])),
            "metadata/location/translation/z"   : _float_feature(float(x[1].numpy()[2])),
            "metadata/location/rotation/x"      : _float_feature(float(x[1].numpy()[3])),
            "metadata/location/rotation/y"      : _float_feature(float(x[1].numpy()[4])),
            "metadata/location/rotation/z"      : _float_feature(float(x[1].numpy()[5])),
            "metadata/time"                     : _int64_feature(int(x[2].numpy())),
            "metadata/label"                    : _int64_feature(int(x[3].numpy())), # 0: Unknown, 1: No anomaly, 2: Contains an anomaly
            "metadata/rosbag"                   : _bytes_feature(str(x[4].numpy())),
            "metadata/tfrecord"                 : _bytes_feature(output_file),
            "image/height"      : _int64_feature(x[0].numpy().shape[0]),
            "image/width"       : _int64_feature(x[0].numpy().shape[1]),
            "image/channels"    : _int64_feature(channels),
            "image/colorspace"  : _bytes_feature(colorspace),
            "image/format"      : _bytes_feature("jpeg"),
            "image/encoded"     : _bytes_feature(encoded.tobytes())
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        
        tfWriter.write(example.SerializeToString())
        per_bin_count += 1
        total_saved_count += 1

    if tfWriter:
        tfWriter.close()

if __name__ == "__main__":
    images_to_tfrecord()