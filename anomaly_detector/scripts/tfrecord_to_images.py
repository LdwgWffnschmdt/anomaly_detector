#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser(description="Extract images from tfrecords.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("files", metavar="F", type=str, nargs='*',
                    help="The feature file(s). Supports \"path/to/*.tfrecord\"")

parser.add_argument("--output_dir", metavar="OUT", dest="output_dir", type=str,
                    help="Output directory (default: {bag_file}/Images)")

args = parser.parse_args()

import os
import time
import sys
import traceback
import yaml
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from common import utils, logger

def tfrecord_to_images():
    files = args.files
    output_dir = args.output_dir

    # Expand wildcards
    files_expanded = []
    for s in files:
        files_expanded += glob(s)
    files = sorted(list(set(files_expanded))) # Remove duplicates

    # Check parameters
    if not files or len(files) < 1 or files[0] == "":
        logger.error("No input file specified.")
        return
    
    if output_dir is None or output_dir == "" or not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        output_dir = os.path.join(os.path.abspath(os.path.dirname(files[0])), "Images")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger.info("Output directory set to %s" % output_dir)

    parsed_dataset = utils.load_tfrecords(files)

    # Get number of examples in dataset
    total = sum(1 for record in tqdm(parsed_dataset, desc="Loading dataset"))

    # Add features to list
    for x in tqdm(parsed_dataset, desc="Extracting images", total=total, file=sys.stderr):
        output_file = os.path.join(output_dir, str(x[2].numpy()))
        cv2.imwrite(output_file + ".jpg", x[0].numpy())
        
        feature_dict = {
            "location/translation/x"   : float(x[1].numpy()[0]),
            "location/translation/y"   : float(x[1].numpy()[1]),
            "location/translation/z"   : float(x[1].numpy()[2]),
            "location/rotation/x"      : float(x[1].numpy()[3]),
            "location/rotation/y"      : float(x[1].numpy()[4]),
            "location/rotation/z"      : float(x[1].numpy()[5]),
            "time"                     : int(x[2].numpy()),
            "label"                    : int(x[3].numpy()), # 0: Unknown, 1: No anomaly, 2: Contains an anomaly
            "rosbag"                   : str(x[4].numpy())
        }

        with open(output_file + ".yml", "w") as yaml_file:
            yaml.dump(feature_dict, yaml_file, default_flow_style=False)
        


if __name__ == "__main__":
    tfrecord_to_images()