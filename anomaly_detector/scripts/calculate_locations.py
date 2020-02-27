#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser(description="A small script to add patch locations to feature files.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("files", metavar="F", type=str, nargs='+',
                    help="The feature file(s). Supports \"path/to/*.h5\"")

args = parser.parse_args()

import os
import time
import logging
from glob import glob

import common.utils as utils
from common import FeatureArray

def calculate_locations():
    ################
    #  Parameters  #
    ################
    files       = args.files

    # Check parameters
    if not files or len(files) < 1 or files[0] == "":
        raise ValueError("Please specify at least one filename (%s)" % files)
    
    # Expand wildcards
    files_expanded = []
    for s in files:
        files_expanded += glob(s)
    files = list(set(files_expanded)) # Remove duplicates

    for features_file in files:
        # Check parameters
        if features_file == "" or not os.path.exists(features_file) or not os.path.isfile(features_file):
            logging.error("Specified feature file does not exist (%s)" % features_file)
            return

        logging.info("Calculating locations for %s" % features_file)
        
        # Load the file
        features = FeatureArray(features_file)

        # Calculate and save the locations
        features.save_locations_to_file()

if __name__ == "__main__":
    calculate_locations()