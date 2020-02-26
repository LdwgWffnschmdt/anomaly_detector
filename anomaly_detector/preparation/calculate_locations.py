#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import time
import argparse

parser = argparse.ArgumentParser(description="A small script to add patch locations to feature files",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("files", metavar="F", type=str, nargs='+',
                    help="The feature file(s). Supports \"path/to/*.h5\"")

args = parser.parse_args()

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
    
    # Get all bags in a folder if the file ends with *.h5
    if len(files) == 1 and files[0].endswith("*.h5"):
        path = files[0].replace("*.h5", "")
        files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(".h5")]
        if len(files) < 1:
            raise ValueError("There is no *.h5 file in %s." % path)

    for features_file in files:
        # Check parameters
        if features_file == "" or not os.path.exists(features_file) or not os.path.isfile(features_file):
            logging.error("Specified feature file does not exist (%s)" % features_file)
            return

        logging.info("Calculating features for %s" % features_file)
        
        # Load the file
        features = FeatureArray(features_file)

        # Calculate and save the locations
        features.save_locations_to_file()

if __name__ == "__main__":
    calculate_locations()