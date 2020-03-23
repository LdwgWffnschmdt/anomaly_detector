#! /usr/bin/env python
# -*- coding: utf-8 -*-

import consts
import argparse

parser = argparse.ArgumentParser(description="Add patch locations to feature files.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--files", metavar="F", dest="files", type=str, nargs='*', default=consts.FEATURES_FILES,
                    help="The feature file(s). Supports \"path/to/*.h5\"")

args = parser.parse_args()

import os
import sys
import time
import traceback
from glob import glob

from tqdm import tqdm

from common import utils, logger, FeatureArray

def calculate_locations():
    ################
    #  Parameters  #
    ################
    files       = args.files

    # Check parameters
    if not files or len(files) < 1 or files[0] == "":
        raise ValueError("Please specify at least one filename (%s)" % files)
    
    if isinstance(files, basestring):
        files = [files]
        
    # Expand wildcards
    files_expanded = []
    for s in files:
        files_expanded += glob(s)
    files = sorted(list(set(files_expanded))) # Remove duplicates

    for features_file in tqdm(files, file=sys.stderr):
        # Check parameters
        if features_file == "" or not os.path.exists(features_file) or not os.path.isfile(features_file):
            logger.error("Specified feature file does not exist (%s)" % features_file)
            return

        logger.info("Calculating locations for %s" % features_file)
        
        try:
            # Load the file
            features = FeatureArray(features_file)

            # Calculate and save the locations
            features.save_locations_to_file()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            logger.error("%s: %s" % (features_file, traceback.format_exc()))

if __name__ == "__main__":
    calculate_locations()
    pass