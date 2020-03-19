#! /usr/bin/env python
# -*- coding: utf-8 -*-

import consts
import argparse

parser = argparse.ArgumentParser(description="Change labels in the supplied h5 files and .",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--files", metavar="F", dest="files", type=str, nargs='*', default=consts.FEATURES_FILES,
                    help="The feature file(s). Read from first, write labels to all. Supports \"path/to/*.h5\"")

args = parser.parse_args()

import os
import sys
import time
import traceback
from glob import glob
import numpy as np

from common import utils, logger, FeatureArray, Visualize

last_index = 0
label = -1

def relabel():
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
    files = list(set(files_expanded)) # Remove duplicates

    # Check parameters
    features_file = files[1]
    if features_file == "" or not os.path.exists(features_file) or not os.path.isfile(features_file):
        logger.error("Specified feature file does not exist (%s)" % features_file)
        return

    # Load the file
    features = FeatureArray(features_file)

    # Visualize
    vis = Visualize(features)
    vis.pause = True

    def _key_func(v, key):
        global last_index, label

        # Get the new label from now on
        
        if key == 48:   # [0]   => Unknown
            new_label = 0
        elif key == 49: # [1]   => No anomaly
            new_label = 1
        elif key == 50: # [2]   => Contains anomaly
            new_label = 2
        else:
            new_label = label
        
        if new_label == -1:
            v.pause = True
            return

        # If we skipped back
        if v.index < last_index:
            last_index = v.index
        else:
            # Label this frame with the new label
            features[v.index, 0, 0].label = new_label
            v.__draw__()
        
        indices = range(last_index, v.index)
        last_index = v.index
        
        # Label recent frames with the old label
        for i in indices:
            features[i, 0, 0].label = label

        label = new_label

    vis.key_func = _key_func

    vis.show()

    labels = np.array([x.label for x in features[:, 0, 0]])
    
    # TODO: Save these labels

if __name__ == "__main__":
    relabel()
    pass