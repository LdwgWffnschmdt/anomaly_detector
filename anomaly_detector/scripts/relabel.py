#! /usr/bin/env python
# -*- coding: utf-8 -*-

import consts
import argparse

parser = argparse.ArgumentParser(description="Change labels.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--images", metavar="F", dest="images", type=str, nargs='*', default=consts.IMAGES_PATH,
                    help="Path to images (Default: %s)" % consts.IMAGES_PATH)

args = parser.parse_args()

import os
import sys
import time
import traceback
from glob import glob
import numpy as np
from tqdm import tqdm

from common import utils, logger, FeatureArray, Visualize

last_index = 0
label = -1

def relabel():
    # Check parameters
    if args.images == "" or not os.path.exists(args.images) or not os.path.isdir(args.images):
        logger.error("Specified path does not exist (%s)" % args.images)
        return

    # Load the file
    features = FeatureArray(args.images)

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
        
        if label != -1:
            # Label recent frames with the old label
            for i in indices:
                features[i, 0, 0].label = label

        label = new_label

    vis.key_func = _key_func

    vis.show()

    # Save labels
    for i in tqdm(range(features.shape[0]), desc="Saving labels", file=sys.stderr):
        features[i, 0, 0].save_metadata()

if __name__ == "__main__":
    relabel()
    pass