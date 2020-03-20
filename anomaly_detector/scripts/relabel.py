#! /usr/bin/env python
# -*- coding: utf-8 -*-

import consts
import argparse

parser = argparse.ArgumentParser(description="Change labels.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("images", metavar="F", type=str, default=consts.IMAGES_PATH,
                    help="Path to images (default: %s)" % consts.IMAGES_PATH)

parser.add_argument("--single", dest="single", action="store_true",
                    help="Only label single frames (default: True)")

parser.add_argument("--mode", metavar="M", dest="mode", type=str, default="label",
                    help="Set \"label\" or \"direction_and_round\" (default: \"label\")")

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
direction = 0
round_number = 0

def relabel():
    # Check parameters
    if args.images == "" or not os.path.exists(args.images) or not os.path.isdir(args.images):
        logger.error("Specified path does not exist (%s)" % args.images)
        return

    # Load the file
    features = FeatureArray(args.images)

    # Visualize
    vis = Visualize(features, images_path=args.images)
    vis.pause = True

    def _key_func_label(v, key):
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
        
        # Stop here if we just want to label single images
        if args.single:
            return

        indices = range(last_index, v.index)
        last_index = v.index
        
        if label != -1:
            # Label recent frames with the old label
            for i in indices:
                features[i, 0, 0].label = label

        label = new_label

    def _key_func_dir_round(v, key):
        global last_index, direction, round_number

        # Get the new label from now on
        
        if key == 44:   # [,]   => Direction is CCW
            new_direction = 1
        elif key == 46: # [.]   => Direction is CW
            new_direction = 2
        elif key == 35: # [#]   => Direction unknown
            new_direction = 0
        elif key == 43: # [+]   => Increase round number by 1
            round_number += 1
            new_direction = direction
        elif key == 45: # [-]   => Decrease round number by 1
            round_number -= 1
            new_direction = direction
        else:
            new_direction = direction
        
        if new_direction == None:
            v.pause = True
            return

        # If we skipped back
        if v.index < last_index:
            last_index = v.index
        else:
            # Set the metadata for this frame
            features[v.index, 0, 0].direction   = new_direction
            features[v.index, 0, 0].round_number = round_number
            v.__draw__()
        
        # Stop here if we just want to label single images
        if args.single:
            return

        indices = range(last_index, v.index)
        last_index = v.index
        
        if direction != -1:
            # Set recent frames with the old metadata
            for i in indices:
                features[i, 0, 0].direction   = direction
                features[i, 0, 0].round_number = round_number

        direction = new_direction

    if args.mode == "label":
        vis.key_func = _key_func_label
    elif args.mode == "direction_and_round":
        vis.key_func = _key_func_dir_round
    else:
        raise ValueError("Mode not supported")

    vis.show()

    # Save metadata
    features.save_metadata()

if __name__ == "__main__":
    relabel()
    pass