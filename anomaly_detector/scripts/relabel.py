#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import consts
import argparse

parser = argparse.ArgumentParser(description="Change labels.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--images", dest="images", metavar="F", type=str, default=consts.IMAGES_PATH,
                    help="Path to images (default: %s)" % consts.IMAGES_PATH)

parser.add_argument("--single", dest="single", action="store_true",
                    help="Only label single frames (default: True)")

args = parser.parse_args()

from common import logger, FeatureArray, Visualize

def relabel():
    # Check parameters
    if args.images == "" or not os.path.exists(args.images) or not os.path.isdir(args.images):
        logger.error("Specified path does not exist (%s)" % args.images)
        return

    # Load the file
    features = FeatureArray(args.images)
    features.preload_metadata()

    # Visualize
    vis = Visualize(features, images_path=args.images)
    vis.pause = True
    vis.show()

if __name__ == "__main__":
    relabel()
    pass