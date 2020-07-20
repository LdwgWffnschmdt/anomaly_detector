#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import consts
import argparse

parser = argparse.ArgumentParser(description="Label images.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--images", dest="images", metavar="F", type=str, default=consts.IMAGES_PATH,
                    help="Path to images (default: %s)" % consts.IMAGES_PATH)

args = parser.parse_args()

from common import logger, PatchArray, Visualize

def relabel():
    # Check parameters
    if args.images == "" or not os.path.exists(args.images) or not os.path.isdir(args.images):
        logger.error("Specified path does not exist (%s)" % args.images)
        return

    # Load the file
    patches = PatchArray(args.images)

    # Visualize
    vis = Visualize(patches, images_path=args.images)
    vis.pause = True
    vis.show()

if __name__ == "__main__":
    relabel()
    pass