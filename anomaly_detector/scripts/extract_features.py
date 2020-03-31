#! /usr/bin/env python
# -*- coding: utf-8 -*-

import consts
import argparse

parser = argparse.ArgumentParser(description="Extract features from tfrecords.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--list", dest="list", action="store_true",
                    help="List all extractors and exit")

parser.add_argument("--files", metavar="F", dest="files", type=str, nargs='*', default=consts.EXTRACT_FILES,
                    help="File(s) to use (*.tfrecord, *.jpg)")

parser.add_argument("--filter", metavar="FI", dest="filter", type=str,# default="direction_ccw",
                    help="Filter only works with *.jpg files (\"no_anomaly\", \"anomaly\", \"round_number\", \"direction_ccw\", ...)")

parser.add_argument("--filter_argument", metavar="FIA", dest="filter_argument", type=int,
                    help="Use with --filter=\"round_number\"")

parser.add_argument("--extractor", metavar="EXT", dest="extractor", nargs='*', type=str,
                    help="Extractor name. Leave empty for all extractors (default: \"\")")

args = parser.parse_args()

import os
import time
from tqdm import tqdm
from common import utils, logger, PatchArray
import traceback


def extract_features():
    if not args.list and len(args.files) == 0:
        logger.error("No input file specified.")
        return

    import tensorflow as tf
    import inspect
    import feature_extractor as feature_extractor

    # Add before any TF calls (https://github.com/tensorflow/tensorflow/issues/29931#issuecomment-504217770)
    # Initialize the keras global outside of any tf.functions
    temp = tf.zeros([4, 32, 32, 3])
    tf.keras.applications.vgg16.preprocess_input(temp)

    # Get all the available feature extractor names
    extractor_names = map(lambda e: e[0], inspect.getmembers(feature_extractor, inspect.isclass))

    if args.list:
        for e in extractor_names:
            print e
        return

    if args.extractor is None:
        args.extractor = extractor_names

    if isinstance(args.files, basestring):
        args.files = [args.files]
        
    dataset = None
    total = 0

    if args.files[0].endswith(".jpg") and args.filter is not None:
        patches = PatchArray(args.files)
        patches = getattr(patches, args.filter, None)
        assert patches is not None, "The filter was not valid."
        if args.filter_argument is not None:
            patches = patches(args.filter_argument)
            assert patches is not None, "The filter argument was not valid."
        dataset = patches.to_dataset()
        total = patches.shape[0]
    else:
        dataset, total = utils.load_dataset(args.files)

    module = __import__("feature_extractor")
    
    # Add progress bar if multiple extractors
    if len(args.extractor) > 1:
        args.extractor = tqdm(args.extractor, desc="Extractors", file=sys.stderr)

    for extractor_name in args.extractor:
        logger.info("Instantiating %s" % extractor_name)
        try:
            # Get an instance
            extractor = getattr(module, extractor_name)()
            extractor.extract_dataset(dataset, total, filter=args.filter)
        except KeyboardInterrupt:
            logger.info("Terminated by CTRL-C")
            return
        except:
            logger.error("%s: %s" % (extractor_name, traceback.format_exc()))

if __name__ == "__main__":
    extract_features()