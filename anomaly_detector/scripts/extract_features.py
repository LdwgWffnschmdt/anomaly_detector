#! /usr/bin/env python
# -*- coding: utf-8 -*-

import consts
import argparse

parser = argparse.ArgumentParser(description="Extract features from tfrecords.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--list", dest="list", action="store_true",
                    help="List all extractors and exit")

parser.add_argument("--files", metavar="F", dest="files", type=str, nargs='*', default=consts.EXTRACT_FILES,
                    help="File(s) to use (*.jpg)")

parser.add_argument("--extractor", metavar="EXT", dest="extractor", nargs='*', type=str,
                    help="Extractor name. Leave empty for all extractors (default: \"\")")

args = parser.parse_args()

import os
import sys
import time
from tqdm import tqdm
from common import utils, logger, PatchArray, Visualize
import traceback

import numpy as np

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
    
    patches = PatchArray(args.files)
    
    p = patches[:, 0, 0]

    f = np.zeros(p.shape, dtype=np.bool)
    f[:] = np.logical_and(p.directions == 1,                                   # CCW and
                            np.logical_or(p.labels == 2,                         #   Anomaly or
                                        np.logical_and(p.round_numbers >= 7,   #     Round between 2 and 5
                                                        p.round_numbers <= 9)))

    # Let's make contiguous blocks of at least 10, so
    # we can do some meaningful temporal smoothing afterwards
    for i, b in enumerate(f):
        if b and i - 10 >= 0:
            f[i - 10:i] = True

    patches = patches[f]

    vis = Visualize(patches)
    vis.show()
    
    # patches = getattr(patches, args.filter, None)
    # assert patches is not None, "The filter was not valid."
    # if args.filter_argument is not None:
    #     patches = patches(args.filter_argument)
    #     assert patches is not None, "The filter argument was not valid."
    dataset = patches.to_dataset().cache()
    dataset_3D = patches.to_temporal_dataset(16).cache()
    total = patches.shape[0]

    module = __import__("feature_extractor")
    
    # Add progress bar if multiple extractors
    if len(args.extractor) > 1:
        args.extractor = tqdm(args.extractor, desc="Extractors", file=sys.stderr)

    for extractor_name in args.extractor:
        logger.info("Instantiating %s" % extractor_name)
        try:
            bs = getattr(module, extractor_name).TEMPORAL_BATCH_SIZE
            # Get an instance
            if bs > 1:
                extractor = getattr(module, extractor_name)()
                extractor.extract_dataset(dataset_3D, total)
            else:
                pass
                # extractor.extract_dataset(dataset, total)
        except KeyboardInterrupt:
            logger.info("Terminated by CTRL-C")
            return
        except:
            logger.error("%s: %s" % (extractor_name, traceback.format_exc()))

if __name__ == "__main__":
    extract_features()