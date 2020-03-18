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

parser.add_argument("--extractor", metavar="EXT", dest="extractor", nargs='*', type=str,
                    help="Extractor name. Leave empty for all extractors (default: \"\")")

args = parser.parse_args()

import os
import time
from common import logger
import traceback

def extract_features():
    if not args.list and len(args.files) == 0:
        logger.error("No input file specified.")
        return

    import inspect
    import feature_extractor as feature_extractor

    # Get all the available feature extractor names
    extractor_names = map(lambda e: e[0], inspect.getmembers(feature_extractor, inspect.isclass))

    if args.list:
        for e in extractor_names:
            print e
        return

    if args.extractor is None:
        args.extractor = extractor_names

    module = __import__("feature_extractor")
    
    for extractor_name in args.extractor:
        logger.info("Instantiating %s" % extractor_name)
        try:
            # Get an instance
            extractor = getattr(module, extractor_name)()
            extractor.extract_files(args.files)
        except KeyboardInterrupt:
            logger.info("Terminated by CTRL-C")
            return
        except:
            logger.error("%s: %s" % (extractor_name, traceback.format_exc()))

if __name__ == "__main__":
    extract_features()