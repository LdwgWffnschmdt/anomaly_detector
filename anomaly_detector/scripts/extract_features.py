#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser(description="Extract features from tfrecords.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--list", dest="list", action="store_true",
                    help="List all extractors and exit")

parser.add_argument("files", metavar="F", type=str, nargs='*',
                    help="The feature file(s). Supports \"path/to/*.tfrecord\"")

parser.add_argument("--extractor", metavar="EXT", dest="extractor", nargs='*', type=str,
                    help="Extractor name. Leave empty for all extractors (default: \"\")")

args = parser.parse_args()

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.INFO)

if not args.list and len(args.files) == 0:
    logging.error("No input file specified.")
    return

import os
import time
import logging

import inspect
import feature_extractor as feature_extractor

def extract_features():
    # Get all the available feature extractor names
    extractor_names = map(lambda e: e[0], inspect.getmembers(feature_extractor, inspect.isclass))

    if args.list:
        for e in extractor_names:
            print e
        return

    if args.extractor is None:
        args.extractor = extractor_names

    module = __import__("feature_extractor")
    
    try:
        for extractor_name in args.extractor:
            logging.info("Instantiating %s" % extractor_name)

            # Get an instance
            extractor = getattr(module, extractor_name[0])()

            extractor.extract_files(args.files)
    finally:
        pass

if __name__ == "__main__":
    extract_features()