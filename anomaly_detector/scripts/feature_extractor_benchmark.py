#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser(description="Benchmark the specified feature extractors.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("tfrecord", metavar="F", dest="tfrecord", type=str,
                    help="TFRecord file to use for benchmarks")

parser.add_argument("--extractor", metavar="EXT", dest="extractor", nargs='*', type=str,
                    help="Extractor name. Leave empty for all extractors (default: \"\")")

args = parser.parse_args()

import os
import logging
from datetime import datetime
import inspect
import timeit

import xlsxwriter
import tensorflow as tf

import feature_extractor
import common.utils as utils

def feature_extractor_benchmark():
    BATCH_SIZE = 32
    
    # Create a workbook and add a worksheet for meta data.
    workbook = xlsxwriter.Workbook(os.path.join(os.path.dirname(os.path.realpath(__file__)), datetime.now().strftime("%Y_%m_%d_%H_%M_benchmark.xlsx")))
    heading_format = workbook.add_format({'bold': True, 'font_color': '#0071bc', "font_size": 20})
    subheading_format = workbook.add_format({'bold': True, "font_size": 12})

    #Write metadata
    worksheet_meta = workbook.add_worksheet("Meta")
    worksheet_meta.set_column(0, 0, 25)
    worksheet_meta.set_column(1, 1, 40)

    row = 0
    def add_meta(n, s="", format=None):
        global row
        worksheet_meta.write(row, 0, n, format)
        worksheet_meta.write(row, 1, s, format)
        row += 1

    add_meta("Feature extractor benchmark", format=heading_format)

    add_meta("Start", datetime.now().strftime("%d.%m.%Y, %H:%M:%S"))

    computer_info = utils.getComputerInfo()

    for key, value in computer_info.items():
        add_meta(key, value)

    # Get all the available feature extractor names
    extractor_names = map(lambda e: e[0], inspect.getmembers(feature_extractor, inspect.isclass))

    if args.extractor is None:
        args.extractor = extractor_names

    # Get an instance of each class
    module = __import__("feature_extractor")
    
    try:
        for extractor_name in args.extractor:
            worksheet = workbook.add_worksheet(extractor_name.replace("FeatureExtractor", ""))
            worksheet.set_column(0, 20, 20)

            col = 0
            def add_to_excel(n, times):
                global col
                worksheet.write(0, col, n, subheading_format)
                for i, t in enumerate(times):
                    worksheet.write_number(i + 1, col, t)
                col += 1

            def log(s, t):
                """Log duration t with info string s"""
                logging.info("%-40s (%s): %s" % (extractor_name, s, str(t)))
                add_to_excel(s, t)

            _class = getattr(module, extractor_name)

            # Test extractor initialization
            log("Initialization", timeit.repeat(lambda: _class(), number=1, repeat=5))

            # Load a test dataset
            dataset = utils.load_tfrecords(args.tfrecord)

            # Test batch extraction
            extractor = _class()
            batch = list(dataset.take(BATCH_SIZE).batch(BATCH_SIZE).as_numpy_iterator())[0]
            log("Extract batch", timeit.repeat(lambda: extractor.extract_batch(batch[0]), number=1, repeat=10))

            # Test single image extraction
            extractor = _class()
            single = list(dataset.take(1).as_numpy_iterator())[0] # Get a single entry
            log("Extract single", timeit.repeat(lambda: extractor.extract(single[0]), number=1, repeat=10))
    finally:
        workbook.close()

if __name__ == "__main__":
    feature_extractor_benchmark()