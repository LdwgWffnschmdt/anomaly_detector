#! /usr/bin/env python
# -*- coding: utf-8 -*-

import consts
import argparse

parser = argparse.ArgumentParser(description="Benchmark the specified feature extractors.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--files", metavar="F", dest="files", type=str, default=consts.EXTRACT_FILES,
                    help="File(s) to use for benchmarks (*.tfrecord, *.jpg)")

parser.add_argument("--extractor", metavar="EXT", dest="extractor", nargs='*', type=str,
                    help="Extractor name. Leave empty for all extractors (default: \"\")")

parser.add_argument("--batch_sizes", metavar="B", dest="batch_sizes", nargs="*", type=int, default=[4,8,16,32,64,128,256,512],
                    help="Batch size for testing batched extraction. (default: [8,16,32,64,128,256,512])")

parser.add_argument("--init_repeat", metavar="B", dest="init_repeat", type=int, default=3,
                    help="Number of initialization repetitions. (default: 3)")

parser.add_argument("--extract_single_repeat", metavar="B", dest="extract_single_repeat", type=int, default=100,
                    help="Number of single extraction repetitions. (default: 100)")

parser.add_argument("--extract_batch_repeat", metavar="B", dest="extract_batch_repeat", type=int, default=10,
                    help="Number of batch extraction repetitions. (default: 10)")

args = parser.parse_args()

import os
import logging
from datetime import datetime
import inspect
import traceback
import timeit
from glob import glob

import numpy as np
import xlsxwriter
import tensorflow as tf
from tqdm import tqdm

import feature_extractor
import common.utils as utils

row = 0
col = 0

def feature_extractor_benchmark():
    global col

    
    if isinstance(args.files, basestring):
        args.files = [args.files]
        
    # Expand wildcards
    files_expanded = []
    for s in args.files:
        files_expanded += glob(s)
    files = list(set(files_expanded)) # Remove duplicates

    # Check parameters
    if not files or len(files) < 1 or files[0] == "":
        raise ValueError("Please specify at least one filename (%s)" % files)
        

    # Create a workbook and add a worksheet for meta data.
    workbook = xlsxwriter.Workbook(os.path.join(os.path.dirname(os.path.realpath(__file__)), datetime.now().strftime("%Y_%m_%d_%H_%M_benchmark.xlsx")))
    heading_format = workbook.add_format({'bold': True, 'font_color': '#0071bc', "font_size": 20})
    subheading_format = workbook.add_format({'bold': True, "font_size": 12})

    #Write metadata
    worksheet_meta = workbook.add_worksheet("Meta")
    worksheet_meta.set_column(0, 0, 25)
    worksheet_meta.set_column(1, 1, 40)

    
    def add_meta(n, s="", format=None):
        global row
        worksheet_meta.write(row, 0, n, format)
        worksheet_meta.write(row, 1, s, format)
        row += 1

    add_meta("Feature extractor benchmark", format=heading_format)

    add_meta("Start", datetime.now().strftime("%d.%m.%Y, %H:%M:%S"))
    add_meta("Batch size", str(args.batch_sizes))

    computer_info = utils.getComputerInfo()

    for key, value in computer_info.items():
        add_meta(key, value)

    # Get all the available feature extractor names
    extractor_names = map(lambda e: e[0], inspect.getmembers(feature_extractor, inspect.isclass))

    if args.extractor is None:
        args.extractor = extractor_names

    # Get an instance of each class
    module = __import__("feature_extractor")
    
    for extractor_name in tqdm(args.extractor, desc="Benchmarking extractors"):
        worksheet = workbook.add_worksheet(extractor_name.replace("FeatureExtractor", ""))
        worksheet.set_column(0, 20, 20)

        col = 0

        def log(s, times):
            """Log duration t with info string s"""
            global col
            
            logging.info("%-40s (%s): %.5fs  -  %.5fs" % (extractor_name, s, np.min(times), np.max(times)))
            
            worksheet.write(0, col, s, subheading_format)
            for i, t in enumerate(times):
                worksheet.write_number(i + 1, col, t)
            col += 1

        def logerr(s, err):
            """Log duration t with info string s"""
            global col
            
            logging.error("%-40s (%s): %s" % (extractor_name, s, err))
            
            worksheet.write(0, col, s, subheading_format)
            worksheet.write_string(1, col, err)
            col += 1

        try:
            _class = getattr(module, extractor_name)

            # Test extractor initialization
            log("Initialization", np.array(timeit.repeat(lambda: _class(), number=1, repeat=args.init_repeat)))

            extractor = _class()
    
            # Load dataset
            if files[0].endswith(".tfrecord"):
                dataset = utils.load_tfrecords(files, preprocess_function=extractor.format_image)
            elif files[0].endswith(".jpg"):
                dataset = utils.load_jpgs(files, preprocess_function=extractor.format_image)
            else:
                raise ValueError("Supported file types are *.tfrecord and *.jpg")
            
            # Call internal transformations (eg. temporal windowing for 3D networks)
            dataset = extractor.__transform_dataset__(dataset)

            mins = []

            # Test single image extraction
            try:
                single = list(dataset.take(1).as_numpy_iterator())[0] # Get a single entry
                times = np.array(timeit.repeat(lambda: extractor.extract(single[0]), number=1, repeat=args.extract_single_repeat))
                mins.append(np.min(times))
                log("Extract single", times)
            except KeyboardInterrupt:
                raise KeyboardInterrupt()
            except:
                logerr("Extract single", traceback.format_exc())
            
            # Test batch extraction
            for batch_size in args.batch_sizes:
                try:
                    batch = list(dataset.batch(batch_size).take(1).as_numpy_iterator())[0]
                    times = np.array(timeit.repeat(lambda: extractor.extract_batch(batch[0]), number=1, repeat=args.extract_batch_repeat))
                    times = times / batch_size
                    mins.append(np.min(times))
                    log("Extract batch (%i)" % batch_size, times)
                except KeyboardInterrupt:
                    raise KeyboardInterrupt()
                except:
                    logerr("Extract batch (%i)" % batch_size, traceback.format_exc())

            log("Extraction minima", mins)
        except KeyboardInterrupt:
            logging.info("Cancelled")
            break
        except:
            logerr("Error?", traceback.format_exc())
    workbook.close()

if __name__ == "__main__":
    feature_extractor_benchmark()