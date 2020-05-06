#! /usr/bin/env python
# -*- coding: utf-8 -*-

import consts
import argparse

parser = argparse.ArgumentParser(description="Benchmark the specified feature extractors.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--files", metavar="F", dest="files", type=str, default=consts.EXTRACT_FILES_TEST,
                    help="File(s) to use for benchmarks (*.tfrecord, *.jpg)")

parser.add_argument("--extractor", metavar="EXT", dest="extractor", nargs="*", type=str,
                    help="Extractor name. Leave empty for all extractors (default: \"\")")

parser.add_argument("--output", metavar="OUT", dest="output", type=str,
                    help="Output file (default: \"\")")

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
from common import utils, logger
import sys
from datetime import datetime
import inspect
import traceback
import timeit
from glob import glob

import numpy as np
import csv
import tensorflow as tf
from tqdm import tqdm
import subprocess

import feature_extractor

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
    files = sorted(list(set(files_expanded))) # Remove duplicates

    # Check parameters
    if not files or len(files) < 1 or files[0] == "":
        raise ValueError("Please specify at least one filename (%s)" % files)
        
    if args.output is None:
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), datetime.now().strftime("%Y_%m_%d_%H_%M_benchmark.csv"))
    else:
        filename = args.output
    
    write_header = not os.path.exists(filename)

    csvfile = open(filename, "a")
    fieldnames = ["Extractor", "Initialization", "Single"]
    for b in args.batch_sizes:
        fieldnames.append("Batch (%i)" % b)
    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    if write_header:
        writer.writeheader()

    if args.extractor is None:
        # Get all the available feature extractor names
        extractor_names = map(lambda e: e[0], inspect.getmembers(feature_extractor, inspect.isclass))
        extractor_names = filter(lambda f: f != "FeatureExtractorBase", extractor_names)
        args.extractor = extractor_names

    # Get an instance of each class
    module = __import__("feature_extractor")
    
    if len(args.extractor) > 1:
        csvfile.close()
        for extractor_name in tqdm(args.extractor, desc="Benchmarking extractors", file=sys.stderr):
            command = "/home/ldwg/anomaly_detector/.env/bin/python /home/ldwg/anomaly_detector/anomaly_detector/scripts/feature_extractor_benchmark.py --extractor %s --output %s" % (extractor_name, filename)
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            for line in process.stdout:
                tqdm.write(line)
            process.wait()
    else:
        extractor_name = args.extractor[0]
        result = {"Extractor": extractor_name.replace("FeatureExtractor", "")}

        logger.info("Benchmarking %s" % extractor_name)

        def log(s, times):
            """Log duration t with info string s"""
            logger.info("%-40s (%s): %.5fs  -  %.5fs" % (extractor_name, s, np.min(times), np.max(times)))
            result[s] = np.min(times)

        def logerr(s, err):
            """Log duration t with info string s"""
            logger.error("%-40s (%s): %s" % (extractor_name, s, err))
            result[s] = "-"

        try:
            _class = getattr(module, extractor_name)

            # Test extractor initialization
            try:
                log("Initialization", np.array(timeit.repeat(lambda: _class(), number=1, repeat=args.init_repeat)))
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                logerr("Initialization", traceback.format_exc())

            extractor = _class()
    
            # Load dataset
            if files[0].endswith(".tfrecord"):
                dataset = utils.load_tfrecords(files)
            elif files[0].endswith(".jpg"):
                dataset = utils.load_jpgs(files)
            else:
                raise ValueError("Supported file types are *.tfrecord and *.jpg")
            
            dataset = dataset.map(lambda image, time: (extractor.format_image(image), time),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

            # Test single image extraction
            try:
                single = list(dataset.take(1).as_numpy_iterator())[0] # Get a single entry
                times = np.array(timeit.repeat(lambda: extractor.extract(single[0]), number=1, repeat=args.extract_single_repeat))
                log("Single", times)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                logerr("Single", traceback.format_exc())
            
            # Test batch extraction
            for batch_size in args.batch_sizes:
                try:
                    batch = list(dataset.batch(batch_size).take(1).as_numpy_iterator())[0]
                    times = np.array(timeit.repeat(lambda: extractor.extract_batch(batch[0]), number=1, repeat=args.extract_batch_repeat))
                    times = times / float(batch_size)
                    log("Batch (%i)" % batch_size, times)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    logerr("Batch (%i)" % batch_size, traceback.format_exc())
        except KeyboardInterrupt:
            logger.info("Cancelled")
            raise
        except:
            logerr("Error?", traceback.format_exc())

        writer.writerow(result)
        csvfile.close()

if __name__ == "__main__":
    feature_extractor_benchmark()