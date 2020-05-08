#! /usr/bin/env python
# -*- coding: utf-8 -*-

import consts
import argparse

parser = argparse.ArgumentParser(description="Benchmark the specified anomaly models.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--files", metavar="F", dest="files", type=str, nargs='*', default=consts.FEATURES_PATH + "Benchmark/*.h5",
                    help="The feature file(s). Supports \"path/to/*.h5\"")

parser.add_argument("--output", metavar="OUT", dest="output", type=str,
                    help="Output file (default: \"\")")

args = parser.parse_args()

import os
import time
from common import utils, logger, PatchArray, ImageLocationUtility
import sys
from datetime import datetime
import inspect
import traceback
import timeit
from glob import glob

from joblib import Parallel, delayed

import numpy as np
import csv
from tqdm import tqdm

def rasterization_benchmark():
    ################
    #  Parameters  #
    ################
    files = args.files

    # Check parameters
    if not files or len(files) < 1 or files[0] == "":
        raise ValueError("Please specify at least one filename (%s)" % files)
    
    if isinstance(files, basestring):
        files = [files]
        
    # Expand wildcards
    files_expanded = []
    for s in files:
        files_expanded += glob(s)
    files = sorted(list(set(files_expanded))) # Remove duplicates

    if args.output is None:
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), datetime.now().strftime("%Y_%m_%d_%H_%M_benchmark_rasterization.csv"))
    else:
        filename = args.output
    
    write_header = not os.path.exists(filename)

    with open(filename, "a") as csvfile:
        writer = None
        with tqdm(total=len(files), file=sys.stderr, desc="Benchmarking anomaly models") as pbar:
            for features_file in files:
                extractor_name = os.path.basename(features_file).replace(".h5", "")

                result = {"Extractor": extractor_name.replace("FeatureExtractor", "")}

                def log(s, times):
                    """Log duration t with info string s"""
                    logger.info("%-40s (%s): %.5fs  -  %.5fs" % (extractor_name, s, np.min(times), np.max(times)))
                    result[s] = np.min(times)
    
                pbar.set_description(os.path.basename(features_file))
                # Check parameters
                if features_file == "" or not os.path.exists(features_file) or not os.path.isfile(features_file):
                    logger.error("Specified feature file does not exist (%s)" % features_file)
                    continue


                # Load the file
                patches = PatchArray(features_file)
                
                for fake in [True, False]:
                    #####################
                    #  RECEPTIVE FIELD  #
                    #####################
                    key = "locations"
                    if fake: key = "fake_" + key

                    start = time.time()
                    image_locations = patches._get_receptive_fields(fake)
                    
                    relative_locations = patches._image_to_relative(image_locations)
                    
                    for i in tqdm(range(patches[key].shape[0]), desc="Calculating locations", file=sys.stderr):
                        patches[key][i] = patches._relative_to_absolute(relative_locations, patches[i, 0, 0].camera_locations)

                    end = time.time()

                    patches.contains_locations = True

                    patches._save_patch_locations(fake, start, end)

                    patches.contains_locations = True

                    # Time individual blocks
                    log("RF (img) [f: %s]" % fake,      np.array(timeit.repeat(lambda: patches._get_receptive_fields(fake=fake), number=1, repeat=5)))
                    log("RF --> rel [f: %s]" % fake, np.array(timeit.repeat(lambda: patches._image_to_relative(image_locations), number=1, repeat=5)))
                    log("RF --> abs [f: %s]" % fake, np.array(timeit.repeat(lambda: patches._relative_to_absolute(relative_locations, patches[0, 0, 0].camera_locations), number=1, repeat=10)))

                    #####################
                    #   RASTERIZATION   #
                    #####################
                    for cell_size in [0.2, 0.5]:
                        key = "%.2f" % cell_size
                        if fake: key = "fake_" + key

                        grid, shape = patches._calculate_grid(cell_size, fake=fake)

                        rf_factor = patches.receptive_field[0] / patches.image_size

                        logger.info("%i bins in x and %i bins in y direction (with cell size %.2f)" % (shape + (cell_size,)))

                        start = time.time()
                        
                        # Get the corresponding bin for every feature
                        Parallel(n_jobs=2, prefer="threads")(
                            delayed(patches._bin)(i, grid, shape, rf_factor, key, fake) for i in tqdm(range(patches.shape[0]), desc="Calculating bins", file=sys.stderr))

                        end = time.time()

                        patches._save_rasterization(key, grid, shape, start, end)
                        
                        patches.contains_bins[key] = True
                        patches.rasterizations[key] = np.vectorize(lambda b: b.patches, otypes=[object])(patches.rasterizations[key])

                        # Time individual blocks
                        log("Grid [%.2f, f: %s]" % (cell_size, fake), np.array(timeit.repeat(lambda: patches._calculate_grid(cell_size, fake=fake), number=1, repeat=3)))
                        log("Bins [%.2f, f: %s]" % (cell_size, fake), np.array(timeit.repeat(lambda: patches._bin(0, grid, shape, rf_factor, key, fake), number=1, repeat=3)))

                if writer is None:
                    writer = csv.DictWriter(csvfile, fieldnames=result.keys())

                    if write_header:
                        writer.writeheader()


                writer.writerow(result)
                pbar.update()

if __name__ == "__main__":
    rasterization_benchmark()