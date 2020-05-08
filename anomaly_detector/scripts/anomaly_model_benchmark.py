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

import numpy as np
import csv
from tqdm import tqdm
from anomaly_model import AnomalyModelSVG, AnomalyModelBalancedDistribution, AnomalyModelBalancedDistributionSVG, AnomalyModelSpatialBinsBase

def anomaly_model_benchmark():
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
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), datetime.now().strftime("%Y_%m_%d_%H_%M_benchmark_anomaly_model.csv"))
    else:
        filename = args.output
    
    write_header = not os.path.exists(filename)

    with tqdm(total=len(files), file=sys.stderr, desc="Calculating SVG") as pbar:
        for features_file in files:
            # Load the file
            patches = PatchArray(features_file)
            m = AnomalyModelSVG()
            m.load_or_generate(patches, silent=True)
            pbar.update()
            
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
                
                models = [AnomalyModelSVG()]

                for fake in [True, False]:
                    for cell_size in [0.2, 0.5]:
                        models.append(AnomalyModelSpatialBinsBase(AnomalyModelSVG, cell_size=cell_size, fake=fake))
                        # models.append(AnomalyModelSpatialBinsBase(lambda: AnomalyModelBalancedDistributionSVG(initial_normal_features=10, threshold_learning=threshold_learning, pruning_parameter=0.5), cell_size=cell_size, fake=fake))

                # Calculate anomaly models
                if patches.contains_mahalanobis_distances and "SVG" in patches.mahalanobis_distances.dtype.names:
                    threshold_learning = int(np.mean(patches.mahalanobis_distances["SVG"]))
                    models.append(AnomalyModelBalancedDistributionSVG(initial_normal_features=20, threshold_learning=threshold_learning, pruning_parameter=0.5))

                with tqdm(total=len(models), file=sys.stderr) as pbar2:
                    for m in models:
                        try:
                            pbar2.set_description(m.NAME)
                            logger.info("Calculating %s" % m.NAME)
                            
                            log("%s (Creation)" % m.NAME, np.array(timeit.repeat(lambda: m.__generate_model__(patches, silent=True), number=1, repeat=3)))
                            
                            def _evaluate_frame():
                                for i in np.ndindex(patches.shape):
                                    m.__mahalanobis_distance__(patches[i])
                            
                            log("%s (Maha per patch)" % m.NAME, np.array(timeit.repeat(lambda: m.__mahalanobis_distance__(patches[0, 0, 0]), number=1, repeat=10)))
                            log("%s (Maha per frame)" % m.NAME, np.array(timeit.repeat(lambda: _evaluate_frame(), number=1, repeat=10)) / float(patches.shape[0]))

                        except (KeyboardInterrupt, SystemExit):
                            raise
                        except:
                            logger.error("%s: %s" % (features_file, traceback.format_exc()))
                        pbar2.update()
                
                if writer is None:
                    writer = csv.DictWriter(csvfile, fieldnames=result.keys())

                    if write_header:
                        writer.writeheader()


                writer.writerow(result)
                pbar.update()

if __name__ == "__main__":
    anomaly_model_benchmark()