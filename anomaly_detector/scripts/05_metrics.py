#! /usr/bin/env python
# -*- coding: utf-8 -*-

import consts
import argparse

parser = argparse.ArgumentParser(description="Calculate metrics for the specified anomaly models.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--files", metavar="F", dest="files", type=str, nargs='*', default=consts.FEATURES_FILES,
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

def metrics():
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

    # files = filter(lambda f: "EfficientNetB6" in f, files)
    # files = filter(lambda f: "EfficientNetB6_Level6" not in f, files)

    if args.output is None:
        filename = os.path.join(consts.METRICS_PATH, datetime.now().strftime("%Y_%m_%d_%H_%M_metrics.csv"))
    else:
        filename = args.output

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    write_header = not os.path.exists(filename)

    with open(filename, "a") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "Extractor",
            "Measure",
            "Model",
            "Gaussian filter",
            "Other filter",
            "ROC_AUC",
            "AUC_PR",
            "Max. f1",
            "FPR at TPR=0.9",
            "FPR at TPR=0.95",
            "FPR at TPR=0.99",
            "FPR at TPR=0.995",
            "FPR at TPR=0.999",
            "FPR at TPR=0.9999"
        ])

        if write_header:
            writer.writeheader()
        
        with tqdm(total=len(files), file=sys.stderr, desc="Calculating metrics") as pbar:
            for features_file in files:

                pbar.set_description(os.path.basename(features_file))
                # Check parameters
                if features_file == "" or not os.path.exists(features_file) or not os.path.isfile(features_file):
                    logger.error("Specified feature file does not exist (%s)" % features_file)
                    continue

                # Load the file
                patches = PatchArray(features_file)
                
                patches.calculate_patch_labels()
                
                res = patches.calculate_metrics()

                for extractor, measure, model, gauss_filter, other_filter, roc_auc, auc_pr, max_f1, fpr0, fpr1, fpr2, fpr3, fpr4, fpr5 in res:
                    writer.writerow({
                        "Extractor": extractor,
                        "Measure": measure,
                        "Model": model,
                        "Gaussian filter": gauss_filter,
                        "Other filter": other_filter,
                        "ROC_AUC": roc_auc,
                        "AUC_PR": auc_pr,
                        "Max. f1": max_f1,
                        "FPR at TPR=0.9": fpr0,
                        "FPR at TPR=0.95": fpr1,
                        "FPR at TPR=0.99": fpr2,
                        "FPR at TPR=0.995": fpr3,
                        "FPR at TPR=0.999": fpr4,
                        "FPR at TPR=0.9999": fpr5
                    })

                pbar.update()

if __name__ == "__main__":
    metrics()