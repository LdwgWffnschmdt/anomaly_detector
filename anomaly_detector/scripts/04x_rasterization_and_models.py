#! /usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################################################
# This script calculates                                                                        #
#   - patch locations                                                                           #
#   - rasterizations                                                                            #
#   - models + mahalanobis distances.                                                           #
# These calculations - especially rasterization and mahalanobis distances - are very slow       #
# and only seem to use one core.                                                                #
# You can call *04_rasterization_and_models_parallel.sh* instead for to run multiple            #
# instances of this script, which will then utilize more CPU cores. But beware heyvy RAM use!   #
#################################################################################################


import consts
import argparse

parser = argparse.ArgumentParser(description="Add patch locations to feature files.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--files", metavar="F", dest="files", type=str, nargs='*', default=consts.FEATURES_FILES,
                    help="The feature file(s). Supports \"path/to/*.h5\"")

parser.add_argument("--index", metavar="I", dest="index", type=int, default=None,
                    help="")
parser.add_argument("--total", metavar="T", dest="total", type=int, default=None,
                    help="")

args = parser.parse_args()

import os
import sys
import time
import traceback
from glob import glob

from tqdm import tqdm
import numpy as np

from common import utils, logger, PatchArray
from anomaly_model import AnomalyModelSVG, AnomalyModelBalancedDistribution, AnomalyModelBalancedDistributionSVG, AnomalyModelSpatialBinsBase

def calculate_locations():
    ################
    #  Parameters  #
    ################
    files       = args.files

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

    files = filter(lambda f: not "EfficientNet" in f, files)

    if args.index is not None:
        files = files[args.index::args.total]

    with tqdm(total=len(files), file=sys.stderr) as pbar:
        for features_file in files:
            pbar.set_description(os.path.basename(features_file))
            # Check parameters
            if features_file == "" or not os.path.exists(features_file) or not os.path.isfile(features_file):
                logger.error("Specified feature file does not exist (%s)" % features_file)
                continue

            try:
                # Load the file
                patches = PatchArray(features_file)

                models = [AnomalyModelSVG()]

                # Calculate and save the locations
                for fake in [False]:
                    patches.calculate_patch_locations(fake=fake)
                    for cell_size in [0.2, 0.5]:
                        patches.calculate_rasterization(cell_size, fake=fake)

                        models.append(AnomalyModelSpatialBinsBase(AnomalyModelSVG, cell_size=cell_size, fake=fake))
                        # models.append(AnomalyModelSpatialBinsBase(lambda: AnomalyModelBalancedDistributionSVG(initial_normal_features=10, threshold_learning=threshold_learning, pruning_parameter=0.5), cell_size=cell_size, fake=fake))

                # Calculate anomaly models
                if patches.contains_mahalanobis_distances and "SVG" in patches.mahalanobis_distances.dtype.names:
                    threshold_learning = int(np.mean(patches.mahalanobis_distances["SVG"]))
                    models.append(AnomalyModelBalancedDistributionSVG(initial_normal_features=500, threshold_learning=threshold_learning, pruning_parameter=0.5))

                with tqdm(total=len(models), file=sys.stderr) as pbar2:
                    for m in models:
                        try:
                            pbar2.set_description(m.NAME)
                            logger.info("Calculating %s" % m.NAME)
                            
                            model, mdist = m.is_in_file(features_file)

                            if not model:
                                m.load_or_generate(patches, silent=True)
                            elif not mdist:
                                logger.info("Model already calculated")
                                m.load_from_file(features_file)
                                m.patches = patches
                                m.calculate_mahalanobis_distances()
                            else:
                                logger.info("Model and mahalanobis distances already calculated")

                        except (KeyboardInterrupt, SystemExit):
                            raise
                        except:
                            logger.error("%s: %s" % (features_file, traceback.format_exc()))
                        pbar2.update()

            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                logger.error("%s: %s" % (features_file, traceback.format_exc()))
            pbar.update()

if __name__ == "__main__":
    calculate_locations()
    pass