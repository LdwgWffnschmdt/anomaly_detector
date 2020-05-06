#! /usr/bin/env python
# -*- coding: utf-8 -*-

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

    # files = filter(lambda f: f.endswith("C3D.h5"), files)

    if args.index is not None:
        files = files[args.index::args.total]

    with tqdm(total=len(files), file=sys.stderr) as pbar:
        metrics = list()
        for features_file in files:
            pbar.set_description(os.path.basename(features_file))
            # Check parameters
            if features_file == "" or not os.path.exists(features_file) or not os.path.isfile(features_file):
                logger.error("Specified feature file does not exist (%s)" % features_file)
                continue

            try:
                # Load the file
                patches = PatchArray(features_file)

                # # patches.calculate_patch_labels()

                # # Calculate and save the locations
                # if patches.contains_locations:
                #     logger.info("Locations already calculated")
                # else:
                #     patches.calculate_patch_locations()

                # patches.calculate_rasterization(0.2)
                # patches.calculate_rasterization(0.5)

                # # Calculate anomaly models

                # models = [
                #     # AnomalyModelSVG(),
                #     AnomalyModelSpatialBinsBase(AnomalyModelSVG, cell_size=0.2),
                #     AnomalyModelSpatialBinsBase(AnomalyModelSVG, cell_size=0.5)
                #     # AnomalyModelSpatialBinsBase(lambda: AnomalyModelBalancedDistributionSVG(initial_normal_features=10, threshold_learning=threshold_learning, pruning_parameter=0.5), cell_size=0.5)
                # ]

                # if patches.contains_mahalanobis_distances and "SVG" in patches.mahalanobis_distances.dtype.names:
                #     threshold_learning = int(np.mean(patches.mahalanobis_distances["SVG"]))
                #     models.append(AnomalyModelBalancedDistributionSVG(initial_normal_features=500, threshold_learning=threshold_learning, pruning_parameter=0.5))

                # with tqdm(total=len(models), file=sys.stderr) as pbar2:
                #     for m in models:
                #         try:
                #             pbar2.set_description(m.NAME)
                #             logger.info("Calculating %s" % m.NAME)
                            
                #             model, mdist = m.is_in_file(features_file)

                #             if not model:
                #                 m.load_or_generate(patches, silent=True)
                #             elif not mdist:
                #                 logger.info("Model already calculated")
                #                 m.load_from_file(features_file)
                #                 m.patches = patches
                #                 m.calculate_mahalanobis_distances()
                #             else:
                #                 logger.info("Model and mahalanobis distances already calculated")

                #         except (KeyboardInterrupt, SystemExit):
                #             raise
                #         except:
                #             logger.error("%s: %s" % (features_file, traceback.format_exc()))
                #         pbar2.update()
                
                if patches.contains_mahalanobis_distances:
                    # patches.calculate_tsne()
                    metrics.extend(patches.calculate_metrics())

            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                logger.error("%s: %s" % (features_file, traceback.format_exc()))
            pbar.update()


        print("%-30s | %-15s | %-40s | %-10s | %-10s | %-10s | %-10s" % ("EXTRACTOR", "MEASURE", "MODEL", "FILTER", "ROC_AUC", "AUC_PR", "Max. f1"))
        print("-" * 100)
        for m in metrics:
            print("%-30s | %-15s | %-40s | %-10s | %.8f | %.8f | %.8f" % m)

if __name__ == "__main__":
    calculate_locations()
    pass


# EXTRACTOR                      | MEASURE         | MODEL                                    | FILTER     | ROC_AUC    | AUC_PR     | Max. f1   
# -------------------------------|-----------------|------------------------------------------|------------|------------|------------|-----------
# EfficientNetB0                 | per patch       | SpatialBin/SVG/0.20                      | None       | 0.66596601 | 0.11299871 | 0.27527613
# EfficientNetB0                 | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.70364329 | 0.13515349 | 0.29105691
# EfficientNetB0                 | per patch       | SVG                                      | None       | 0.66187937 | 0.11215289 | 0.27357689
# EfficientNetB0                 | per patch       | SpatialBin/SVG/0.50                      | None       | 0.66584282 | 0.11299436 | 0.27551020
# EfficientNetB0                 | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.69931229 | 0.13475733 | 0.29225908
# EfficientNetB0                 | per patch       | BalancedDistributionSVG/500/40/0.30      | None       | 0.69369169 | 0.11985114 | 0.27950311
# EfficientNetB0                 | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.71872275 | 0.13728940 | 0.27794872
# EfficientNetB0                 | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.73684724 | 0.16775549 | 0.28629032
# EfficientNetB0                 | per patch       | SVG                                      | (1, 1, 1)  | 0.71433430 | 0.13737867 | 0.27628238
# EfficientNetB0                 | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.71872436 | 0.13734116 | 0.27809133
# EfficientNetB0                 | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.73918808 | 0.15311765 | 0.29009193
# EfficientNetB0                 | per patch       | BalancedDistributionSVG/500/40/0.30      | (1, 1, 1)  | 0.74586141 | 0.14127968 | 0.28070175
# EfficientNetB0                 | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.71974896 | 0.14223547 | 0.26282051
# EfficientNetB0                 | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.74303183 | 0.19458616 | 0.30751708
# EfficientNetB0                 | per patch       | SVG                                      | (2, 2, 2)  | 0.71351724 | 0.14217881 | 0.26300851
# EfficientNetB0                 | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.71980252 | 0.14228762 | 0.26270023
# EfficientNetB0                 | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.74178066 | 0.15228423 | 0.29865772
# EfficientNetB0                 | per patch       | BalancedDistributionSVG/500/40/0.30      | (2, 2, 2)  | 0.73033352 | 0.14179640 | 0.26472019
# EfficientNetB0                 | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.67313986 | 0.12309312 | 0.29635258
# EfficientNetB0                 | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.71190481 | 0.15139391 | 0.31123517
# EfficientNetB0                 | per patch       | SVG                                      | (0, 1, 1)  | 0.66567116 | 0.12138199 | 0.29460896
# EfficientNetB0                 | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.67289321 | 0.12305456 | 0.29635258
# EfficientNetB0                 | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.71813386 | 0.14454098 | 0.32104121
# EfficientNetB0                 | per patch       | BalancedDistributionSVG/500/40/0.30      | (0, 1, 1)  | 0.70208777 | 0.12726505 | 0.30339321
# EfficientNetB0                 | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.69565574 | 0.12666598 | 0.31206657
# EfficientNetB0                 | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.73699908 | 0.16831351 | 0.32595870
# EfficientNetB0                 | per patch       | SVG                                      | (0, 2, 2)  | 0.68518419 | 0.12336757 | 0.31105048
# EfficientNetB0                 | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.69523154 | 0.12661720 | 0.31163435
# EfficientNetB0                 | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.74589703 | 0.15239677 | 0.34113208
# EfficientNetB0                 | per patch       | BalancedDistributionSVG/500/40/0.30      | (0, 2, 2)  | 0.71645207 | 0.12610942 | 0.30950729
# EfficientNetB0                 | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.71881354 | 0.13230619 | 0.25560267
# EfficientNetB0                 | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.74043282 | 0.15158099 | 0.27781048
# EfficientNetB0                 | per patch       | SVG                                      | (1, 0, 0)  | 0.71611598 | 0.13218700 | 0.25469982
# EfficientNetB0                 | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.71882318 | 0.13235068 | 0.25560267
# EfficientNetB0                 | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.73320354 | 0.14923538 | 0.27532777
# EfficientNetB0                 | per patch       | BalancedDistributionSVG/500/40/0.30      | (1, 0, 0)  | 0.74513219 | 0.13837636 | 0.26984951
# EfficientNetB0                 | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.71793756 | 0.13712704 | 0.22969188
# EfficientNetB0                 | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.74194027 | 0.16086476 | 0.25950783
# EfficientNetB0                 | per patch       | SVG                                      | (2, 0, 0)  | 0.71578417 | 0.13650915 | 0.22986037
# EfficientNetB0                 | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.71797532 | 0.13719189 | 0.22969188
# EfficientNetB0                 | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.73212752 | 0.15470591 | 0.25983667
# EfficientNetB0                 | per patch       | BalancedDistributionSVG/500/40/0.30      | (2, 0, 0)  | 0.74268743 | 0.14232119 | 0.23568702
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.54485905 | 0.56104557 | 0.61940299
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.54099895 | 0.52172116 | 0.61710037
# EfficientNetB0                 | per frame (max) | SVG                                      | None       | 0.52017780 | 0.54017590 | 0.61940299
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.54497602 | 0.56071725 | 0.61940299
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.50157913 | 0.49294220 | 0.61710037
# EfficientNetB0                 | per frame (max) | BalancedDistributionSVG/500/40/0.30      | None       | 0.55105860 | 0.55263327 | 0.61940299
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.56825360 | 0.60022987 | 0.61710037
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.57679261 | 0.56529350 | 0.61710037
# EfficientNetB0                 | per frame (max) | SVG                                      | (1, 1, 1)  | 0.53398058 | 0.58458779 | 0.61710037
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.56731781 | 0.59984086 | 0.61710037
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.53842555 | 0.50651844 | 0.61710037
# EfficientNetB0                 | per frame (max) | BalancedDistributionSVG/500/40/0.30      | (1, 1, 1)  | 0.58346005 | 0.60822897 | 0.61710037
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.55304714 | 0.60208173 | 0.61940299
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.58930869 | 0.60430045 | 0.61710037
# EfficientNetB0                 | per frame (max) | SVG                                      | (2, 2, 2)  | 0.51128787 | 0.58884856 | 0.61710037
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.55187741 | 0.60169793 | 0.61940299
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.51526494 | 0.49490902 | 0.61710037
# EfficientNetB0                 | per frame (max) | BalancedDistributionSVG/500/40/0.30      | (2, 2, 2)  | 0.51795532 | 0.59366589 | 0.62172285
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.54626272 | 0.55999049 | 0.61710037
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.55222833 | 0.54791617 | 0.61710037
# EfficientNetB0                 | per frame (max) | SVG                                      | (0, 1, 1)  | 0.51865715 | 0.54704382 | 0.61940299
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.54544391 | 0.55973479 | 0.61710037
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.53409756 | 0.51708741 | 0.61710037
# EfficientNetB0                 | per frame (max) | BalancedDistributionSVG/500/40/0.30      | (0, 1, 1)  | 0.58884080 | 0.57875623 | 0.61710037
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.58310914 | 0.58439877 | 0.64341085
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.60439818 | 0.57686839 | 0.63000000
# EfficientNetB0                 | per frame (max) | SVG                                      | (0, 2, 2)  | 0.53491636 | 0.55834286 | 0.62406015
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.58322611 | 0.58404821 | 0.64591440
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.58533162 | 0.55394100 | 0.62357414
# EfficientNetB0                 | per frame (max) | BalancedDistributionSVG/500/40/0.30      | (0, 2, 2)  | 0.55468476 | 0.56967995 | 0.62641509
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.57445315 | 0.59467989 | 0.61710037
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.56146918 | 0.52489763 | 0.61710037
# EfficientNetB0                 | per frame (max) | SVG                                      | (1, 0, 0)  | 0.54298748 | 0.58325293 | 0.61710037
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.57281553 | 0.59371053 | 0.61710037
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.51795532 | 0.48642122 | 0.61710037
# EfficientNetB0                 | per frame (max) | BalancedDistributionSVG/500/40/0.30      | (1, 0, 0)  | 0.55012282 | 0.59114158 | 0.61710037
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.56146918 | 0.59278675 | 0.61710037
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.55667330 | 0.52941064 | 0.61710037
# EfficientNetB0                 | per frame (max) | SVG                                      | (2, 0, 0)  | 0.52251725 | 0.57936022 | 0.61710037
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.56100129 | 0.59366520 | 0.61710037
# EfficientNetB0                 | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.49666628 | 0.45900565 | 0.61710037
# EfficientNetB0                 | per frame (max) | BalancedDistributionSVG/500/40/0.30      | (2, 0, 0)  | 0.50298281 | 0.57781960 | 0.61710037
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.58930869 | 0.58831839 | 0.62172285
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.62884548 | 0.59970266 | 0.64000000
# EfficientNetB0                 | per frame (sum) | SVG                                      | None       | 0.52310212 | 0.54556448 | 0.61940299
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.58743713 | 0.58748566 | 0.62172285
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.59036145 | 0.57212309 | 0.62172285
# EfficientNetB0                 | per frame (sum) | BalancedDistributionSVG/500/40/0.30      | None       | 0.53327875 | 0.55197290 | 0.61940299
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.58942566 | 0.60844230 | 0.62172285
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.61305416 | 0.61782465 | 0.63565891
# EfficientNetB0                 | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.53468242 | 0.58615339 | 0.62172285
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.58732015 | 0.60719346 | 0.62172285
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.58112060 | 0.56745647 | 0.62121212
# EfficientNetB0                 | per frame (sum) | BalancedDistributionSVG/500/40/0.30      | (1, 1, 1)  | 0.53210902 | 0.58599745 | 0.62641509
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.54556088 | 0.59956358 | 0.61710037
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.58498070 | 0.61938436 | 0.62698413
# EfficientNetB0                 | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.50988420 | 0.58824008 | 0.61710037
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.54439116 | 0.59925822 | 0.61710037
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.54216867 | 0.55290351 | 0.62172285
# EfficientNetB0                 | per frame (sum) | BalancedDistributionSVG/500/40/0.30      | (2, 2, 2)  | 0.50789566 | 0.58951920 | 0.62172285
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.58930869 | 0.58831839 | 0.62172285
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.62884548 | 0.59970266 | 0.64000000
# EfficientNetB0                 | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.52310212 | 0.54556448 | 0.61940299
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.58743713 | 0.58748566 | 0.62172285
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.59036145 | 0.57212309 | 0.62172285
# EfficientNetB0                 | per frame (sum) | BalancedDistributionSVG/500/40/0.30      | (0, 1, 1)  | 0.53327875 | 0.55197290 | 0.61940299
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.58930869 | 0.58831839 | 0.62172285
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.62884548 | 0.59970266 | 0.64000000
# EfficientNetB0                 | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.52310212 | 0.54556448 | 0.61940299
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.58743713 | 0.58748566 | 0.62172285
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.59036145 | 0.57212309 | 0.62172285
# EfficientNetB0                 | per frame (sum) | BalancedDistributionSVG/500/40/0.30      | (0, 2, 2)  | 0.53327875 | 0.55197290 | 0.61940299
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.58942566 | 0.60844230 | 0.62172285
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.61305416 | 0.61782465 | 0.63565891
# EfficientNetB0                 | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.53468242 | 0.58615339 | 0.62172285
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.58732015 | 0.60719346 | 0.62172285
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.58112060 | 0.56745647 | 0.62121212
# EfficientNetB0                 | per frame (sum) | BalancedDistributionSVG/500/40/0.30      | (1, 0, 0)  | 0.53210902 | 0.58599745 | 0.62641509
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.54556088 | 0.59956358 | 0.61710037
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.58498070 | 0.61938436 | 0.62698413
# EfficientNetB0                 | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.50988420 | 0.58824008 | 0.61710037
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.54439116 | 0.59925822 | 0.61710037
# EfficientNetB0                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.54216867 | 0.55290351 | 0.62172285
# EfficientNetB0                 | per frame (sum) | BalancedDistributionSVG/500/40/0.30      | (2, 0, 0)  | 0.50789566 | 0.58951920 | 0.62172285
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/0.20                      | None       | 0.65785486 | 0.11018277 | 0.26118721
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.72628290 | 0.16660272 | 0.28989977
# EfficientNetB0_Block6          | per patch       | SVG                                      | None       | 0.65185988 | 0.10889292 | 0.25915996
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/0.50                      | None       | 0.65757126 | 0.11014516 | 0.26118721
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.72429193 | 0.16664420 | 0.29352069
# EfficientNetB0_Block6          | per patch       | BalancedDistributionSVG/500/14/0.30      | None       | 0.66842710 | 0.12560966 | 0.27502034
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.72336347 | 0.15054889 | 0.27476415
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.76233437 | 0.16736544 | 0.29041096
# EfficientNetB0_Block6          | per patch       | SVG                                      | (1, 1, 1)  | 0.71698205 | 0.14990939 | 0.27362319
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.72317092 | 0.15055355 | 0.27476415
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.77704600 | 0.18556569 | 0.32302937
# EfficientNetB0_Block6          | per patch       | BalancedDistributionSVG/500/14/0.30      | (1, 1, 1)  | 0.71248192 | 0.15843642 | 0.27717671
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.71846191 | 0.16136364 | 0.28386454
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.75744941 | 0.16451607 | 0.30207588
# EfficientNetB0_Block6          | per patch       | SVG                                      | (2, 2, 2)  | 0.71286140 | 0.16087897 | 0.28458106
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.71834381 | 0.16132305 | 0.28400598
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.77541027 | 0.17113544 | 0.33146067
# EfficientNetB0_Block6          | per patch       | BalancedDistributionSVG/500/14/0.30      | (2, 2, 2)  | 0.71273392 | 0.16731550 | 0.28970512
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.67276815 | 0.12455895 | 0.27581330
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.74373614 | 0.16790906 | 0.29371688
# EfficientNetB0_Block6          | per patch       | SVG                                      | (0, 1, 1)  | 0.66364123 | 0.12231987 | 0.26128591
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.67216238 | 0.12446367 | 0.27581330
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.75810499 | 0.18102432 | 0.31974922
# EfficientNetB0_Block6          | per patch       | BalancedDistributionSVG/500/14/0.30      | (0, 1, 1)  | 0.65012908 | 0.12821083 | 0.29590489
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.65882618 | 0.13176137 | 0.30068027
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.74403876 | 0.16762014 | 0.30635118
# EfficientNetB0_Block6          | per patch       | SVG                                      | (0, 2, 2)  | 0.64921642 | 0.13019315 | 0.29457364
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.65808571 | 0.13170601 | 0.30047587
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.76999400 | 0.17835576 | 0.32561852
# EfficientNetB0_Block6          | per patch       | BalancedDistributionSVG/500/14/0.30      | (0, 2, 2)  | 0.64451865 | 0.13164373 | 0.29820051
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.70827129 | 0.13283993 | 0.26424051
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.76487258 | 0.17723719 | 0.28847584
# EfficientNetB0_Block6          | per patch       | SVG                                      | (1, 0, 0)  | 0.70375858 | 0.13216437 | 0.26507937
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.70812025 | 0.13284178 | 0.26424051
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.75855409 | 0.18067118 | 0.29223744
# EfficientNetB0_Block6          | per patch       | BalancedDistributionSVG/500/14/0.30      | (1, 0, 0)  | 0.72298587 | 0.15126267 | 0.28074385
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.70814944 | 0.12296935 | 0.24080268
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.76506325 | 0.17202180 | 0.29205807
# EfficientNetB0_Block6          | per patch       | SVG                                      | (2, 0, 0)  | 0.70470901 | 0.12188722 | 0.23799865
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.70806776 | 0.12294620 | 0.24080268
# EfficientNetB0_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.76155507 | 0.18120655 | 0.30557940
# EfficientNetB0_Block6          | per patch       | BalancedDistributionSVG/500/14/0.30      | (2, 0, 0)  | 0.72917769 | 0.15423290 | 0.26323120
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.77775178 | 0.70891829 | 0.72631579
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.57304948 | 0.47970065 | 0.65060241
# EfficientNetB0_Block6          | per frame (max) | SVG                                      | None       | 0.67037080 | 0.62711176 | 0.65048544
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.77985729 | 0.71062001 | 0.72826087
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.56778571 | 0.50096547 | 0.65873016
# EfficientNetB0_Block6          | per frame (max) | BalancedDistributionSVG/500/14/0.30      | None       | 0.55971459 | 0.55281807 | 0.61710037
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.67352907 | 0.65631774 | 0.65437788
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.55994853 | 0.48141560 | 0.61710037
# EfficientNetB0_Block6          | per frame (max) | SVG                                      | (1, 1, 1)  | 0.61024681 | 0.62151082 | 0.62172285
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.67329512 | 0.65616017 | 0.65137615
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.54298748 | 0.48183322 | 0.61940299
# EfficientNetB0_Block6          | per frame (max) | BalancedDistributionSVG/500/14/0.30      | (1, 1, 1)  | 0.60860919 | 0.62163443 | 0.62172285
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.57609077 | 0.62006669 | 0.62500000
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.55643935 | 0.49358244 | 0.61710037
# EfficientNetB0_Block6          | per frame (max) | SVG                                      | (2, 2, 2)  | 0.54556088 | 0.60986509 | 0.61710037
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.57503802 | 0.61953626 | 0.62400000
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.50766171 | 0.46053171 | 0.61710037
# EfficientNetB0_Block6          | per frame (max) | BalancedDistributionSVG/500/14/0.30      | (2, 2, 2)  | 0.54263657 | 0.60755594 | 0.61940299
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.71797871 | 0.64989196 | 0.69607843
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.56731781 | 0.49320653 | 0.62280702
# EfficientNetB0_Block6          | per frame (max) | SVG                                      | (0, 1, 1)  | 0.64931571 | 0.60792008 | 0.63681592
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.71493742 | 0.64798107 | 0.68656716
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.57176278 | 0.52193177 | 0.62068966
# EfficientNetB0_Block6          | per frame (max) | BalancedDistributionSVG/500/14/0.30      | (0, 1, 1)  | 0.64182945 | 0.59869996 | 0.64646465
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.69446719 | 0.64571450 | 0.70000000
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.58240730 | 0.51158166 | 0.62555066
# EfficientNetB0_Block6          | per frame (max) | SVG                                      | (0, 2, 2)  | 0.63434320 | 0.59838052 | 0.64583333
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.69119195 | 0.64423964 | 0.69346734
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.58076968 | 0.54379564 | 0.62393162
# EfficientNetB0_Block6          | per frame (max) | BalancedDistributionSVG/500/14/0.30      | (0, 2, 2)  | 0.63925605 | 0.60340933 | 0.65306122
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.74570125 | 0.69783174 | 0.71276596
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.56766873 | 0.46962465 | 0.63967611
# EfficientNetB0_Block6          | per frame (max) | SVG                                      | (1, 0, 0)  | 0.65130425 | 0.63876240 | 0.65306122
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.74254299 | 0.69602086 | 0.70157068
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.53035443 | 0.46607729 | 0.64197531
# EfficientNetB0_Block6          | per frame (max) | BalancedDistributionSVG/500/14/0.30      | (1, 0, 0)  | 0.54661364 | 0.59799511 | 0.61710037
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.68663002 | 0.65937042 | 0.67980296
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.55281319 | 0.46392952 | 0.63076923
# EfficientNetB0_Block6          | per frame (max) | SVG                                      | (2, 0, 0)  | 0.59375366 | 0.61900118 | 0.62745098
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.68581121 | 0.65909789 | 0.67980296
# EfficientNetB0_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.48368230 | 0.42629905 | 0.63076923
# EfficientNetB0_Block6          | per frame (max) | BalancedDistributionSVG/500/14/0.30      | (2, 0, 0)  | 0.51549889 | 0.59705008 | 0.61710037
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.63867119 | 0.62035110 | 0.61818182
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.63446017 | 0.56582512 | 0.63565891
# EfficientNetB0_Block6          | per frame (sum) | SVG                                      | None       | 0.57901509 | 0.56887794 | 0.61710037
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.63270558 | 0.61758711 | 0.61710037
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.62837759 | 0.59487817 | 0.62595420
# EfficientNetB0_Block6          | per frame (sum) | BalancedDistributionSVG/500/14/0.30      | None       | 0.59071236 | 0.57456854 | 0.61776062
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.61153351 | 0.63059219 | 0.61710037
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.62369868 | 0.57193942 | 0.63779528
# EfficientNetB0_Block6          | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.56801965 | 0.60769649 | 0.61710037
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.60907709 | 0.62947709 | 0.61847390
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.60463212 | 0.56813849 | 0.62406015
# EfficientNetB0_Block6          | per frame (sum) | BalancedDistributionSVG/500/14/0.30      | (1, 1, 1)  | 0.57889812 | 0.61063806 | 0.62121212
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.56497836 | 0.62026870 | 0.62015504
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.59819862 | 0.56356898 | 0.63601533
# EfficientNetB0_Block6          | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.52836589 | 0.60537787 | 0.61710037
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.56345771 | 0.61969529 | 0.61776062
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.55456779 | 0.52308198 | 0.62172285
# EfficientNetB0_Block6          | per frame (sum) | BalancedDistributionSVG/500/14/0.30      | (2, 2, 2)  | 0.54123289 | 0.60727101 | 0.62172285
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.63867119 | 0.62035110 | 0.61818182
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.63446017 | 0.56582512 | 0.63565891
# EfficientNetB0_Block6          | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.57901509 | 0.56887794 | 0.61710037
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.63270558 | 0.61758711 | 0.61710037
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.62837759 | 0.59487817 | 0.62595420
# EfficientNetB0_Block6          | per frame (sum) | BalancedDistributionSVG/500/14/0.30      | (0, 1, 1)  | 0.59071236 | 0.57456854 | 0.61776062
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.63867119 | 0.62035110 | 0.61818182
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.63446017 | 0.56582512 | 0.63565891
# EfficientNetB0_Block6          | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.57901509 | 0.56887794 | 0.61710037
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.63270558 | 0.61758711 | 0.61710037
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.62837759 | 0.59487817 | 0.62595420
# EfficientNetB0_Block6          | per frame (sum) | BalancedDistributionSVG/500/14/0.30      | (0, 2, 2)  | 0.59071236 | 0.57456854 | 0.61776062
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.61153351 | 0.63059219 | 0.61710037
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.62369868 | 0.57193942 | 0.63779528
# EfficientNetB0_Block6          | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.56801965 | 0.60769649 | 0.61710037
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.60907709 | 0.62947709 | 0.61847390
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.60463212 | 0.56813849 | 0.62406015
# EfficientNetB0_Block6          | per frame (sum) | BalancedDistributionSVG/500/14/0.30      | (1, 0, 0)  | 0.57889812 | 0.61063806 | 0.62121212
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.56497836 | 0.62026870 | 0.62015504
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.59819862 | 0.56356898 | 0.63601533
# EfficientNetB0_Block6          | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.52836589 | 0.60537787 | 0.61710037
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.56345771 | 0.61969529 | 0.61776062
# EfficientNetB0_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.55456779 | 0.52308198 | 0.62172285
# EfficientNetB0_Block6          | per frame (sum) | BalancedDistributionSVG/500/14/0.30      | (2, 0, 0)  | 0.54123289 | 0.60727101 | 0.62172285
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/0.20                      | None       | 0.59592510 | 0.07161622 | 0.14328796
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.58980512 | 0.06333959 | 0.11772330
# EfficientNetB3_Block5          | per patch       | SVG                                      | None       | 0.59538475 | 0.07418073 | 0.14475500
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/0.50                      | None       | 0.59604923 | 0.07276633 | 0.14320507
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.58911868 | 0.06369531 | 0.12087412
# EfficientNetB3_Block5          | per patch       | BalancedDistributionSVG/500/9/0.30       | None       | 0.60737364 | 0.07334659 | 0.14463019
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.60605416 | 0.07338804 | 0.14519378
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.59915614 | 0.06507753 | 0.13193295
# EfficientNetB3_Block5          | per patch       | SVG                                      | (1, 1, 1)  | 0.60571478 | 0.07540409 | 0.14669367
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.60653405 | 0.07440842 | 0.14555386
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.59315152 | 0.06397417 | 0.12724983
# EfficientNetB3_Block5          | per patch       | BalancedDistributionSVG/500/9/0.30       | (1, 1, 1)  | 0.65933743 | 0.08285205 | 0.15007780
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.61768822 | 0.07683827 | 0.14803015
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.59815571 | 0.06449570 | 0.13057304
# EfficientNetB3_Block5          | per patch       | SVG                                      | (2, 2, 2)  | 0.61765263 | 0.07836333 | 0.14948139
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.61819849 | 0.07760296 | 0.14915463
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.59333682 | 0.06416062 | 0.12982699
# EfficientNetB3_Block5          | per patch       | BalancedDistributionSVG/500/9/0.30       | (2, 2, 2)  | 0.67337620 | 0.08454019 | 0.14929898
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.60116354 | 0.07322166 | 0.14463066
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.59153647 | 0.06416931 | 0.13028704
# EfficientNetB3_Block5          | per patch       | SVG                                      | (0, 1, 1)  | 0.60086700 | 0.07515363 | 0.14446757
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.60151854 | 0.07405916 | 0.14432671
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.59265130 | 0.06505418 | 0.12896045
# EfficientNetB3_Block5          | per patch       | BalancedDistributionSVG/500/9/0.30       | (0, 1, 1)  | 0.64975633 | 0.08141418 | 0.14988259
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.61887972 | 0.07734966 | 0.14596500
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.59708037 | 0.06538633 | 0.13554237
# EfficientNetB3_Block5          | per patch       | SVG                                      | (0, 2, 2)  | 0.61876706 | 0.07888953 | 0.14841790
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.61927878 | 0.07785811 | 0.14603017
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.59948940 | 0.06670433 | 0.13584147
# EfficientNetB3_Block5          | per patch       | BalancedDistributionSVG/500/9/0.30       | (0, 2, 2)  | 0.66609337 | 0.08443684 | 0.15474584
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.60400151 | 0.07180922 | 0.14470368
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.59930774 | 0.06475662 | 0.11961428
# EfficientNetB3_Block5          | per patch       | SVG                                      | (1, 0, 0)  | 0.60337239 | 0.07398682 | 0.14635381
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.60419595 | 0.07295326 | 0.14484585
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.59172611 | 0.06286444 | 0.11443825
# EfficientNetB3_Block5          | per patch       | BalancedDistributionSVG/500/9/0.30       | (1, 0, 0)  | 0.61317801 | 0.07336120 | 0.13747635
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.60586224 | 0.07203353 | 0.14417532
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.59601527 | 0.06478919 | 0.11789991
# EfficientNetB3_Block5          | per patch       | SVG                                      | (2, 0, 0)  | 0.60527783 | 0.07374561 | 0.14572907
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.60610793 | 0.07310497 | 0.14448994
# EfficientNetB3_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.58824439 | 0.06168785 | 0.11082654
# EfficientNetB3_Block5          | per patch       | BalancedDistributionSVG/500/9/0.30       | (2, 0, 0)  | 0.61203780 | 0.07323736 | 0.13508171
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.59024447 | 0.50818012 | 0.64356436
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.49070067 | 0.42128863 | 0.62172285
# EfficientNetB3_Block5          | per frame (max) | SVG                                      | None       | 0.59328576 | 0.55424010 | 0.62559242
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.61480875 | 0.55299838 | 0.62295082
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.49011580 | 0.41587746 | 0.62406015
# EfficientNetB3_Block5          | per frame (max) | BalancedDistributionSVG/500/9/0.30       | None       | 0.51175576 | 0.45520198 | 0.62790698
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.57726050 | 0.48910303 | 0.65116279
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.46017078 | 0.39666382 | 0.61710037
# EfficientNetB3_Block5          | per frame (max) | SVG                                      | (1, 1, 1)  | 0.55948064 | 0.50216460 | 0.64545455
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.58252427 | 0.49314981 | 0.64390244
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.48613873 | 0.41715420 | 0.61710037
# EfficientNetB3_Block5          | per frame (max) | BalancedDistributionSVG/500/9/0.30       | (1, 1, 1)  | 0.56205404 | 0.51298880 | 0.64341085
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.61094865 | 0.52911461 | 0.65116279
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.44531524 | 0.38736328 | 0.61710037
# EfficientNetB3_Block5          | per frame (max) | SVG                                      | (2, 2, 2)  | 0.62381565 | 0.56432620 | 0.69527897
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.62276290 | 0.53594858 | 0.66990291
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.47572816 | 0.40591053 | 0.61710037
# EfficientNetB3_Block5          | per frame (max) | BalancedDistributionSVG/500/9/0.30       | (2, 2, 2)  | 0.59843257 | 0.55705074 | 0.68067227
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.57819628 | 0.50054318 | 0.63594470
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.47058135 | 0.40448328 | 0.61710037
# EfficientNetB3_Block5          | per frame (max) | SVG                                      | (0, 1, 1)  | 0.57492104 | 0.53221098 | 0.63601533
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.58439584 | 0.50780807 | 0.62903226
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.46449877 | 0.39734234 | 0.61940299
# EfficientNetB3_Block5          | per frame (max) | BalancedDistributionSVG/500/9/0.30       | (0, 1, 1)  | 0.56509533 | 0.51947113 | 0.63601533
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.57667563 | 0.51763130 | 0.62439024
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.47993917 | 0.40568715 | 0.61940299
# EfficientNetB3_Block5          | per frame (max) | SVG                                      | (0, 2, 2)  | 0.58217335 | 0.54452528 | 0.64516129
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.59644403 | 0.52781893 | 0.63054187
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.48052404 | 0.40533185 | 0.61940299
# EfficientNetB3_Block5          | per frame (max) | BalancedDistributionSVG/500/9/0.30       | (0, 2, 2)  | 0.53748976 | 0.51723923 | 0.63453815
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.56486139 | 0.48057843 | 0.63436123
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.47093227 | 0.40547281 | 0.61710037
# EfficientNetB3_Block5          | per frame (max) | SVG                                      | (1, 0, 0)  | 0.54719850 | 0.50464928 | 0.63565891
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.57796233 | 0.50088286 | 0.63247863
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.51362732 | 0.43399313 | 0.62348178
# EfficientNetB3_Block5          | per frame (max) | BalancedDistributionSVG/500/9/0.30       | (1, 0, 0)  | 0.48239560 | 0.43559607 | 0.63813230
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.58170546 | 0.49099151 | 0.62626263
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.44414551 | 0.39079896 | 0.61710037
# EfficientNetB3_Block5          | per frame (max) | SVG                                      | (2, 0, 0)  | 0.54860218 | 0.49096974 | 0.64341085
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.58884080 | 0.50011153 | 0.66371681
# EfficientNetB3_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.47713183 | 0.41687015 | 0.62151394
# EfficientNetB3_Block5          | per frame (max) | BalancedDistributionSVG/500/9/0.30       | (2, 0, 0)  | 0.48064101 | 0.44087716 | 0.64341085
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.60287753 | 0.52825374 | 0.65454545
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.52766405 | 0.44151610 | 0.61710037
# EfficientNetB3_Block5          | per frame (sum) | SVG                                      | None       | 0.64451983 | 0.54997723 | 0.65777778
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.62557024 | 0.54458068 | 0.68627451
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.53760674 | 0.44925134 | 0.61940299
# EfficientNetB3_Block5          | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | None       | 0.62252895 | 0.54293187 | 0.64541833
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.64112762 | 0.57487943 | 0.68103448
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.53515031 | 0.44089526 | 0.62500000
# EfficientNetB3_Block5          | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.65668499 | 0.57261052 | 0.66666667
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.64779506 | 0.57315384 | 0.68444444
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.54860218 | 0.45098835 | 0.62886598
# EfficientNetB3_Block5          | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | (1, 1, 1)  | 0.63551293 | 0.55489341 | 0.64356436
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.62381565 | 0.56198217 | 0.66666667
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.49245526 | 0.41461739 | 0.61904762
# EfficientNetB3_Block5          | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.62849456 | 0.56514752 | 0.64197531
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.63679963 | 0.56656753 | 0.68619247
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.50356767 | 0.43199443 | 0.61940299
# EfficientNetB3_Block5          | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | (2, 2, 2)  | 0.61469178 | 0.55319655 | 0.63414634
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.60287753 | 0.52825374 | 0.65454545
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.52766405 | 0.44151610 | 0.61710037
# EfficientNetB3_Block5          | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.64451983 | 0.54997723 | 0.65777778
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.62557024 | 0.54458068 | 0.68627451
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.53760674 | 0.44925134 | 0.61940299
# EfficientNetB3_Block5          | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | (0, 1, 1)  | 0.62252895 | 0.54293187 | 0.64541833
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.60287753 | 0.52825374 | 0.65454545
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.52766405 | 0.44151610 | 0.61710037
# EfficientNetB3_Block5          | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.64451983 | 0.54997723 | 0.65777778
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.62557024 | 0.54458068 | 0.68627451
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.53760674 | 0.44925134 | 0.61940299
# EfficientNetB3_Block5          | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | (0, 2, 2)  | 0.62252895 | 0.54293187 | 0.64541833
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.64112762 | 0.57487943 | 0.68103448
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.53515031 | 0.44089526 | 0.62500000
# EfficientNetB3_Block5          | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.65668499 | 0.57261052 | 0.66666667
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.64779506 | 0.57315384 | 0.68444444
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.54860218 | 0.45098835 | 0.62886598
# EfficientNetB3_Block5          | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | (1, 0, 0)  | 0.63551293 | 0.55489341 | 0.64356436
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.62381565 | 0.56198217 | 0.66666667
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.49245526 | 0.41461739 | 0.61904762
# EfficientNetB3_Block5          | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.62849456 | 0.56514752 | 0.64197531
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.63679963 | 0.56656753 | 0.68619247
# EfficientNetB3_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.50356767 | 0.43199443 | 0.61940299
# EfficientNetB3_Block5          | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | (2, 0, 0)  | 0.61469178 | 0.55319655 | 0.63414634
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/0.20                      | None       | 0.57774868 | 0.05577907 | 0.11591688
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.56435374 | 0.05218411 | 0.10941041
# EfficientNetB6_Block4          | per patch       | SVG                                      | None       | 0.55879709 | 0.05803952 | 0.11347276
# EfficientNetB6_Block4          | per patch       | BalancedDistributionSVG/500/10/0.30      | None       | 0.55171565 | 0.05872055 | 0.12489335
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/0.50                      | None       | 0.57828222 | 0.05570898 | 0.11569491
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.55995969 | 0.05160883 | 0.10789337
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.57990550 | 0.05564426 | 0.11350144
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.56800548 | 0.05490981 | 0.10864745
# EfficientNetB6_Block4          | per patch       | SVG                                      | (1, 1, 1)  | 0.56379016 | 0.05776773 | 0.10582117
# EfficientNetB6_Block4          | per patch       | BalancedDistributionSVG/500/10/0.30      | (1, 1, 1)  | 0.57119251 | 0.05932271 | 0.11787945
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.58053149 | 0.05560730 | 0.11356889
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.56316990 | 0.05361926 | 0.10671890
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.57382542 | 0.05588336 | 0.10712422
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.55084298 | 0.05227020 | 0.10388978
# EfficientNetB6_Block4          | per patch       | SVG                                      | (2, 2, 2)  | 0.57020640 | 0.05880335 | 0.11312074
# EfficientNetB6_Block4          | per patch       | BalancedDistributionSVG/500/10/0.30      | (2, 2, 2)  | 0.59198891 | 0.06127154 | 0.11622152
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.57389526 | 0.05584079 | 0.10723570
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.54769523 | 0.05184911 | 0.10289579
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.58036570 | 0.05576896 | 0.11328205
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.56289007 | 0.05180622 | 0.10794448
# EfficientNetB6_Block4          | per patch       | SVG                                      | (0, 1, 1)  | 0.56443524 | 0.05815597 | 0.10704578
# EfficientNetB6_Block4          | per patch       | BalancedDistributionSVG/500/10/0.30      | (0, 1, 1)  | 0.57375295 | 0.06012276 | 0.12049031
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.58106574 | 0.05578293 | 0.11336437
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.55783616 | 0.05103702 | 0.10612405
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.57444790 | 0.05600401 | 0.10694137
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.55772321 | 0.05130712 | 0.10444172
# EfficientNetB6_Block4          | per patch       | SVG                                      | (0, 2, 2)  | 0.57074681 | 0.05943170 | 0.11327896
# EfficientNetB6_Block4          | per patch       | BalancedDistributionSVG/500/10/0.30      | (0, 2, 2)  | 0.59316743 | 0.06222027 | 0.11811285
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.57470762 | 0.05602433 | 0.10702432
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.55288676 | 0.05054230 | 0.10344023
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.57731191 | 0.05565085 | 0.11557060
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.57073789 | 0.05547360 | 0.11047165
# EfficientNetB6_Block4          | per patch       | SVG                                      | (1, 0, 0)  | 0.55659797 | 0.05723898 | 0.11068237
# EfficientNetB6_Block4          | per patch       | BalancedDistributionSVG/500/10/0.30      | (1, 0, 0)  | 0.54725441 | 0.05758322 | 0.11927007
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.57794760 | 0.05554626 | 0.11559199
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.56730290 | 0.05457392 | 0.10907235
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.57687419 | 0.05557650 | 0.11573087
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.56695627 | 0.05416500 | 0.11042715
# EfficientNetB6_Block4          | per patch       | SVG                                      | (2, 0, 0)  | 0.55665698 | 0.05669775 | 0.10909175
# EfficientNetB6_Block4          | per patch       | BalancedDistributionSVG/500/10/0.30      | (2, 0, 0)  | 0.54670711 | 0.05686918 | 0.11628931
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.57750186 | 0.05548629 | 0.11583520
# EfficientNetB6_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.56292970 | 0.05393167 | 0.10849611
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.54544391 | 0.46436515 | 0.62068966
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.53140718 | 0.46907859 | 0.63601533
# EfficientNetB6_Block4          | per frame (max) | SVG                                      | None       | 0.55211136 | 0.54263941 | 0.61847390
# EfficientNetB6_Block4          | per frame (max) | BalancedDistributionSVG/500/10/0.30      | None       | 0.54158381 | 0.54018200 | 0.61710037
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.51503100 | 0.43163743 | 0.61710037
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.52181542 | 0.47762702 | 0.61710037
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.55866183 | 0.47296328 | 0.62439024
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.58907475 | 0.49541077 | 0.64092664
# EfficientNetB6_Block4          | per frame (max) | SVG                                      | (1, 1, 1)  | 0.53947830 | 0.54614409 | 0.61710037
# EfficientNetB6_Block4          | per frame (max) | BalancedDistributionSVG/500/10/0.30      | (1, 1, 1)  | 0.58965961 | 0.52125871 | 0.64253394
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.52743011 | 0.45062437 | 0.61835749
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.64019184 | 0.52946155 | 0.67661692
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.42332437 | 0.37603100 | 0.61710037
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.50695988 | 0.44330926 | 0.63846154
# EfficientNetB6_Block4          | per frame (max) | SVG                                      | (2, 2, 2)  | 0.60626974 | 0.53280693 | 0.66055046
# EfficientNetB6_Block4          | per frame (max) | BalancedDistributionSVG/500/10/0.30      | (2, 2, 2)  | 0.60802433 | 0.54631639 | 0.65573770
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.41712481 | 0.37521673 | 0.61710037
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.51409522 | 0.45740443 | 0.61710037
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.55082466 | 0.46687493 | 0.62068966
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.53971225 | 0.48254275 | 0.63601533
# EfficientNetB6_Block4          | per frame (max) | SVG                                      | (0, 1, 1)  | 0.53748976 | 0.53514793 | 0.61710037
# EfficientNetB6_Block4          | per frame (max) | BalancedDistributionSVG/500/10/0.30      | (0, 1, 1)  | 0.56298982 | 0.53149330 | 0.63846154
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.52310212 | 0.44615451 | 0.61710037
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.53433150 | 0.47976762 | 0.61710037
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.42847117 | 0.37764415 | 0.61710037
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.47607907 | 0.42338515 | 0.62790698
# EfficientNetB6_Block4          | per frame (max) | SVG                                      | (0, 2, 2)  | 0.59422155 | 0.56012350 | 0.63636364
# EfficientNetB6_Block4          | per frame (max) | BalancedDistributionSVG/500/10/0.30      | (0, 2, 2)  | 0.59726284 | 0.58062360 | 0.62910798
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.38542520 | 0.36037421 | 0.61710037
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.44777167 | 0.42216025 | 0.61710037
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.54614575 | 0.46269640 | 0.62439024
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.57889812 | 0.48768497 | 0.63846154
# EfficientNetB6_Block4          | per frame (max) | SVG                                      | (1, 0, 0)  | 0.54251959 | 0.54151857 | 0.61904762
# EfficientNetB6_Block4          | per frame (max) | BalancedDistributionSVG/500/10/0.30      | (1, 0, 0)  | 0.54333840 | 0.54185459 | 0.61904762
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.51620073 | 0.43459636 | 0.62439024
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.62638905 | 0.54126302 | 0.67256637
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.52801497 | 0.44123489 | 0.62801932
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.54193473 | 0.46134784 | 0.63358779
# EfficientNetB6_Block4          | per frame (max) | SVG                                      | (2, 0, 0)  | 0.53784068 | 0.54246770 | 0.61710037
# EfficientNetB6_Block4          | per frame (max) | BalancedDistributionSVG/500/10/0.30      | (2, 0, 0)  | 0.53795766 | 0.54314925 | 0.61710037
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.51140484 | 0.43032267 | 0.62801932
# EfficientNetB6_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.56439350 | 0.52014509 | 0.64347826
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.47783366 | 0.42111119 | 0.63358779
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.40601240 | 0.38400839 | 0.61710037
# EfficientNetB6_Block4          | per frame (sum) | SVG                                      | None       | 0.47900339 | 0.47014662 | 0.62121212
# EfficientNetB6_Block4          | per frame (sum) | BalancedDistributionSVG/500/10/0.30      | None       | 0.45584279 | 0.44169747 | 0.61832061
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.46461574 | 0.41165370 | 0.62878788
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.40297111 | 0.39098850 | 0.61710037
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.48461808 | 0.42781660 | 0.63117871
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.45338636 | 0.42061119 | 0.61710037
# EfficientNetB6_Block4          | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.46660428 | 0.49889551 | 0.61832061
# EfficientNetB6_Block4          | per frame (sum) | BalancedDistributionSVG/500/10/0.30      | (1, 1, 1)  | 0.44122120 | 0.44721882 | 0.62307692
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.47116622 | 0.41898849 | 0.63117871
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.48578781 | 0.44250649 | 0.61710037
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.47701486 | 0.42901272 | 0.63358779
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.44823956 | 0.41732681 | 0.61710037
# EfficientNetB6_Block4          | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.44180606 | 0.49736659 | 0.62172285
# EfficientNetB6_Block4          | per frame (sum) | BalancedDistributionSVG/500/10/0.30      | (2, 2, 2)  | 0.42238858 | 0.46568782 | 0.62641509
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.46414785 | 0.42114054 | 0.62878788
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.46438180 | 0.42383492 | 0.61710037
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.47783366 | 0.42111119 | 0.63358779
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.40601240 | 0.38400839 | 0.61710037
# EfficientNetB6_Block4          | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.47900339 | 0.47014662 | 0.62121212
# EfficientNetB6_Block4          | per frame (sum) | BalancedDistributionSVG/500/10/0.30      | (0, 1, 1)  | 0.45584279 | 0.44169747 | 0.61832061
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.46461574 | 0.41165370 | 0.62878788
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.40297111 | 0.39098850 | 0.61710037
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.47783366 | 0.42111119 | 0.63358779
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.40601240 | 0.38400839 | 0.61710037
# EfficientNetB6_Block4          | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.47900339 | 0.47014662 | 0.62121212
# EfficientNetB6_Block4          | per frame (sum) | BalancedDistributionSVG/500/10/0.30      | (0, 2, 2)  | 0.45584279 | 0.44169747 | 0.61832061
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.46461574 | 0.41165370 | 0.62878788
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.40297111 | 0.39098850 | 0.61710037
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.48461808 | 0.42781660 | 0.63117871
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.45338636 | 0.42061119 | 0.61710037
# EfficientNetB6_Block4          | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.46660428 | 0.49889551 | 0.61832061
# EfficientNetB6_Block4          | per frame (sum) | BalancedDistributionSVG/500/10/0.30      | (1, 0, 0)  | 0.44122120 | 0.44721882 | 0.62307692
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.47116622 | 0.41898849 | 0.63117871
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.48578781 | 0.44250649 | 0.61710037
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.47701486 | 0.42901272 | 0.63358779
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.44823956 | 0.41732681 | 0.61710037
# EfficientNetB6_Block4          | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.44180606 | 0.49736659 | 0.62172285
# EfficientNetB6_Block4          | per frame (sum) | BalancedDistributionSVG/500/10/0.30      | (2, 0, 0)  | 0.42238858 | 0.46568782 | 0.62641509
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.46414785 | 0.42114054 | 0.62878788
# EfficientNetB6_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.46438180 | 0.42383492 | 0.61710037
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/0.20                      | None       | 0.86837808 | 0.15564169 | 0.27362331
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.87982134 | 0.21003087 | 0.30614754
# MobileNetV2_Block12            | per patch       | SVG                                      | None       | 0.86455217 | 0.15192158 | 0.27009237
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/0.50                      | None       | 0.86836999 | 0.15558226 | 0.27385288
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.87731573 | 0.21755066 | 0.31543168
# MobileNetV2_Block12            | per patch       | BalancedDistributionSVG/500/9/0.30       | None       | 0.86368484 | 0.16690903 | 0.27493341
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.86561400 | 0.14535644 | 0.27453442
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.89474900 | 0.21118866 | 0.31481804
# MobileNetV2_Block12            | per patch       | SVG                                      | (1, 1, 1)  | 0.86224470 | 0.14448800 | 0.27097865
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.86570234 | 0.14555429 | 0.27408725
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.89460511 | 0.22204704 | 0.32334897
# MobileNetV2_Block12            | per patch       | BalancedDistributionSVG/500/9/0.30       | (1, 1, 1)  | 0.87538365 | 0.17017386 | 0.28962452
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.86073946 | 0.15369147 | 0.26979792
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.89565297 | 0.21732919 | 0.32675826
# MobileNetV2_Block12            | per patch       | SVG                                      | (2, 2, 2)  | 0.85548803 | 0.15509514 | 0.26075891
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.86059810 | 0.15391144 | 0.27015203
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.89647958 | 0.24153479 | 0.32540999
# MobileNetV2_Block12            | per patch       | BalancedDistributionSVG/500/9/0.30       | (2, 2, 2)  | 0.86549905 | 0.18140602 | 0.28060816
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.88189889 | 0.16525278 | 0.29687668
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.90517381 | 0.26940730 | 0.35824225
# MobileNetV2_Block12            | per patch       | SVG                                      | (0, 1, 1)  | 0.87723668 | 0.16124908 | 0.29147806
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.88187175 | 0.16529911 | 0.29631947
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.90292061 | 0.28630648 | 0.36359393
# MobileNetV2_Block12            | per patch       | BalancedDistributionSVG/500/9/0.30       | (0, 1, 1)  | 0.88660960 | 0.19272069 | 0.30320553
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.89972571 | 0.19442853 | 0.33396138
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.92811703 | 0.34833861 | 0.39581060
# MobileNetV2_Block12            | per patch       | SVG                                      | (0, 2, 2)  | 0.89239368 | 0.18593520 | 0.31705171
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.89947111 | 0.19416321 | 0.33254494
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.92600187 | 0.39081249 | 0.40193470
# MobileNetV2_Block12            | per patch       | BalancedDistributionSVG/500/9/0.30       | (0, 2, 2)  | 0.90113212 | 0.22475917 | 0.32554672
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.85961229 | 0.14105488 | 0.26584459
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.87920607 | 0.18567063 | 0.29087894
# MobileNetV2_Block12            | per patch       | SVG                                      | (1, 0, 0)  | 0.85648447 | 0.13932642 | 0.26255642
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.85968838 | 0.14115004 | 0.26567657
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.87781656 | 0.19306169 | 0.29592788
# MobileNetV2_Block12            | per patch       | BalancedDistributionSVG/500/9/0.30       | (1, 0, 0)  | 0.86561782 | 0.15780988 | 0.27673101
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.83819951 | 0.12614091 | 0.24129466
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.86024089 | 0.15604740 | 0.26181255
# MobileNetV2_Block12            | per patch       | SVG                                      | (2, 0, 0)  | 0.83619619 | 0.12624946 | 0.23690510
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.83827430 | 0.12627784 | 0.24170444
# MobileNetV2_Block12            | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.85963575 | 0.16101334 | 0.26303656
# MobileNetV2_Block12            | per patch       | BalancedDistributionSVG/500/9/0.30       | (2, 0, 0)  | 0.84642070 | 0.14082853 | 0.25426901
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.48777635 | 0.44503937 | 0.62878788
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.51877413 | 0.45934863 | 0.61710037
# MobileNetV2_Block12            | per frame (max) | SVG                                      | None       | 0.56486139 | 0.51428620 | 0.62745098
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.49011580 | 0.44920318 | 0.62641509
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.51351035 | 0.44993639 | 0.61710037
# MobileNetV2_Block12            | per frame (max) | BalancedDistributionSVG/500/9/0.30       | None       | 0.56684992 | 0.55675991 | 0.61940299
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.53632004 | 0.45550570 | 0.65079365
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.48286349 | 0.41391782 | 0.61710037
# MobileNetV2_Block12            | per frame (max) | SVG                                      | (1, 1, 1)  | 0.54170078 | 0.44913698 | 0.66135458
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.53468242 | 0.45327964 | 0.65322581
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.51163879 | 0.44023055 | 0.61710037
# MobileNetV2_Block12            | per frame (max) | BalancedDistributionSVG/500/9/0.30       | (1, 1, 1)  | 0.54380629 | 0.45676080 | 0.66396761
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.50637501 | 0.41338655 | 0.69198312
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.52836589 | 0.45201369 | 0.63813230
# MobileNetV2_Block12            | per frame (max) | SVG                                      | (2, 2, 2)  | 0.51772137 | 0.42156669 | 0.67479675
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.50450345 | 0.41259149 | 0.69166667
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.59866651 | 0.57013816 | 0.62992126
# MobileNetV2_Block12            | per frame (max) | BalancedDistributionSVG/500/9/0.30       | (2, 2, 2)  | 0.51245760 | 0.42008875 | 0.65600000
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.53374664 | 0.44905588 | 0.66666667
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.55480173 | 0.47573674 | 0.61940299
# MobileNetV2_Block12            | per frame (max) | SVG                                      | (0, 1, 1)  | 0.55550357 | 0.46041854 | 0.65822785
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.53245994 | 0.44797959 | 0.66400000
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.56415955 | 0.48472259 | 0.62068966
# MobileNetV2_Block12            | per frame (max) | BalancedDistributionSVG/500/9/0.30       | (0, 1, 1)  | 0.56076734 | 0.48585575 | 0.65271967
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.50157913 | 0.40603300 | 0.65354331
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.63095099 | 0.54938301 | 0.63241107
# MobileNetV2_Block12            | per frame (max) | SVG                                      | (0, 2, 2)  | 0.50368464 | 0.41306807 | 0.64062500
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.50298281 | 0.40666962 | 0.65354331
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.66417125 | 0.63609823 | 0.63478261
# MobileNetV2_Block12            | per frame (max) | BalancedDistributionSVG/500/9/0.30       | (0, 2, 2)  | 0.52298514 | 0.45358057 | 0.63320463
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.48953094 | 0.41647852 | 0.62641509
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.48087496 | 0.42298255 | 0.61710037
# MobileNetV2_Block12            | per frame (max) | SVG                                      | (1, 0, 0)  | 0.53924436 | 0.47891787 | 0.63601533
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.48754240 | 0.41814638 | 0.62641509
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.48216166 | 0.42815302 | 0.61710037
# MobileNetV2_Block12            | per frame (max) | BalancedDistributionSVG/500/9/0.30       | (1, 0, 0)  | 0.60112294 | 0.56780749 | 0.63779528
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.47666394 | 0.41652991 | 0.62172285
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.45209966 | 0.39348454 | 0.61940299
# MobileNetV2_Block12            | per frame (max) | SVG                                      | (2, 0, 0)  | 0.52520763 | 0.45843815 | 0.62992126
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.47409054 | 0.41671616 | 0.62172285
# MobileNetV2_Block12            | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.50216400 | 0.43268231 | 0.61710037
# MobileNetV2_Block12            | per frame (max) | BalancedDistributionSVG/500/9/0.30       | (2, 0, 0)  | 0.58486373 | 0.53948040 | 0.62992126
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.72920809 | 0.70883936 | 0.66666667
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.70780208 | 0.68419551 | 0.65771812
# MobileNetV2_Block12            | per frame (sum) | SVG                                      | None       | 0.69832729 | 0.68086016 | 0.65322581
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.72698561 | 0.70766211 | 0.66382979
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.70359106 | 0.69635814 | 0.65771812
# MobileNetV2_Block12            | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | None       | 0.69926307 | 0.69124224 | 0.65000000
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.71610715 | 0.70280256 | 0.67058824
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.68405661 | 0.66369611 | 0.64556962
# MobileNetV2_Block12            | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.68323781 | 0.66773843 | 0.64377682
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.71318283 | 0.70191881 | 0.66382979
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.68405661 | 0.67550560 | 0.64088398
# MobileNetV2_Block12            | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | (1, 1, 1)  | 0.68651304 | 0.67825849 | 0.65497076
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.68756580 | 0.67435491 | 0.64705882
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.65960931 | 0.63513699 | 0.63387978
# MobileNetV2_Block12            | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.65411159 | 0.63820200 | 0.64489796
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.68546029 | 0.67261304 | 0.64435146
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.65937537 | 0.64633413 | 0.63829787
# MobileNetV2_Block12            | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | (2, 2, 2)  | 0.66206574 | 0.64586127 | 0.65020576
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.72920809 | 0.70883936 | 0.66666667
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.70780208 | 0.68419551 | 0.65771812
# MobileNetV2_Block12            | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.69832729 | 0.68086016 | 0.65322581
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.72698561 | 0.70766211 | 0.66382979
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.70359106 | 0.69635814 | 0.65771812
# MobileNetV2_Block12            | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | (0, 1, 1)  | 0.69926307 | 0.69124224 | 0.65000000
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.72920809 | 0.70883936 | 0.66666667
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.70780208 | 0.68419551 | 0.65771812
# MobileNetV2_Block12            | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.69832729 | 0.68086016 | 0.65322581
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.72698561 | 0.70766211 | 0.66382979
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.70359106 | 0.69635814 | 0.65771812
# MobileNetV2_Block12            | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | (0, 2, 2)  | 0.69926307 | 0.69124224 | 0.65000000
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.71610715 | 0.70280256 | 0.67058824
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.68405661 | 0.66369611 | 0.64556962
# MobileNetV2_Block12            | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.68323781 | 0.66773843 | 0.64377682
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.71318283 | 0.70191881 | 0.66382979
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.68405661 | 0.67550560 | 0.64088398
# MobileNetV2_Block12            | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | (1, 0, 0)  | 0.68651304 | 0.67825849 | 0.65497076
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.68756580 | 0.67435491 | 0.64705882
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.65960931 | 0.63513699 | 0.63387978
# MobileNetV2_Block12            | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.65411159 | 0.63820200 | 0.64489796
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.68546029 | 0.67261304 | 0.64435146
# MobileNetV2_Block12            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.65937537 | 0.64633413 | 0.63829787
# MobileNetV2_Block12            | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | (2, 0, 0)  | 0.66206574 | 0.64586127 | 0.65020576
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/0.20                      | None       | 0.89222522 | 0.26472794 | 0.30289308
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.89794483 | 0.29182191 | 0.32375887
# MobileNetV2_Block6             | per patch       | SVG                                      | None       | 0.86931635 | 0.14690583 | 0.30356385
# MobileNetV2_Block6             | per patch       | BalancedDistributionSVG/500/7/0.30       | None       | 0.82932677 | 0.14958522 | 0.28626296
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/0.50                      | None       | 0.89201240 | 0.25560869 | 0.30252496
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.89512196 | 0.30877073 | 0.32411526
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.89766237 | 0.25891455 | 0.32386496
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.91208717 | 0.29928731 | 0.34589996
# MobileNetV2_Block6             | per patch       | SVG                                      | (1, 1, 1)  | 0.86395540 | 0.14074131 | 0.28520264
# MobileNetV2_Block6             | per patch       | BalancedDistributionSVG/500/7/0.30       | (1, 1, 1)  | 0.87761131 | 0.16067359 | 0.30848675
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.89620357 | 0.24612395 | 0.32370506
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.91149513 | 0.32407804 | 0.35036735
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.88859897 | 0.22530223 | 0.30700523
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.90369365 | 0.27990023 | 0.34942758
# MobileNetV2_Block6             | per patch       | SVG                                      | (2, 2, 2)  | 0.84923923 | 0.13872931 | 0.24994696
# MobileNetV2_Block6             | per patch       | BalancedDistributionSVG/500/7/0.30       | (2, 2, 2)  | 0.86188885 | 0.15496546 | 0.27629187
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.88697518 | 0.21259518 | 0.30279685
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.90364464 | 0.30482771 | 0.35074074
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.91211829 | 0.33977048 | 0.33173491
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.92382978 | 0.38752656 | 0.38395674
# MobileNetV2_Block6             | per patch       | SVG                                      | (0, 1, 1)  | 0.87637891 | 0.15107323 | 0.30832598
# MobileNetV2_Block6             | per patch       | BalancedDistributionSVG/500/7/0.30       | (0, 1, 1)  | 0.88261870 | 0.16916445 | 0.32790781
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.91124970 | 0.32676149 | 0.33158088
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.92198581 | 0.41424743 | 0.39082058
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.91955476 | 0.35172690 | 0.37569252
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.93196619 | 0.43153130 | 0.42817164
# MobileNetV2_Block6             | per patch       | SVG                                      | (0, 2, 2)  | 0.87942082 | 0.15824503 | 0.30442055
# MobileNetV2_Block6             | per patch       | BalancedDistributionSVG/500/7/0.30       | (0, 2, 2)  | 0.88613841 | 0.17204400 | 0.32122762
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.91859339 | 0.34046085 | 0.37459283
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.93094908 | 0.45896551 | 0.44415174
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.88348922 | 0.21493854 | 0.29861259
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.89266648 | 0.23613139 | 0.31283792
# MobileNetV2_Block6             | per patch       | SVG                                      | (1, 0, 0)  | 0.85930389 | 0.13700471 | 0.29038218
# MobileNetV2_Block6             | per patch       | BalancedDistributionSVG/500/7/0.30       | (1, 0, 0)  | 0.84824342 | 0.14873733 | 0.29038370
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.88263198 | 0.20837349 | 0.29913574
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.89105443 | 0.25450153 | 0.31261676
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.86115649 | 0.17896214 | 0.27025423
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.87048628 | 0.19158022 | 0.28007890
# MobileNetV2_Block6             | per patch       | SVG                                      | (2, 0, 0)  | 0.84331410 | 0.12816414 | 0.25743114
# MobileNetV2_Block6             | per patch       | BalancedDistributionSVG/500/7/0.30       | (2, 0, 0)  | 0.83314752 | 0.13859156 | 0.26756946
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.86074087 | 0.17390380 | 0.27098201
# MobileNetV2_Block6             | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.86920927 | 0.20774341 | 0.28231761
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.58533162 | 0.61284057 | 0.61940299
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.58427886 | 0.55795399 | 0.63358779
# MobileNetV2_Block6             | per frame (max) | SVG                                      | None       | 0.50754474 | 0.42756847 | 0.63076923
# MobileNetV2_Block6             | per frame (max) | BalancedDistributionSVG/500/7/0.30       | None       | 0.50532226 | 0.43516941 | 0.61710037
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.59866651 | 0.62181583 | 0.61710037
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.60697158 | 0.59588747 | 0.63117871
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.56006550 | 0.57607377 | 0.61718750
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.62089133 | 0.56243557 | 0.67768595
# MobileNetV2_Block6             | per frame (max) | SVG                                      | (1, 1, 1)  | 0.55620540 | 0.49813621 | 0.64978903
# MobileNetV2_Block6             | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (1, 1, 1)  | 0.50836355 | 0.46975165 | 0.62595420
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.55374898 | 0.56688393 | 0.62068966
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.64674231 | 0.63193140 | 0.66382979
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.58813896 | 0.55856552 | 0.63117871
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.61644637 | 0.56221502 | 0.64000000
# MobileNetV2_Block6             | per frame (max) | SVG                                      | (2, 2, 2)  | 0.59539127 | 0.56800755 | 0.64754098
# MobileNetV2_Block6             | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (2, 2, 2)  | 0.58229033 | 0.51120282 | 0.65098039
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.59702889 | 0.56469298 | 0.63358779
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.61551059 | 0.57336079 | 0.63507109
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.58205638 | 0.59445956 | 0.62878788
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.66007720 | 0.59970894 | 0.66129032
# MobileNetV2_Block6             | per frame (max) | SVG                                      | (0, 1, 1)  | 0.51468008 | 0.46994040 | 0.63117871
# MobileNetV2_Block6             | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (0, 1, 1)  | 0.47795064 | 0.44467212 | 0.62172285
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.57737747 | 0.57735452 | 0.63358779
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.68943736 | 0.67439129 | 0.66666667
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.59328576 | 0.60176629 | 0.62641509
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.66534098 | 0.63738353 | 0.63900415
# MobileNetV2_Block6             | per frame (max) | SVG                                      | (0, 2, 2)  | 0.54018014 | 0.49262200 | 0.62878788
# MobileNetV2_Block6             | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (0, 2, 2)  | 0.51245760 | 0.47165018 | 0.62172285
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.58895777 | 0.59961002 | 0.62878788
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.67657036 | 0.67831168 | 0.64242424
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.55164347 | 0.56861041 | 0.63598326
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.56532928 | 0.50219328 | 0.66129032
# MobileNetV2_Block6             | per frame (max) | SVG                                      | (1, 0, 0)  | 0.55176044 | 0.46548751 | 0.62745098
# MobileNetV2_Block6             | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (1, 0, 0)  | 0.53550123 | 0.46174973 | 0.63076923
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.55936367 | 0.57270566 | 0.63070539
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.58568254 | 0.57140499 | 0.65843621
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.55398292 | 0.54812554 | 0.66393443
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.55047374 | 0.47431546 | 0.66666667
# MobileNetV2_Block6             | per frame (max) | SVG                                      | (2, 0, 0)  | 0.59164815 | 0.48272051 | 0.64591440
# MobileNetV2_Block6             | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (2, 0, 0)  | 0.58287519 | 0.48273266 | 0.63779528
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.57258159 | 0.55510452 | 0.66666667
# MobileNetV2_Block6             | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.56287285 | 0.55462845 | 0.66938776
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.63258861 | 0.59845737 | 0.63348416
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.66931805 | 0.64986143 | 0.65168539
# MobileNetV2_Block6             | per frame (sum) | SVG                                      | None       | 0.67037080 | 0.59625531 | 0.68899522
# MobileNetV2_Block6             | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | None       | 0.67224237 | 0.60306356 | 0.67942584
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.63562990 | 0.59925028 | 0.63392857
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.66370336 | 0.65347059 | 0.64000000
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.61118259 | 0.58733012 | 0.61710037
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.65177214 | 0.63348516 | 0.62433862
# MobileNetV2_Block6             | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.65528132 | 0.59014288 | 0.66666667
# MobileNetV2_Block6             | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (1, 1, 1)  | 0.68113230 | 0.60842125 | 0.70754717
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.61445783 | 0.58882576 | 0.61710037
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.64604047 | 0.63583424 | 0.62433862
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.58626740 | 0.57363666 | 0.61940299
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.63340742 | 0.61068946 | 0.62121212
# MobileNetV2_Block6             | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.62627208 | 0.57722281 | 0.65546218
# MobileNetV2_Block6             | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (2, 2, 2)  | 0.67423090 | 0.60362419 | 0.71497585
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.58813896 | 0.57605525 | 0.61940299
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.62744181 | 0.61187473 | 0.62641509
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.63258861 | 0.59845737 | 0.63348416
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.66931805 | 0.64986143 | 0.65168539
# MobileNetV2_Block6             | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.67037080 | 0.59625531 | 0.68899522
# MobileNetV2_Block6             | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (0, 1, 1)  | 0.67224237 | 0.60306356 | 0.67942584
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.63562990 | 0.59925028 | 0.63392857
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.66370336 | 0.65347059 | 0.64000000
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.63258861 | 0.59845737 | 0.63348416
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.66931805 | 0.64986143 | 0.65168539
# MobileNetV2_Block6             | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.67037080 | 0.59625531 | 0.68899522
# MobileNetV2_Block6             | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (0, 2, 2)  | 0.67224237 | 0.60306356 | 0.67942584
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.63562990 | 0.59925028 | 0.63392857
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.66370336 | 0.65347059 | 0.64000000
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.61118259 | 0.58733012 | 0.61710037
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.65177214 | 0.63348516 | 0.62433862
# MobileNetV2_Block6             | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.65528132 | 0.59014288 | 0.66666667
# MobileNetV2_Block6             | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (1, 0, 0)  | 0.68113230 | 0.60842125 | 0.70754717
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.61445783 | 0.58882576 | 0.61710037
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.64604047 | 0.63583424 | 0.62433862
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.58626740 | 0.57363666 | 0.61940299
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.63340742 | 0.61068946 | 0.62121212
# MobileNetV2_Block6             | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.62627208 | 0.57722281 | 0.65546218
# MobileNetV2_Block6             | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (2, 0, 0)  | 0.67423090 | 0.60362419 | 0.71497585
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.58813896 | 0.57605525 | 0.61940299
# MobileNetV2_Block6             | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.62744181 | 0.61187473 | 0.62641509
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/0.20                      | None       | 0.91696374 | 0.35579471 | 0.37976970
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.91544338 | 0.37059229 | 0.40446238
# ResNet50V2_Stack3              | per patch       | SVG                                      | None       | 0.90287667 | 0.23909329 | 0.32968560
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/0.50                      | None       | 0.91751021 | 0.35493289 | 0.38277960
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.91116285 | 0.36562070 | 0.39803198
# ResNet50V2_Stack3              | per patch       | BalancedDistributionSVG/500/21/0.30      | None       | 0.91414114 | 0.26666622 | 0.36717217
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.92890203 | 0.35638834 | 0.40237099
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.93384900 | 0.36221453 | 0.42570806
# ResNet50V2_Stack3              | per patch       | SVG                                      | (1, 1, 1)  | 0.89480781 | 0.23562484 | 0.30361057
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.92848825 | 0.35365056 | 0.40035492
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.93109777 | 0.36464399 | 0.41833811
# ResNet50V2_Stack3              | per patch       | BalancedDistributionSVG/500/21/0.30      | (1, 1, 1)  | 0.91706824 | 0.32086744 | 0.35312158
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.91288990 | 0.26057423 | 0.39042664
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.92227732 | 0.29393600 | 0.41084722
# ResNet50V2_Stack3              | per patch       | SVG                                      | (2, 2, 2)  | 0.87429991 | 0.21818678 | 0.27756069
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.91191236 | 0.25587043 | 0.38725591
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.92189083 | 0.29835603 | 0.39978791
# ResNet50V2_Stack3              | per patch       | BalancedDistributionSVG/500/21/0.30      | (2, 2, 2)  | 0.89120037 | 0.31048952 | 0.30464102
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.94254356 | 0.45546707 | 0.45231072
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.94497303 | 0.46455273 | 0.47807229
# ResNet50V2_Stack3              | per patch       | SVG                                      | (0, 1, 1)  | 0.91623869 | 0.28594567 | 0.35370575
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.94291070 | 0.45797583 | 0.45386435
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.94165482 | 0.45517654 | 0.46908919
# ResNet50V2_Stack3              | per patch       | BalancedDistributionSVG/500/21/0.30      | (0, 1, 1)  | 0.93546368 | 0.38808956 | 0.42473118
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.94232654 | 0.43491384 | 0.47710458
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.94895678 | 0.45502461 | 0.48619673
# ResNet50V2_Stack3              | per patch       | SVG                                      | (0, 2, 2)  | 0.91263109 | 0.29000978 | 0.34604677
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.94252581 | 0.43815799 | 0.48061056
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.94716864 | 0.45394426 | 0.48372093
# ResNet50V2_Stack3              | per patch       | BalancedDistributionSVG/500/21/0.30      | (0, 2, 2)  | 0.92976679 | 0.41946069 | 0.41185113
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.91171581 | 0.29512737 | 0.34555754
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.91371148 | 0.31425856 | 0.36280424
# ResNet50V2_Stack3              | per patch       | SVG                                      | (1, 0, 0)  | 0.88904956 | 0.19671167 | 0.29023543
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.91155895 | 0.29193115 | 0.34357683
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.90968866 | 0.31628537 | 0.36602151
# ResNet50V2_Stack3              | per patch       | BalancedDistributionSVG/500/21/0.30      | (1, 0, 0)  | 0.90786295 | 0.23729774 | 0.33910035
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.88988648 | 0.23368528 | 0.29537738
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.89503506 | 0.23885361 | 0.31221278
# ResNet50V2_Stack3              | per patch       | SVG                                      | (2, 0, 0)  | 0.86800729 | 0.17119834 | 0.26198023
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.88948948 | 0.23116451 | 0.29490806
# ResNet50V2_Stack3              | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.89171828 | 0.24271707 | 0.31382735
# ResNet50V2_Stack3              | per patch       | BalancedDistributionSVG/500/21/0.30      | (2, 0, 0)  | 0.88707067 | 0.20411178 | 0.30174419
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.63188677 | 0.65079245 | 0.63333333
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.64136156 | 0.57636304 | 0.64730290
# ResNet50V2_Stack3              | per frame (max) | SVG                                      | None       | 0.59597614 | 0.53512084 | 0.62641509
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.63013218 | 0.64686840 | 0.63559322
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.60860919 | 0.52834243 | 0.64000000
# ResNet50V2_Stack3              | per frame (max) | BalancedDistributionSVG/500/21/0.30      | None       | 0.57620774 | 0.52048886 | 0.62406015
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.63621476 | 0.63269629 | 0.65178571
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.64218037 | 0.55505210 | 0.65271967
# ResNet50V2_Stack3              | per frame (max) | SVG                                      | (1, 1, 1)  | 0.63095099 | 0.56713119 | 0.64341085
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.64136156 | 0.64167638 | 0.65178571
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.57480407 | 0.49622972 | 0.63934426
# ResNet50V2_Stack3              | per frame (max) | BalancedDistributionSVG/500/21/0.30      | (1, 1, 1)  | 0.60545093 | 0.56835338 | 0.63846154
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.67072172 | 0.61861618 | 0.66331658
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.69154287 | 0.59795644 | 0.68000000
# ResNet50V2_Stack3              | per frame (max) | SVG                                      | (2, 2, 2)  | 0.69107498 | 0.68048067 | 0.65240642
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.67879284 | 0.62799815 | 0.66666667
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.61773307 | 0.53999975 | 0.64489796
# ResNet50V2_Stack3              | per frame (max) | BalancedDistributionSVG/500/21/0.30      | (2, 2, 2)  | 0.65902445 | 0.66024337 | 0.64390244
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.69084103 | 0.67961723 | 0.66972477
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.70218739 | 0.61844481 | 0.67942584
# ResNet50V2_Stack3              | per frame (max) | SVG                                      | (0, 1, 1)  | 0.67504971 | 0.62624054 | 0.65486726
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.69306352 | 0.68413413 | 0.67248908
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.66007720 | 0.56073941 | 0.66367713
# ResNet50V2_Stack3              | per frame (max) | BalancedDistributionSVG/500/21/0.30      | (0, 1, 1)  | 0.64498772 | 0.62849364 | 0.63829787
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.68861855 | 0.65998049 | 0.67619048
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.72780442 | 0.67827098 | 0.66371681
# ResNet50V2_Stack3              | per frame (max) | SVG                                      | (0, 2, 2)  | 0.66148087 | 0.62225355 | 0.64069264
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.68920342 | 0.66739929 | 0.66981132
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.69551994 | 0.63994258 | 0.65789474
# ResNet50V2_Stack3              | per frame (max) | BalancedDistributionSVG/500/21/0.30      | (0, 2, 2)  | 0.63586384 | 0.64572331 | 0.62564103
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.62171014 | 0.63637778 | 0.64166667
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.65247397 | 0.55790272 | 0.68103448
# ResNet50V2_Stack3              | per frame (max) | SVG                                      | (1, 0, 0)  | 0.59445549 | 0.47990089 | 0.66666667
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.62849456 | 0.63735683 | 0.64705882
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.58860685 | 0.49090817 | 0.67213115
# ResNet50V2_Stack3              | per frame (max) | BalancedDistributionSVG/500/21/0.30      | (1, 0, 0)  | 0.58041876 | 0.48117423 | 0.65517241
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.60989589 | 0.60765362 | 0.63598326
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.62592116 | 0.53457729 | 0.65863454
# ResNet50V2_Stack3              | per frame (max) | SVG                                      | (2, 0, 0)  | 0.64756112 | 0.50304985 | 0.68750000
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.63352439 | 0.62341980 | 0.63745020
# ResNet50V2_Stack3              | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.52789800 | 0.45634460 | 0.65079365
# ResNet50V2_Stack3              | per frame (max) | BalancedDistributionSVG/500/21/0.30      | (2, 0, 0)  | 0.61761610 | 0.49279918 | 0.68067227
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.66931805 | 0.62998938 | 0.65591398
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.71142824 | 0.67591893 | 0.66666667
# ResNet50V2_Stack3              | per frame (sum) | SVG                                      | None       | 0.70745116 | 0.67040711 | 0.67281106
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.67224237 | 0.63541335 | 0.66310160
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.69821032 | 0.67572609 | 0.67924528
# ResNet50V2_Stack3              | per frame (sum) | BalancedDistributionSVG/500/21/0.30      | None       | 0.70125161 | 0.67463309 | 0.67317073
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.65001755 | 0.62233318 | 0.62910798
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.70207042 | 0.67315640 | 0.65934066
# ResNet50V2_Stack3              | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.68674699 | 0.65105710 | 0.65178571
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.65376067 | 0.62957207 | 0.63101604
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.68452451 | 0.66594322 | 0.65168539
# ResNet50V2_Stack3              | per frame (sum) | BalancedDistributionSVG/500/21/0.30      | (1, 1, 1)  | 0.68522634 | 0.66233990 | 0.66079295
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.62440051 | 0.60572431 | 0.62559242
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.69271260 | 0.65964805 | 0.65024631
# ResNet50V2_Stack3              | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.65843958 | 0.63235019 | 0.65000000
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.62978126 | 0.61107523 | 0.61940299
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.66346941 | 0.64666983 | 0.64591440
# ResNet50V2_Stack3              | per frame (sum) | BalancedDistributionSVG/500/21/0.30      | (2, 2, 2)  | 0.65668499 | 0.64025558 | 0.65822785
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.66931805 | 0.62998938 | 0.65591398
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.71142824 | 0.67591893 | 0.66666667
# ResNet50V2_Stack3              | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.70745116 | 0.67040711 | 0.67281106
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.67224237 | 0.63541335 | 0.66310160
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.69821032 | 0.67572609 | 0.67924528
# ResNet50V2_Stack3              | per frame (sum) | BalancedDistributionSVG/500/21/0.30      | (0, 1, 1)  | 0.70125161 | 0.67463309 | 0.67317073
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.66931805 | 0.62998938 | 0.65591398
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.71142824 | 0.67591893 | 0.66666667
# ResNet50V2_Stack3              | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.70745116 | 0.67040711 | 0.67281106
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.67224237 | 0.63541335 | 0.66310160
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.69821032 | 0.67572609 | 0.67924528
# ResNet50V2_Stack3              | per frame (sum) | BalancedDistributionSVG/500/21/0.30      | (0, 2, 2)  | 0.70125161 | 0.67463309 | 0.67317073
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.65001755 | 0.62233318 | 0.62910798
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.70207042 | 0.67315640 | 0.65934066
# ResNet50V2_Stack3              | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.68674699 | 0.65105710 | 0.65178571
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.65376067 | 0.62957207 | 0.63101604
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.68452451 | 0.66594322 | 0.65168539
# ResNet50V2_Stack3              | per frame (sum) | BalancedDistributionSVG/500/21/0.30      | (1, 0, 0)  | 0.68522634 | 0.66233990 | 0.66079295
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.62440051 | 0.60572431 | 0.62559242
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.69271260 | 0.65964805 | 0.65024631
# ResNet50V2_Stack3              | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.65843958 | 0.63235019 | 0.65000000
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.62978126 | 0.61107523 | 0.61940299
# ResNet50V2_Stack3              | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.66346941 | 0.64666983 | 0.64591440
# ResNet50V2_Stack3              | per frame (sum) | BalancedDistributionSVG/500/21/0.30      | (2, 0, 0)  | 0.65668499 | 0.64025558 | 0.65822785
# VGG16                          | per patch       | BalancedDistributionSVG/500/19/0.30      | None       | 0.93353015 | 0.45845023 | 0.47937672
# VGG16                          | per patch       | SpatialBin/SVG/0.20                      | None       | 0.96611158 | 0.52826167 | 0.58011869
# VGG16                          | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.96320773 | 0.50825691 | 0.56600442
# VGG16                          | per patch       | SVG                                      | None       | 0.94427631 | 0.45573353 | 0.47516663
# VGG16                          | per patch       | SpatialBin/SVG/0.50                      | None       | 0.96626974 | 0.53325355 | 0.58331375
# VGG16                          | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.95812722 | 0.49205232 | 0.55426621
# VGG16                          | per patch       | BalancedDistributionSVG/500/19/0.30      | (1, 1, 1)  | 0.95874158 | 0.54529418 | 0.55406746
# VGG16                          | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.96506773 | 0.48165401 | 0.55984556
# VGG16                          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.95786669 | 0.43925979 | 0.51121991
# VGG16                          | per patch       | SVG                                      | (1, 1, 1)  | 0.95810741 | 0.52632926 | 0.53574788
# VGG16                          | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.96750176 | 0.50554291 | 0.57655909
# VGG16                          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.95824593 | 0.43403708 | 0.51293103
# VGG16                          | per patch       | BalancedDistributionSVG/500/19/0.30      | (2, 2, 2)  | 0.94869663 | 0.50950374 | 0.50835073
# VGG16                          | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.94978589 | 0.38427530 | 0.49162731
# VGG16                          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.94167190 | 0.33373546 | 0.46034946
# VGG16                          | per patch       | SVG                                      | (2, 2, 2)  | 0.94734398 | 0.49263889 | 0.49410559
# VGG16                          | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.95236793 | 0.40958609 | 0.50628445
# VGG16                          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.94164285 | 0.32623690 | 0.46232007
# VGG16                          | per patch       | BalancedDistributionSVG/500/19/0.30      | (0, 1, 1)  | 0.95973540 | 0.56879065 | 0.57000785
# VGG16                          | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.96847000 | 0.54100879 | 0.58745713
# VGG16                          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.96446818 | 0.50386662 | 0.54185496
# VGG16                          | per patch       | SVG                                      | (0, 1, 1)  | 0.96160719 | 0.55783797 | 0.56278366
# VGG16                          | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.96985250 | 0.55339140 | 0.59921799
# VGG16                          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.96457695 | 0.50091075 | 0.54658885
# VGG16                          | per patch       | BalancedDistributionSVG/500/19/0.30      | (0, 2, 2)  | 0.96135107 | 0.60257076 | 0.60616342
# VGG16                          | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.96060794 | 0.49777247 | 0.56635319
# VGG16                          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.95604699 | 0.44431852 | 0.51441332
# VGG16                          | per patch       | SVG                                      | (0, 2, 2)  | 0.96308088 | 0.60043920 | 0.60244233
# VGG16                          | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.96239888 | 0.51337627 | 0.58502674
# VGG16                          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.95752516 | 0.45014837 | 0.53193633
# VGG16                          | per patch       | BalancedDistributionSVG/500/19/0.30      | (1, 0, 0)  | 0.94238709 | 0.45370988 | 0.48377445
# VGG16                          | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.96092257 | 0.46240357 | 0.54397468
# VGG16                          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.95538569 | 0.42139297 | 0.50372458
# VGG16                          | per patch       | SVG                                      | (1, 0, 0)  | 0.94542106 | 0.44099771 | 0.47048497
# VGG16                          | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.96234496 | 0.47931237 | 0.55655558
# VGG16                          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.95268860 | 0.41453632 | 0.49657747
# VGG16                          | per patch       | BalancedDistributionSVG/500/19/0.30      | (2, 0, 0)  | 0.92728366 | 0.39120032 | 0.42851100
# VGG16                          | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.94615874 | 0.38892633 | 0.48401196
# VGG16                          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.93835656 | 0.33869132 | 0.43008314
# VGG16                          | per patch       | SVG                                      | (2, 0, 0)  | 0.92896601 | 0.37539791 | 0.41302005
# VGG16                          | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.94809795 | 0.40746012 | 0.49668165
# VGG16                          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.93724455 | 0.33357919 | 0.43259827
# VGG16                          | per frame (max) | BalancedDistributionSVG/500/19/0.30      | None       | 0.73985261 | 0.73191944 | 0.67346939
# VGG16                          | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.77026553 | 0.73141212 | 0.71356784
# VGG16                          | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.77143526 | 0.76161558 | 0.68644068
# VGG16                          | per frame (max) | SVG                                      | None       | 0.73985261 | 0.73051514 | 0.68062827
# VGG16                          | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.77810270 | 0.75054544 | 0.71502591
# VGG16                          | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.77330682 | 0.76478300 | 0.69005848
# VGG16                          | per frame (max) | BalancedDistributionSVG/500/19/0.30      | (1, 1, 1)  | 0.77447655 | 0.75167108 | 0.70103093
# VGG16                          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.78558896 | 0.70692472 | 0.72558140
# VGG16                          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.79249035 | 0.72868429 | 0.72277228
# VGG16                          | per frame (max) | SVG                                      | (1, 1, 1)  | 0.76383203 | 0.73959620 | 0.70526316
# VGG16                          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.78757749 | 0.75299715 | 0.70718232
# VGG16                          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.80933443 | 0.75557168 | 0.74736842
# VGG16                          | per frame (max) | BalancedDistributionSVG/500/19/0.30      | (2, 2, 2)  | 0.77658206 | 0.77486288 | 0.71764706
# VGG16                          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.77085039 | 0.67917369 | 0.71717172
# VGG16                          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.81015323 | 0.71127344 | 0.74345550
# VGG16                          | per frame (max) | SVG                                      | (2, 2, 2)  | 0.76932975 | 0.77455936 | 0.68852459
# VGG16                          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.76979764 | 0.73025938 | 0.71957672
# VGG16                          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.82465785 | 0.72794140 | 0.75000000
# VGG16                          | per frame (max) | BalancedDistributionSVG/500/19/0.30      | (0, 1, 1)  | 0.74429758 | 0.74401992 | 0.67924528
# VGG16                          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.78593988 | 0.75441334 | 0.73195876
# VGG16                          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.80138028 | 0.78767840 | 0.71356784
# VGG16                          | per frame (max) | SVG                                      | (0, 1, 1)  | 0.74242601 | 0.73567778 | 0.67973856
# VGG16                          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.78804539 | 0.77681300 | 0.71276596
# VGG16                          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.80290092 | 0.79330842 | 0.72093023
# VGG16                          | per frame (max) | BalancedDistributionSVG/500/19/0.30      | (0, 2, 2)  | 0.76149257 | 0.77957257 | 0.68263473
# VGG16                          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.79904082 | 0.77311243 | 0.74074074
# VGG16                          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.81506609 | 0.80348982 | 0.73291925
# VGG16                          | per frame (max) | SVG                                      | (0, 2, 2)  | 0.75927009 | 0.77454675 | 0.68831169
# VGG16                          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.79085273 | 0.78095236 | 0.72340426
# VGG16                          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.81962803 | 0.81481790 | 0.73333333
# VGG16                          | per frame (max) | BalancedDistributionSVG/500/19/0.30      | (1, 0, 0)  | 0.76593754 | 0.72741345 | 0.72316384
# VGG16                          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.77482747 | 0.68898696 | 0.73913043
# VGG16                          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.77623114 | 0.72285512 | 0.71351351
# VGG16                          | per frame (max) | SVG                                      | (1, 0, 0)  | 0.75927009 | 0.71738411 | 0.71641791
# VGG16                          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.78055913 | 0.73008032 | 0.72527473
# VGG16                          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.79447889 | 0.72933747 | 0.76571429
# VGG16                          | per frame (max) | BalancedDistributionSVG/500/19/0.30      | (2, 0, 0)  | 0.75073108 | 0.69365356 | 0.73913043
# VGG16                          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.75540999 | 0.64352207 | 0.73118280
# VGG16                          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.78465318 | 0.69551158 | 0.73267327
# VGG16                          | per frame (max) | SVG                                      | (2, 0, 0)  | 0.74008656 | 0.67539378 | 0.72820513
# VGG16                          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.76827699 | 0.70929274 | 0.74594595
# VGG16                          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.79494678 | 0.69610605 | 0.75121951
# VGG16                          | per frame (sum) | BalancedDistributionSVG/500/19/0.30      | None       | 0.83027255 | 0.84707359 | 0.76000000
# VGG16                          | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.80582524 | 0.78575361 | 0.73333333
# VGG16                          | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.82313721 | 0.82708217 | 0.72483221
# VGG16                          | per frame (sum) | SVG                                      | None       | 0.82571061 | 0.84521374 | 0.76729560
# VGG16                          | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.80243303 | 0.78958504 | 0.72636816
# VGG16                          | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.82746520 | 0.84130385 | 0.75159236
# VGG16                          | per frame (sum) | BalancedDistributionSVG/500/19/0.30      | (1, 1, 1)  | 0.82547666 | 0.84072720 | 0.75949367
# VGG16                          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.79599953 | 0.76006970 | 0.73469388
# VGG16                          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.80874956 | 0.78262371 | 0.72251309
# VGG16                          | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.81951105 | 0.83785945 | 0.75471698
# VGG16                          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.79974266 | 0.78047063 | 0.71921182
# VGG16                          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.82208445 | 0.78726179 | 0.72636816
# VGG16                          | per frame (sum) | BalancedDistributionSVG/500/19/0.30      | (2, 2, 2)  | 0.80524038 | 0.80730992 | 0.72258065
# VGG16                          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.77482747 | 0.73082318 | 0.71356784
# VGG16                          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.79892385 | 0.74333608 | 0.73118280
# VGG16                          | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.79974266 | 0.80694063 | 0.72727273
# VGG16                          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.77938940 | 0.75044766 | 0.70707071
# VGG16                          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.81670371 | 0.73346091 | 0.73202614
# VGG16                          | per frame (sum) | BalancedDistributionSVG/500/19/0.30      | (0, 1, 1)  | 0.83027255 | 0.84707359 | 0.76000000
# VGG16                          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.80582524 | 0.78575361 | 0.73333333
# VGG16                          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.82313721 | 0.82708217 | 0.72483221
# VGG16                          | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.82571061 | 0.84521374 | 0.76729560
# VGG16                          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.80243303 | 0.78958504 | 0.72636816
# VGG16                          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.82746520 | 0.84130385 | 0.75159236
# VGG16                          | per frame (sum) | BalancedDistributionSVG/500/19/0.30      | (0, 2, 2)  | 0.83027255 | 0.84707359 | 0.76000000
# VGG16                          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.80582524 | 0.78575361 | 0.73333333
# VGG16                          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.82313721 | 0.82708217 | 0.72483221
# VGG16                          | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.82571061 | 0.84521374 | 0.76729560
# VGG16                          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.80243303 | 0.78958504 | 0.72636816
# VGG16                          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.82746520 | 0.84130385 | 0.75159236
# VGG16                          | per frame (sum) | BalancedDistributionSVG/500/19/0.30      | (1, 0, 0)  | 0.82547666 | 0.84072720 | 0.75949367
# VGG16                          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.79599953 | 0.76006970 | 0.73469388
# VGG16                          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.80874956 | 0.78262371 | 0.72251309
# VGG16                          | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.81951105 | 0.83785945 | 0.75471698
# VGG16                          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.79974266 | 0.78047063 | 0.71921182
# VGG16                          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.82208445 | 0.78726179 | 0.72636816
# VGG16                          | per frame (sum) | BalancedDistributionSVG/500/19/0.30      | (2, 0, 0)  | 0.80524038 | 0.80730992 | 0.72258065
# VGG16                          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.77482747 | 0.73082318 | 0.71356784
# VGG16                          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.79892385 | 0.74333608 | 0.73118280
# VGG16                          | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.79974266 | 0.80694063 | 0.72727273
# VGG16                          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.77938940 | 0.75044766 | 0.70707071
# VGG16                          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.81670371 | 0.73346091 | 0.73202614
# C3D_Block4                     | per patch       | BalancedDistributionSVG/500/19/0.30      | None       | 0.86016686 | 0.26731820 | 0.35822593
# C3D_Block4                     | per patch       | SpatialBin/SVG/0.20                      | None       | 0.91176977 | 0.34939774 | 0.47914343
# C3D_Block4                     | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.91576721 | 0.33634016 | 0.47508170
# C3D_Block4                     | per patch       | SVG                                      | None       | 0.87771266 | 0.27192167 | 0.35818277
# C3D_Block4                     | per patch       | SpatialBin/SVG/0.50                      | None       | 0.91234784 | 0.34311695 | 0.47791725
# C3D_Block4                     | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.91363944 | 0.34423284 | 0.46482248
# C3D_Block4                     | per patch       | BalancedDistributionSVG/500/19/0.30      | (1, 1, 1)  | 0.88738692 | 0.29523062 | 0.37250122
# C3D_Block4                     | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.93941129 | 0.37342049 | 0.50827188
# C3D_Block4                     | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.94867106 | 0.38156848 | 0.51959114
# C3D_Block4                     | per patch       | SVG                                      | (1, 1, 1)  | 0.89911381 | 0.29076371 | 0.36196752
# C3D_Block4                     | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.94101374 | 0.37887089 | 0.51354731
# C3D_Block4                     | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.95063554 | 0.39103490 | 0.52047351
# C3D_Block4                     | per patch       | BalancedDistributionSVG/500/19/0.30      | (2, 2, 2)  | 0.88913366 | 0.31549643 | 0.37188435
# C3D_Block4                     | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.93398634 | 0.33046745 | 0.46310803
# C3D_Block4                     | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.94889491 | 0.37148014 | 0.51457976
# C3D_Block4                     | per patch       | SVG                                      | (2, 2, 2)  | 0.89751339 | 0.30293601 | 0.36098070
# C3D_Block4                     | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.93779683 | 0.35177339 | 0.47755184
# C3D_Block4                     | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.94884893 | 0.36309379 | 0.50289017
# C3D_Block4                     | per patch       | BalancedDistributionSVG/500/19/0.30      | (0, 1, 1)  | 0.88135871 | 0.30298000 | 0.38669101
# C3D_Block4                     | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.93147288 | 0.38890877 | 0.52916389
# C3D_Block4                     | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.93680858 | 0.37474761 | 0.51826822
# C3D_Block4                     | per patch       | SVG                                      | (0, 1, 1)  | 0.89376949 | 0.29990997 | 0.38009823
# C3D_Block4                     | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.93213788 | 0.38159669 | 0.52964760
# C3D_Block4                     | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.93835235 | 0.38465926 | 0.51639685
# C3D_Block4                     | per patch       | BalancedDistributionSVG/500/19/0.30      | (0, 2, 2)  | 0.88823650 | 0.34241663 | 0.41613152
# C3D_Block4                     | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.93199001 | 0.38764518 | 0.52245233
# C3D_Block4                     | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.94074018 | 0.38404450 | 0.52251023
# C3D_Block4                     | per patch       | SVG                                      | (0, 2, 2)  | 0.89948563 | 0.33589018 | 0.40788072
# C3D_Block4                     | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.93352402 | 0.38331459 | 0.52102653
# C3D_Block4                     | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.93982538 | 0.37159429 | 0.51853386
# C3D_Block4                     | per patch       | BalancedDistributionSVG/500/19/0.30      | (1, 0, 0)  | 0.87191906 | 0.26318635 | 0.35752330
# C3D_Block4                     | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.92386146 | 0.34339085 | 0.47434783
# C3D_Block4                     | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.93005359 | 0.34427345 | 0.46721833
# C3D_Block4                     | per patch       | SVG                                      | (1, 0, 0)  | 0.88682133 | 0.26487951 | 0.35154730
# C3D_Block4                     | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.92488005 | 0.34547153 | 0.47884867
# C3D_Block4                     | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.93032310 | 0.35649524 | 0.46717374
# C3D_Block4                     | per patch       | BalancedDistributionSVG/500/19/0.30      | (2, 0, 0)  | 0.86905631 | 0.24947941 | 0.34509804
# C3D_Block4                     | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.91611416 | 0.30184465 | 0.42743741
# C3D_Block4                     | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.92337175 | 0.32024885 | 0.43884035
# C3D_Block4                     | per patch       | SVG                                      | (2, 0, 0)  | 0.88106006 | 0.24937140 | 0.33276949
# C3D_Block4                     | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.91774466 | 0.31162899 | 0.43449281
# C3D_Block4                     | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.92571929 | 0.33589172 | 0.43860004
# C3D_Block4                     | per frame (max) | BalancedDistributionSVG/500/19/0.30      | None       | 0.78944906 | 0.72193240 | 0.74033149
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.79880688 | 0.76584228 | 0.73118280
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.77798573 | 0.72684762 | 0.72514620
# C3D_Block4                     | per frame (max) | SVG                                      | None       | 0.78020821 | 0.71727208 | 0.72916667
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.78570593 | 0.73572507 | 0.71428571
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.83401567 | 0.80882983 | 0.76666667
# C3D_Block4                     | per frame (max) | BalancedDistributionSVG/500/19/0.30      | (1, 1, 1)  | 0.75845128 | 0.74029887 | 0.69613260
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.79120365 | 0.75549106 | 0.73743017
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.82044684 | 0.76121320 | 0.76439791
# C3D_Block4                     | per frame (max) | SVG                                      | (1, 1, 1)  | 0.75739853 | 0.74763134 | 0.68544601
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.80032752 | 0.75603214 | 0.74594595
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.86536437 | 0.82319256 | 0.80219780
# C3D_Block4                     | per frame (max) | BalancedDistributionSVG/500/19/0.30      | (2, 2, 2)  | 0.72815534 | 0.67324050 | 0.70351759
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.78816236 | 0.72003655 | 0.74468085
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.85588958 | 0.80955664 | 0.78494624
# C3D_Block4                     | per frame (max) | SVG                                      | (2, 2, 2)  | 0.70932273 | 0.65689022 | 0.67676768
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.77845362 | 0.72472632 | 0.72727273
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.89472453 | 0.85156755 | 0.82978723
# C3D_Block4                     | per frame (max) | BalancedDistributionSVG/500/19/0.30      | (0, 1, 1)  | 0.76429992 | 0.74066664 | 0.71794872
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.81646976 | 0.77566005 | 0.74468085
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.80348579 | 0.73097904 | 0.76756757
# C3D_Block4                     | per frame (max) | SVG                                      | (0, 1, 1)  | 0.76231138 | 0.74369261 | 0.70103093
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.80524038 | 0.75041535 | 0.74444444
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.84606387 | 0.80263511 | 0.79558011
# C3D_Block4                     | per frame (max) | BalancedDistributionSVG/500/19/0.30      | (0, 2, 2)  | 0.74862557 | 0.71642837 | 0.71038251
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.81132296 | 0.77565055 | 0.74025974
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.81576793 | 0.74508561 | 0.75268817
# C3D_Block4                     | per frame (max) | SVG                                      | (0, 2, 2)  | 0.73856591 | 0.70234243 | 0.70588235
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.79763715 | 0.75206868 | 0.73076923
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.84805240 | 0.80248884 | 0.79558011
# C3D_Block4                     | per frame (max) | BalancedDistributionSVG/500/19/0.30      | (1, 0, 0)  | 0.78710960 | 0.71826430 | 0.72432432
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.78453620 | 0.75727259 | 0.71186441
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.79623348 | 0.75661408 | 0.75000000
# C3D_Block4                     | per frame (max) | SVG                                      | (1, 0, 0)  | 0.78032518 | 0.70987040 | 0.71084337
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.77810270 | 0.74875567 | 0.73913043
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.85974968 | 0.83059042 | 0.80851064
# C3D_Block4                     | per frame (max) | BalancedDistributionSVG/500/19/0.30      | (2, 0, 0)  | 0.74862557 | 0.66927796 | 0.68539326
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.78289858 | 0.73362272 | 0.73575130
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.80395368 | 0.77410703 | 0.75376884
# C3D_Block4                     | per frame (max) | SVG                                      | (2, 0, 0)  | 0.75213475 | 0.68158419 | 0.67515924
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.76207744 | 0.72990348 | 0.70000000
# C3D_Block4                     | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.87671073 | 0.84590322 | 0.80198020
# C3D_Block4                     | per frame (sum) | BalancedDistributionSVG/500/19/0.30      | None       | 0.71142824 | 0.68358227 | 0.66960352
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.80477249 | 0.78838595 | 0.74074074
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.84313955 | 0.80766100 | 0.77528090
# C3D_Block4                     | per frame (sum) | SVG                                      | None       | 0.71388466 | 0.68103248 | 0.68181818
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.79260732 | 0.76381188 | 0.71856287
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.86045151 | 0.84356709 | 0.78160920
# C3D_Block4                     | per frame (sum) | BalancedDistributionSVG/500/19/0.30      | (1, 1, 1)  | 0.70710025 | 0.66871842 | 0.66972477
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.80021055 | 0.77512847 | 0.73417722
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.85939876 | 0.82231852 | 0.77192982
# C3D_Block4                     | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.70815300 | 0.66514385 | 0.68807339
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.79073576 | 0.76227487 | 0.70370370
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.87858229 | 0.84758907 | 0.80722892
# C3D_Block4                     | per frame (sum) | BalancedDistributionSVG/500/19/0.30      | (2, 2, 2)  | 0.69622178 | 0.64565037 | 0.67307692
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.78781144 | 0.75584290 | 0.72514620
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.85530471 | 0.82205913 | 0.79381443
# C3D_Block4                     | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.69692362 | 0.64312779 | 0.69456067
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.77202012 | 0.73799118 | 0.69767442
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.87671073 | 0.83579969 | 0.80924855
# C3D_Block4                     | per frame (sum) | BalancedDistributionSVG/500/19/0.30      | (0, 1, 1)  | 0.71142824 | 0.68358227 | 0.66960352
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.80477249 | 0.78838595 | 0.74074074
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.84313955 | 0.80766100 | 0.77528090
# C3D_Block4                     | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.71388466 | 0.68103248 | 0.68181818
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.79260732 | 0.76381188 | 0.71856287
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.86045151 | 0.84356709 | 0.78160920
# C3D_Block4                     | per frame (sum) | BalancedDistributionSVG/500/19/0.30      | (0, 2, 2)  | 0.71142824 | 0.68358227 | 0.66960352
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.80477249 | 0.78838595 | 0.74074074
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.84313955 | 0.80766100 | 0.77528090
# C3D_Block4                     | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.71388466 | 0.68103248 | 0.68181818
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.79260732 | 0.76381188 | 0.71856287
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.86045151 | 0.84356709 | 0.78160920
# C3D_Block4                     | per frame (sum) | BalancedDistributionSVG/500/19/0.30      | (1, 0, 0)  | 0.70710025 | 0.66871842 | 0.66972477
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.80021055 | 0.77512847 | 0.73417722
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.85939876 | 0.82231852 | 0.77192982
# C3D_Block4                     | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.70815300 | 0.66514385 | 0.68807339
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.79073576 | 0.76227487 | 0.70370370
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.87858229 | 0.84758907 | 0.80722892
# C3D_Block4                     | per frame (sum) | BalancedDistributionSVG/500/19/0.30      | (2, 0, 0)  | 0.69622178 | 0.64565037 | 0.67307692
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.78781144 | 0.75584290 | 0.72514620
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.85530471 | 0.82205913 | 0.79381443
# C3D_Block4                     | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.69692362 | 0.64312779 | 0.69456067
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.77202012 | 0.73799118 | 0.69767442
# C3D_Block4                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.87671073 | 0.83579969 | 0.80924855
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/0.20                      | None       | 0.54661446 | 0.04791396 | 0.11080067
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.57157739 | 0.05115725 | 0.11151257
# EfficientNetB0_Block5          | per patch       | SVG                                      | None       | 0.54457153 | 0.04774728 | 0.11266234
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/0.50                      | None       | 0.54645238 | 0.04792804 | 0.11082803
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.57437561 | 0.05152722 | 0.11192569
# EfficientNetB0_Block5          | per patch       | BalancedDistributionSVG/500/9/0.30       | None       | 0.57669506 | 0.05328077 | 0.11459839
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.54076190 | 0.04648925 | 0.11018467
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.56083770 | 0.04935441 | 0.11157691
# EfficientNetB0_Block5          | per patch       | SVG                                      | (1, 1, 1)  | 0.53760639 | 0.04608311 | 0.11063249
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.54075659 | 0.04649733 | 0.11011375
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.55880144 | 0.04889173 | 0.11161485
# EfficientNetB0_Block5          | per patch       | BalancedDistributionSVG/500/9/0.30       | (1, 1, 1)  | 0.57707998 | 0.05064392 | 0.11582281
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.51596023 | 0.04421117 | 0.10181868
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.54037019 | 0.04778527 | 0.10171458
# EfficientNetB0_Block5          | per patch       | SVG                                      | (2, 2, 2)  | 0.50961215 | 0.04347898 | 0.10120516
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.51610847 | 0.04422747 | 0.10182019
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.53428834 | 0.04656757 | 0.10183609
# EfficientNetB0_Block5          | per patch       | BalancedDistributionSVG/500/9/0.30       | (2, 2, 2)  | 0.54957774 | 0.04724574 | 0.11316360
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.54099525 | 0.04666005 | 0.10855484
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.56696579 | 0.05033448 | 0.11097673
# EfficientNetB0_Block5          | per patch       | SVG                                      | (0, 1, 1)  | 0.53783356 | 0.04626407 | 0.10963828
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.54085031 | 0.04665777 | 0.10849180
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.56673215 | 0.05016226 | 0.11105406
# EfficientNetB0_Block5          | per patch       | BalancedDistributionSVG/500/9/0.30       | (0, 1, 1)  | 0.57797468 | 0.05132955 | 0.11503255
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.52541592 | 0.04521281 | 0.10393712
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.56099527 | 0.05049568 | 0.10546591
# EfficientNetB0_Block5          | per patch       | SVG                                      | (0, 2, 2)  | 0.51759994 | 0.04432438 | 0.10383147
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.52539675 | 0.04520969 | 0.10386316
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.55914697 | 0.04979040 | 0.10589694
# EfficientNetB0_Block5          | per patch       | BalancedDistributionSVG/500/9/0.30       | (0, 2, 2)  | 0.56173039 | 0.05003438 | 0.10726862
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.54654851 | 0.04777140 | 0.11170340
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.56826179 | 0.05049384 | 0.11266861
# EfficientNetB0_Block5          | per patch       | SVG                                      | (1, 0, 0)  | 0.54414933 | 0.04760545 | 0.11323047
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.54650728 | 0.04779343 | 0.11165049
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.57026641 | 0.05067262 | 0.11280006
# EfficientNetB0_Block5          | per patch       | BalancedDistributionSVG/500/9/0.30       | (1, 0, 0)  | 0.57533875 | 0.05238768 | 0.11585396
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.54311816 | 0.04746689 | 0.11153260
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.56338212 | 0.04999584 | 0.11119106
# EfficientNetB0_Block5          | per patch       | SVG                                      | (2, 0, 0)  | 0.54120879 | 0.04734925 | 0.11275988
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.54316870 | 0.04750110 | 0.11144108
# EfficientNetB0_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.56500929 | 0.05004045 | 0.11164119
# EfficientNetB0_Block5          | per patch       | BalancedDistributionSVG/500/9/0.30       | (2, 0, 0)  | 0.56650609 | 0.05099162 | 0.11342210
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.71189613 | 0.70775851 | 0.67579909
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.49912270 | 0.40942391 | 0.64406780
# EfficientNetB0_Block5          | per frame (max) | SVG                                      | None       | 0.66966897 | 0.56143622 | 0.66666667
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.70920576 | 0.70338774 | 0.67213115
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.42215464 | 0.36937004 | 0.63358779
# EfficientNetB0_Block5          | per frame (max) | BalancedDistributionSVG/500/9/0.30       | None       | 0.67154053 | 0.56682920 | 0.67307692
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.70487776 | 0.70046465 | 0.68468468
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.49304012 | 0.40084782 | 0.64935065
# EfficientNetB0_Block5          | per frame (max) | SVG                                      | (1, 1, 1)  | 0.66452217 | 0.56311082 | 0.67307692
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.72406129 | 0.70408206 | 0.71559633
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.40823488 | 0.36410888 | 0.63076923
# EfficientNetB0_Block5          | per frame (max) | BalancedDistributionSVG/500/9/0.30       | (1, 1, 1)  | 0.66662768 | 0.56968853 | 0.67317073
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.67692128 | 0.68714833 | 0.66666667
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.51854018 | 0.42010583 | 0.64935065
# EfficientNetB0_Block5          | per frame (max) | SVG                                      | (2, 2, 2)  | 0.63446017 | 0.59986782 | 0.64150943
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.68288689 | 0.67552310 | 0.67521368
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.43782899 | 0.38261335 | 0.62903226
# EfficientNetB0_Block5          | per frame (max) | BalancedDistributionSVG/500/9/0.30       | (2, 2, 2)  | 0.63270558 | 0.59868391 | 0.64114833
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.71493742 | 0.71123029 | 0.68122271
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.49023278 | 0.40015795 | 0.65020576
# EfficientNetB0_Block5          | per frame (max) | SVG                                      | (0, 1, 1)  | 0.67949468 | 0.56690116 | 0.67619048
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.72230670 | 0.71413177 | 0.68778281
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.43794596 | 0.37535770 | 0.63358779
# EfficientNetB0_Block5          | per frame (max) | BalancedDistributionSVG/500/9/0.30       | (0, 1, 1)  | 0.68136624 | 0.57061797 | 0.67647059
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.71832963 | 0.70762475 | 0.67248908
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.50532226 | 0.41041441 | 0.65020576
# EfficientNetB0_Block5          | per frame (max) | SVG                                      | (0, 2, 2)  | 0.68007954 | 0.59707884 | 0.67647059
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.72183881 | 0.70549139 | 0.67942584
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.46028775 | 0.38562370 | 0.63813230
# EfficientNetB0_Block5          | per frame (max) | BalancedDistributionSVG/500/9/0.30       | (0, 2, 2)  | 0.67949468 | 0.59642141 | 0.67317073
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.70674933 | 0.70070601 | 0.68837209
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.46321207 | 0.38707920 | 0.65020576
# EfficientNetB0_Block5          | per frame (max) | SVG                                      | (1, 0, 0)  | 0.65399462 | 0.55544801 | 0.65714286
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.71505439 | 0.69833322 | 0.69483568
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.35793660 | 0.34619366 | 0.62641509
# EfficientNetB0_Block5          | per frame (max) | BalancedDistributionSVG/500/9/0.30       | (1, 0, 0)  | 0.65902445 | 0.56792722 | 0.65714286
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.68136624 | 0.68638522 | 0.66101695
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.45057902 | 0.38024307 | 0.64435146
# EfficientNetB0_Block5          | per frame (max) | SVG                                      | (2, 0, 0)  | 0.62826062 | 0.57547449 | 0.63636364
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.68663002 | 0.67428392 | 0.68468468
# EfficientNetB0_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.31138145 | 0.33718076 | 0.62641509
# EfficientNetB0_Block5          | per frame (max) | BalancedDistributionSVG/500/9/0.30       | (2, 0, 0)  | 0.63176980 | 0.58458860 | 0.63849765
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.68733185 | 0.67496530 | 0.65116279
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.62802667 | 0.55520536 | 0.65740741
# EfficientNetB0_Block5          | per frame (sum) | SVG                                      | None       | 0.65247397 | 0.63169122 | 0.65873016
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.68452451 | 0.67124157 | 0.65116279
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.58813896 | 0.50387707 | 0.64912281
# EfficientNetB0_Block5          | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | None       | 0.66358638 | 0.63480279 | 0.66122449
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.68265294 | 0.67490308 | 0.64341085
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.61609545 | 0.56262312 | 0.62385321
# EfficientNetB0_Block5          | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.64206340 | 0.65990622 | 0.64822134
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.68288689 | 0.67310451 | 0.65354331
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.57199672 | 0.51877017 | 0.63076923
# EfficientNetB0_Block5          | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | (1, 1, 1)  | 0.65329278 | 0.66385579 | 0.65079365
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.64943268 | 0.67132912 | 0.63846154
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.59480641 | 0.57439783 | 0.61710037
# EfficientNetB0_Block5          | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.60346239 | 0.66093523 | 0.64062500
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.65703591 | 0.67317496 | 0.63846154
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.55924670 | 0.51825096 | 0.62121212
# EfficientNetB0_Block5          | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | (2, 2, 2)  | 0.60977892 | 0.66125809 | 0.63565891
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.68733185 | 0.67496530 | 0.65116279
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.62802667 | 0.55520536 | 0.65740741
# EfficientNetB0_Block5          | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.65247397 | 0.63169122 | 0.65873016
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.68452451 | 0.67124157 | 0.65116279
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.58813896 | 0.50387707 | 0.64912281
# EfficientNetB0_Block5          | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | (0, 1, 1)  | 0.66358638 | 0.63480279 | 0.66122449
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.68733185 | 0.67496530 | 0.65116279
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.62802667 | 0.55520536 | 0.65740741
# EfficientNetB0_Block5          | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.65247397 | 0.63169122 | 0.65873016
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.68452451 | 0.67124157 | 0.65116279
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.58813896 | 0.50387707 | 0.64912281
# EfficientNetB0_Block5          | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | (0, 2, 2)  | 0.66358638 | 0.63480279 | 0.66122449
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.68265294 | 0.67490308 | 0.64341085
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.61609545 | 0.56262312 | 0.62385321
# EfficientNetB0_Block5          | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.64206340 | 0.65990622 | 0.64822134
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.68288689 | 0.67310451 | 0.65354331
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.57199672 | 0.51877017 | 0.63076923
# EfficientNetB0_Block5          | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | (1, 0, 0)  | 0.65329278 | 0.66385579 | 0.65079365
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.64943268 | 0.67132912 | 0.63846154
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.59480641 | 0.57439783 | 0.61710037
# EfficientNetB0_Block5          | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.60346239 | 0.66093523 | 0.64062500
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.65703591 | 0.67317496 | 0.63846154
# EfficientNetB0_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.55924670 | 0.51825096 | 0.62121212
# EfficientNetB0_Block5          | per frame (sum) | BalancedDistributionSVG/500/9/0.30       | (2, 0, 0)  | 0.60977892 | 0.66125809 | 0.63565891
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/0.20                      | None       | 0.58405780 | 0.06281985 | 0.13283917
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.57533011 | 0.06027700 | 0.11410142
# EfficientNetB3_Block4          | per patch       | SVG                                      | None       | 0.56030915 | 0.07016047 | 0.14631840
# EfficientNetB3_Block4          | per patch       | BalancedDistributionSVG/500/7/0.30       | None       | 0.54319659 | 0.06659621 | 0.14696862
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/0.50                      | None       | 0.58480020 | 0.06297913 | 0.13286371
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.57481899 | 0.06059567 | 0.11396146
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.58643551 | 0.06343038 | 0.13233288
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.57989319 | 0.06161155 | 0.11807593
# EfficientNetB3_Block4          | per patch       | SVG                                      | (1, 1, 1)  | 0.58118068 | 0.07494992 | 0.15087775
# EfficientNetB3_Block4          | per patch       | BalancedDistributionSVG/500/7/0.30       | (1, 1, 1)  | 0.59738692 | 0.07346058 | 0.14895330
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.58786114 | 0.06398377 | 0.13290272
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.57206535 | 0.05906668 | 0.11580036
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.59102011 | 0.06527027 | 0.13893173
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.56747414 | 0.05816352 | 0.11531715
# EfficientNetB3_Block4          | per patch       | SVG                                      | (2, 2, 2)  | 0.60098910 | 0.07353041 | 0.14733297
# EfficientNetB3_Block4          | per patch       | BalancedDistributionSVG/500/7/0.30       | (2, 2, 2)  | 0.62708971 | 0.07613152 | 0.14870115
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.59243522 | 0.06587274 | 0.14069342
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.56214799 | 0.05543610 | 0.11469405
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.58671387 | 0.06386053 | 0.13296779
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.57573692 | 0.05982337 | 0.11512718
# EfficientNetB3_Block4          | per patch       | SVG                                      | (0, 1, 1)  | 0.57812920 | 0.07509934 | 0.15128205
# EfficientNetB3_Block4          | per patch       | BalancedDistributionSVG/500/7/0.30       | (0, 1, 1)  | 0.59470964 | 0.07321477 | 0.14945452
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.58791662 | 0.06433150 | 0.13295477
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.57676930 | 0.06131692 | 0.11572456
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.59205145 | 0.06565916 | 0.13929857
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.57740567 | 0.06030362 | 0.12285382
# EfficientNetB3_Block4          | per patch       | SVG                                      | (0, 2, 2)  | 0.60023893 | 0.07316414 | 0.14801921
# EfficientNetB3_Block4          | per patch       | BalancedDistributionSVG/500/7/0.30       | (0, 2, 2)  | 0.62527294 | 0.07574739 | 0.14922118
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.59363075 | 0.06637572 | 0.14064959
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.57987478 | 0.06230709 | 0.12523749
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.58455323 | 0.06261190 | 0.13252777
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.58160917 | 0.06310458 | 0.12092732
# EfficientNetB3_Block4          | per patch       | SVG                                      | (1, 0, 0)  | 0.56510334 | 0.07018867 | 0.14608981
# EfficientNetB3_Block4          | per patch       | BalancedDistributionSVG/500/7/0.30       | (1, 0, 0)  | 0.54541924 | 0.06674783 | 0.14666667
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.58539404 | 0.06265390 | 0.13270791
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.57251996 | 0.05966580 | 0.11192493
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.58391718 | 0.06254862 | 0.13254185
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.57694004 | 0.06134714 | 0.11624913
# EfficientNetB3_Block4          | per patch       | SVG                                      | (2, 0, 0)  | 0.56901129 | 0.06981892 | 0.14413989
# EfficientNetB3_Block4          | per patch       | BalancedDistributionSVG/500/7/0.30       | (2, 0, 0)  | 0.54798532 | 0.06659181 | 0.14507173
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.58464837 | 0.06239289 | 0.13257835
# EfficientNetB3_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.56937446 | 0.05845594 | 0.10881881
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.47947128 | 0.40515368 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.51000117 | 0.42955539 | 0.62172285
# EfficientNetB3_Block4          | per frame (max) | SVG                                      | None       | 0.45443912 | 0.41107059 | 0.62878788
# EfficientNetB3_Block4          | per frame (max) | BalancedDistributionSVG/500/7/0.30       | None       | 0.43279916 | 0.39791110 | 0.62172285
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.47678091 | 0.41077373 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.49795298 | 0.41874386 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.45900105 | 0.39218316 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.51503100 | 0.43398516 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SVG                                      | (1, 1, 1)  | 0.46835887 | 0.41651806 | 0.63117871
# EfficientNetB3_Block4          | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (1, 1, 1)  | 0.56006550 | 0.50086257 | 0.66942149
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.47572816 | 0.41061668 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.50345070 | 0.43328020 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.48485203 | 0.40584661 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.47982220 | 0.41500612 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SVG                                      | (2, 2, 2)  | 0.56029945 | 0.51785519 | 0.66135458
# EfficientNetB3_Block4          | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (2, 2, 2)  | 0.56801965 | 0.51104004 | 0.69456067
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.47759972 | 0.40914265 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.46718914 | 0.42073017 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.47584513 | 0.40221268 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.50836355 | 0.42864706 | 0.62172285
# EfficientNetB3_Block4          | per frame (max) | SVG                                      | (0, 1, 1)  | 0.46438180 | 0.41435186 | 0.63117871
# EfficientNetB3_Block4          | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (0, 1, 1)  | 0.55924670 | 0.49213464 | 0.66108787
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.47666394 | 0.40944909 | 0.61940299
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.48789332 | 0.40998559 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.48075798 | 0.40407334 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.51257457 | 0.42978165 | 0.61940299
# EfficientNetB3_Block4          | per frame (max) | SVG                                      | (0, 2, 2)  | 0.52789800 | 0.49874038 | 0.63601533
# EfficientNetB3_Block4          | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (0, 2, 2)  | 0.55842789 | 0.50525479 | 0.65271967
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.48660662 | 0.41335569 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.49982454 | 0.41625434 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.46601942 | 0.39564756 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.50824658 | 0.42814102 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SVG                                      | (1, 0, 0)  | 0.44344368 | 0.39613108 | 0.62878788
# EfficientNetB3_Block4          | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (1, 0, 0)  | 0.41653995 | 0.37949305 | 0.62641509
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.47058135 | 0.40767036 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.49420985 | 0.42724797 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.49385893 | 0.41198465 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.45163177 | 0.39829182 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SVG                                      | (2, 0, 0)  | 0.43116154 | 0.39012835 | 0.63117871
# EfficientNetB3_Block4          | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (2, 0, 0)  | 0.40788396 | 0.37640677 | 0.62878788
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.46461574 | 0.40589961 | 0.61710037
# EfficientNetB3_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.45081296 | 0.41502442 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.54123289 | 0.46684575 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.52895075 | 0.44169999 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SVG                                      | None       | 0.48344836 | 0.48062462 | 0.61886792
# EfficientNetB3_Block4          | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | None       | 0.34834484 | 0.40615338 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.54450813 | 0.47780704 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.50099427 | 0.42117516 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.50988420 | 0.43045129 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.52649433 | 0.44274836 | 0.61855670
# EfficientNetB3_Block4          | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.51807229 | 0.50562169 | 0.62878788
# EfficientNetB3_Block4          | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (1, 1, 1)  | 0.35103521 | 0.40921518 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.54170078 | 0.47206766 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.50239794 | 0.43379656 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.52076266 | 0.43437086 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.47923734 | 0.41781283 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.53772371 | 0.51122625 | 0.63117871
# EfficientNetB3_Block4          | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (2, 2, 2)  | 0.36998479 | 0.41745608 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.53982922 | 0.47587154 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.46251024 | 0.41985372 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.54123289 | 0.46684575 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.52895075 | 0.44169999 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.48344836 | 0.48062462 | 0.61886792
# EfficientNetB3_Block4          | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (0, 1, 1)  | 0.34834484 | 0.40615338 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.54450813 | 0.47780704 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.50099427 | 0.42117516 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.54123289 | 0.46684575 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.52895075 | 0.44169999 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.48344836 | 0.48062462 | 0.61886792
# EfficientNetB3_Block4          | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (0, 2, 2)  | 0.34834484 | 0.40615338 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.54450813 | 0.47780704 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.50099427 | 0.42117516 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.50988420 | 0.43045129 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.52649433 | 0.44274836 | 0.61855670
# EfficientNetB3_Block4          | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.51807229 | 0.50562169 | 0.62878788
# EfficientNetB3_Block4          | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (1, 0, 0)  | 0.35103521 | 0.40921518 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.54170078 | 0.47206766 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.50239794 | 0.43379656 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.52076266 | 0.43437086 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.47923734 | 0.41781283 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.53772371 | 0.51122625 | 0.63117871
# EfficientNetB3_Block4          | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (2, 0, 0)  | 0.36998479 | 0.41745608 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.53982922 | 0.47587154 | 0.61710037
# EfficientNetB3_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.46251024 | 0.41985372 | 0.61710037
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/0.20                      | None       | 0.56528895 | 0.05249900 | 0.10875352
# EfficientNetB6_Block3          | per patch       | BalancedDistributionSVG/500/5/0.30       | None       | 0.52557990 | 0.05453049 | 0.12679858
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.50733831 | 0.04756356 | 0.09408524
# EfficientNetB6_Block3          | per patch       | SVG                                      | None       | 0.55495542 | 0.05377340 | 0.13247181
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/0.50                      | None       | 0.56602863 | 0.05262555 | 0.10970853
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.50172577 | 0.04708444 | 0.09263167
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.56187861 | 0.05151099 | 0.10765703
# EfficientNetB6_Block3          | per patch       | BalancedDistributionSVG/500/5/0.30       | (1, 1, 1)  | 0.52241364 | 0.05122177 | 0.11917719
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.51078928 | 0.04791373 | 0.09436281
# EfficientNetB6_Block3          | per patch       | SVG                                      | (1, 1, 1)  | 0.55251069 | 0.05256186 | 0.12303546
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.56273835 | 0.05159898 | 0.10834190
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.50592254 | 0.04772807 | 0.09317874
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.55659197 | 0.05057003 | 0.10538611
# EfficientNetB6_Block3          | per patch       | BalancedDistributionSVG/500/5/0.30       | (2, 2, 2)  | 0.52338737 | 0.05068023 | 0.11025818
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.51303516 | 0.04785036 | 0.09405412
# EfficientNetB6_Block3          | per patch       | SVG                                      | (2, 2, 2)  | 0.54194823 | 0.05025849 | 0.11172341
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.55700327 | 0.05061062 | 0.10567475
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.50817517 | 0.04812443 | 0.09334938
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.56216577 | 0.05158734 | 0.10785096
# EfficientNetB6_Block3          | per patch       | BalancedDistributionSVG/500/5/0.30       | (0, 1, 1)  | 0.52976968 | 0.05201051 | 0.12367330
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.50769177 | 0.04741286 | 0.09323900
# EfficientNetB6_Block3          | per patch       | SVG                                      | (0, 1, 1)  | 0.55235397 | 0.05276623 | 0.12630382
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.56307755 | 0.05170681 | 0.10856763
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.50232712 | 0.04696490 | 0.09179780
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.55669976 | 0.05061407 | 0.10562041
# EfficientNetB6_Block3          | per patch       | BalancedDistributionSVG/500/5/0.30       | (0, 2, 2)  | 0.52812791 | 0.05136610 | 0.11548283
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.51017418 | 0.04725955 | 0.09347365
# EfficientNetB6_Block3          | per patch       | SVG                                      | (0, 2, 2)  | 0.54338357 | 0.05062127 | 0.11618934
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.55737787 | 0.05071621 | 0.10592529
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.50521471 | 0.04690438 | 0.09230007
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.56514616 | 0.05241178 | 0.10842205
# EfficientNetB6_Block3          | per patch       | BalancedDistributionSVG/500/5/0.30       | (1, 0, 0)  | 0.51959088 | 0.05378904 | 0.12193378
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.51007799 | 0.04797173 | 0.09528505
# EfficientNetB6_Block3          | per patch       | SVG                                      | (1, 0, 0)  | 0.55642878 | 0.05369008 | 0.12966558
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.56596035 | 0.05252127 | 0.10947371
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.50504854 | 0.04773477 | 0.09381944
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.56523476 | 0.05240702 | 0.10818618
# EfficientNetB6_Block3          | per patch       | BalancedDistributionSVG/500/5/0.30       | (2, 0, 0)  | 0.51748857 | 0.05322779 | 0.11786226
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.50937750 | 0.04789352 | 0.09432344
# EfficientNetB6_Block3          | per patch       | SVG                                      | (2, 0, 0)  | 0.55478038 | 0.05328387 | 0.12606217
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.56597310 | 0.05250937 | 0.10890978
# EfficientNetB6_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.50383474 | 0.04820961 | 0.09324627
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.47689788 | 0.39360001 | 0.64705882
# EfficientNetB6_Block3          | per frame (max) | BalancedDistributionSVG/500/5/0.30       | None       | 0.37922564 | 0.37702467 | 0.61710037
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.37653527 | 0.36194342 | 0.62641509
# EfficientNetB6_Block3          | per frame (max) | SVG                                      | None       | 0.37828986 | 0.37668681 | 0.61710037
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.51128787 | 0.42612892 | 0.63846154
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.46964557 | 0.41232248 | 0.63492063
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.48075798 | 0.39977755 | 0.62641509
# EfficientNetB6_Block3          | per frame (max) | BalancedDistributionSVG/500/5/0.30       | (1, 1, 1)  | 0.41349865 | 0.39652966 | 0.61710037
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.44484735 | 0.40978051 | 0.63117871
# EfficientNetB6_Block3          | per frame (max) | SVG                                      | (1, 1, 1)  | 0.40495964 | 0.39271537 | 0.61710037
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.51994385 | 0.43555173 | 0.65306122
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.48508597 | 0.43890234 | 0.63601533
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.51935899 | 0.42674920 | 0.63241107
# EfficientNetB6_Block3          | per frame (max) | BalancedDistributionSVG/500/5/0.30       | (2, 2, 2)  | 0.42835419 | 0.40160271 | 0.61710037
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.47701486 | 0.41672563 | 0.62548263
# EfficientNetB6_Block3          | per frame (max) | SVG                                      | (2, 2, 2)  | 0.41700784 | 0.39955040 | 0.61710037
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.53304480 | 0.45399810 | 0.63358779
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.49818692 | 0.44207816 | 0.63320463
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.48087496 | 0.39892575 | 0.62878788
# EfficientNetB6_Block3          | per frame (max) | BalancedDistributionSVG/500/5/0.30       | (0, 1, 1)  | 0.39431512 | 0.38559768 | 0.62172285
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.37571646 | 0.36311442 | 0.63117871
# EfficientNetB6_Block3          | per frame (max) | SVG                                      | (0, 1, 1)  | 0.38753071 | 0.38036042 | 0.61940299
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.51725348 | 0.43382586 | 0.65000000
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.43993450 | 0.39946822 | 0.63565891
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.50064335 | 0.41037905 | 0.63846154
# EfficientNetB6_Block3          | per frame (max) | BalancedDistributionSVG/500/5/0.30       | (0, 2, 2)  | 0.43934963 | 0.44160382 | 0.61710037
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.41069131 | 0.37612248 | 0.63601533
# EfficientNetB6_Block3          | per frame (max) | SVG                                      | (0, 2, 2)  | 0.37723710 | 0.37573441 | 0.61710037
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.54287051 | 0.46169432 | 0.64313725
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.43268219 | 0.39660743 | 0.63846154
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.47724880 | 0.39483931 | 0.64135021
# EfficientNetB6_Block3          | per frame (max) | BalancedDistributionSVG/500/5/0.30       | (1, 0, 0)  | 0.39232659 | 0.38340189 | 0.61710037
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.44976021 | 0.41512883 | 0.61940299
# EfficientNetB6_Block3          | per frame (max) | SVG                                      | (1, 0, 0)  | 0.39150778 | 0.38300189 | 0.61710037
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.51655164 | 0.42950640 | 0.63934426
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.49467774 | 0.44438624 | 0.63813230
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.47947128 | 0.39638652 | 0.63745020
# EfficientNetB6_Block3          | per frame (max) | BalancedDistributionSVG/500/5/0.30       | (2, 0, 0)  | 0.41794362 | 0.39479561 | 0.61710037
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.46625336 | 0.41892890 | 0.61940299
# EfficientNetB6_Block3          | per frame (max) | SVG                                      | (2, 0, 0)  | 0.41747573 | 0.39460563 | 0.61710037
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.51444613 | 0.42695670 | 0.64166667
# EfficientNetB6_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.51409522 | 0.45147812 | 0.63529412
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.51128787 | 0.42181889 | 0.64406780
# EfficientNetB6_Block3          | per frame (sum) | BalancedDistributionSVG/500/5/0.30       | None       | 0.42566382 | 0.41719598 | 0.63076923
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.33477600 | 0.34553421 | 0.61710037
# EfficientNetB6_Block3          | per frame (sum) | SVG                                      | None       | 0.52754708 | 0.51410032 | 0.63358779
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.51725348 | 0.43897662 | 0.63333333
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.38495730 | 0.37131910 | 0.61710037
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.51807229 | 0.43072555 | 0.64705882
# EfficientNetB6_Block3          | per frame (sum) | BalancedDistributionSVG/500/5/0.30       | (1, 1, 1)  | 0.40133349 | 0.40641330 | 0.63117871
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.41911335 | 0.38901494 | 0.61710037
# EfficientNetB6_Block3          | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.51070301 | 0.51065660 | 0.63358779
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.52204936 | 0.44774721 | 0.63673469
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.43712715 | 0.40960299 | 0.61710037
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.51830623 | 0.43623492 | 0.63114754
# EfficientNetB6_Block3          | per frame (sum) | BalancedDistributionSVG/500/5/0.30       | (2, 2, 2)  | 0.38951924 | 0.38489534 | 0.63358779
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.44157211 | 0.39468755 | 0.61710037
# EfficientNetB6_Block3          | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.48438414 | 0.50627898 | 0.63565891
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.52017780 | 0.45366090 | 0.62151394
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.45981986 | 0.41472566 | 0.61710037
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.51128787 | 0.42181889 | 0.64406780
# EfficientNetB6_Block3          | per frame (sum) | BalancedDistributionSVG/500/5/0.30       | (0, 1, 1)  | 0.42566382 | 0.41719598 | 0.63076923
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.33477600 | 0.34553421 | 0.61710037
# EfficientNetB6_Block3          | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.52754708 | 0.51410032 | 0.63358779
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.51725348 | 0.43897662 | 0.63333333
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.38495730 | 0.37131910 | 0.61710037
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.51128787 | 0.42181889 | 0.64406780
# EfficientNetB6_Block3          | per frame (sum) | BalancedDistributionSVG/500/5/0.30       | (0, 2, 2)  | 0.42566382 | 0.41719598 | 0.63076923
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.33477600 | 0.34553421 | 0.61710037
# EfficientNetB6_Block3          | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.52754708 | 0.51410032 | 0.63358779
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.51725348 | 0.43897662 | 0.63333333
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.38495730 | 0.37131910 | 0.61710037
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.51807229 | 0.43072555 | 0.64705882
# EfficientNetB6_Block3          | per frame (sum) | BalancedDistributionSVG/500/5/0.30       | (1, 0, 0)  | 0.40133349 | 0.40641330 | 0.63117871
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.41911335 | 0.38901494 | 0.61710037
# EfficientNetB6_Block3          | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.51070301 | 0.51065660 | 0.63358779
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.52204936 | 0.44774721 | 0.63673469
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.43712715 | 0.40960299 | 0.61710037
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.51830623 | 0.43623492 | 0.63114754
# EfficientNetB6_Block3          | per frame (sum) | BalancedDistributionSVG/500/5/0.30       | (2, 0, 0)  | 0.38951924 | 0.38489534 | 0.63358779
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.44157211 | 0.39468755 | 0.61710037
# EfficientNetB6_Block3          | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.48438414 | 0.50627898 | 0.63565891
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.52017780 | 0.45366090 | 0.62151394
# EfficientNetB6_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.45981986 | 0.41472566 | 0.61710037
# MobileNetV2                    | per patch       | SpatialBin/SVG/0.20                      | None       | 0.94501034 | 0.41856300 | 0.50999049
# MobileNetV2                    | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.95132816 | 0.49843698 | 0.54754441
# MobileNetV2                    | per patch       | SVG                                      | None       | 0.88287414 | 0.27643609 | 0.32540862
# MobileNetV2                    | per patch       | BalancedDistributionSVG/500/31/0.30      | None       | 0.88983295 | 0.29224138 | 0.35085008
# MobileNetV2                    | per patch       | SpatialBin/SVG/0.50                      | None       | 0.94231251 | 0.39344648 | 0.49248120
# MobileNetV2                    | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.94768942 | 0.52296930 | 0.57086614
# MobileNetV2                    | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.95183122 | 0.39794483 | 0.51551724
# MobileNetV2                    | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.95967028 | 0.49873956 | 0.55612770
# MobileNetV2                    | per patch       | SVG                                      | (1, 1, 1)  | 0.89915589 | 0.31732968 | 0.40612245
# MobileNetV2                    | per patch       | BalancedDistributionSVG/500/31/0.30      | (1, 1, 1)  | 0.91277624 | 0.34711416 | 0.42898551
# MobileNetV2                    | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.94637505 | 0.36999557 | 0.48793103
# MobileNetV2                    | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.96568000 | 0.53552726 | 0.58000000
# MobileNetV2                    | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.92825164 | 0.31213116 | 0.43812709
# MobileNetV2                    | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.93440837 | 0.42706501 | 0.47131509
# MobileNetV2                    | per patch       | SVG                                      | (2, 2, 2)  | 0.87961715 | 0.31632074 | 0.39795918
# MobileNetV2                    | per patch       | BalancedDistributionSVG/500/31/0.30      | (2, 2, 2)  | 0.89759274 | 0.34345340 | 0.40808081
# MobileNetV2                    | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.92149797 | 0.29245352 | 0.42218543
# MobileNetV2                    | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.94061546 | 0.42863394 | 0.47591522
# MobileNetV2                    | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.95412601 | 0.42296209 | 0.52663317
# MobileNetV2                    | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.95558606 | 0.49351523 | 0.54752852
# MobileNetV2                    | per patch       | SVG                                      | (0, 1, 1)  | 0.90245225 | 0.34470199 | 0.40601504
# MobileNetV2                    | per patch       | BalancedDistributionSVG/500/31/0.30      | (0, 1, 1)  | 0.91213887 | 0.36956183 | 0.42549372
# MobileNetV2                    | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.94922016 | 0.38991136 | 0.49272551
# MobileNetV2                    | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.96185072 | 0.54298610 | 0.58708415
# MobileNetV2                    | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.93742354 | 0.35633839 | 0.45994832
# MobileNetV2                    | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.93469278 | 0.43674022 | 0.47594502
# MobileNetV2                    | per patch       | SVG                                      | (0, 2, 2)  | 0.90221605 | 0.38400524 | 0.42918455
# MobileNetV2                    | per patch       | BalancedDistributionSVG/500/31/0.30      | (0, 2, 2)  | 0.91041182 | 0.40539132 | 0.43555556
# MobileNetV2                    | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.93122717 | 0.32955848 | 0.44528875
# MobileNetV2                    | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.94303879 | 0.45519025 | 0.50651769
# MobileNetV2                    | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.94638523 | 0.38833300 | 0.50279330
# MobileNetV2                    | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.95666529 | 0.47952145 | 0.55000000
# MobileNetV2                    | per patch       | SVG                                      | (1, 0, 0)  | 0.88140820 | 0.26767551 | 0.33914729
# MobileNetV2                    | per patch       | BalancedDistributionSVG/500/31/0.30      | (1, 0, 0)  | 0.89365205 | 0.28622986 | 0.35577889
# MobileNetV2                    | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.94215531 | 0.36184092 | 0.48343778
# MobileNetV2                    | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.95675983 | 0.51195043 | 0.56492891
# MobileNetV2                    | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.92968169 | 0.32078896 | 0.44484959
# MobileNetV2                    | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.94415150 | 0.40271800 | 0.49473684
# MobileNetV2                    | per patch       | SVG                                      | (2, 0, 0)  | 0.86064133 | 0.23245738 | 0.31674208
# MobileNetV2                    | per patch       | BalancedDistributionSVG/500/31/0.30      | (2, 0, 0)  | 0.87535243 | 0.24917433 | 0.32911392
# MobileNetV2                    | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.92401878 | 0.29691278 | 0.42433697
# MobileNetV2                    | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.94820493 | 0.44130460 | 0.51179673
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.80021055 | 0.69929936 | 0.73913043
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.75190081 | 0.66316950 | 0.69662921
# MobileNetV2                    | per frame (max) | SVG                                      | None       | 0.75178383 | 0.74221794 | 0.70370370
# MobileNetV2                    | per frame (max) | BalancedDistributionSVG/500/31/0.30      | None       | 0.75903614 | 0.75404352 | 0.70319635
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.78266464 | 0.66326950 | 0.73913043
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.79412797 | 0.72417538 | 0.73958333
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.79225640 | 0.71194588 | 0.72527473
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.78710960 | 0.72117288 | 0.75392670
# MobileNetV2                    | per frame (max) | SVG                                      | (1, 1, 1)  | 0.72651772 | 0.70899129 | 0.69005848
# MobileNetV2                    | per frame (max) | BalancedDistributionSVG/500/31/0.30      | (1, 1, 1)  | 0.73365306 | 0.72653792 | 0.69411765
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.76780910 | 0.67355178 | 0.72222222
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.82816704 | 0.74075454 | 0.76842105
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.76348111 | 0.68188985 | 0.71140940
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.77506141 | 0.74154116 | 0.74528302
# MobileNetV2                    | per frame (max) | SVG                                      | (2, 2, 2)  | 0.71797871 | 0.70671436 | 0.66122449
# MobileNetV2                    | per frame (max) | BalancedDistributionSVG/500/31/0.30      | (2, 2, 2)  | 0.72803837 | 0.71978588 | 0.66310160
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.75154989 | 0.67267726 | 0.70588235
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.82126565 | 0.76034593 | 0.76699029
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.80629313 | 0.72863126 | 0.74213836
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.78816236 | 0.70370284 | 0.73563218
# MobileNetV2                    | per frame (max) | SVG                                      | (0, 1, 1)  | 0.71365072 | 0.70842206 | 0.65158371
# MobileNetV2                    | per frame (max) | BalancedDistributionSVG/500/31/0.30      | (0, 1, 1)  | 0.72335946 | 0.71893759 | 0.66285714
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.78523804 | 0.68714544 | 0.73033708
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.82325418 | 0.76431999 | 0.76666667
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.80699497 | 0.74211638 | 0.73939394
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.79576559 | 0.75634849 | 0.75280899
# MobileNetV2                    | per frame (max) | SVG                                      | (0, 2, 2)  | 0.72476313 | 0.70538272 | 0.65263158
# MobileNetV2                    | per frame (max) | BalancedDistributionSVG/500/31/0.30      | (0, 2, 2)  | 0.72932507 | 0.71983548 | 0.66265060
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.79050181 | 0.71087449 | 0.73684211
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.82290326 | 0.78393438 | 0.78857143
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.77716692 | 0.67884589 | 0.73033708
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.76441689 | 0.66287415 | 0.72164948
# MobileNetV2                    | per frame (max) | SVG                                      | (1, 0, 0)  | 0.76523570 | 0.75281952 | 0.70192308
# MobileNetV2                    | per frame (max) | BalancedDistributionSVG/500/31/0.30      | (1, 0, 0)  | 0.77108434 | 0.76237608 | 0.69523810
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.74710492 | 0.63345496 | 0.70329670
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.80605919 | 0.69738127 | 0.76555024
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.76207744 | 0.66059668 | 0.70652174
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.74406363 | 0.67238918 | 0.73429952
# MobileNetV2                    | per frame (max) | SVG                                      | (2, 0, 0)  | 0.75622880 | 0.72926852 | 0.70689655
# MobileNetV2                    | per frame (max) | BalancedDistributionSVG/500/31/0.30      | (2, 0, 0)  | 0.75798339 | 0.73604564 | 0.69369369
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.72733653 | 0.61195629 | 0.68449198
# MobileNetV2                    | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.79412797 | 0.67159150 | 0.77319588
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.83600421 | 0.76632728 | 0.77528090
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.80067844 | 0.76889196 | 0.75449102
# MobileNetV2                    | per frame (sum) | SVG                                      | None       | 0.77552930 | 0.75279450 | 0.72448980
# MobileNetV2                    | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | None       | 0.77845362 | 0.76151276 | 0.73298429
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.81916014 | 0.73764578 | 0.77380952
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.82571061 | 0.79511355 | 0.77380952
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.82231840 | 0.74126372 | 0.75862069
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.80559130 | 0.77322937 | 0.74146341
# MobileNetV2                    | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.77704995 | 0.75798494 | 0.71657754
# MobileNetV2                    | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | (1, 1, 1)  | 0.78488712 | 0.76902622 | 0.71957672
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.80757983 | 0.72448100 | 0.76571429
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.83038952 | 0.78958312 | 0.75897436
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.79833899 | 0.71085666 | 0.71657754
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.78827933 | 0.76387628 | 0.72727273
# MobileNetV2                    | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.76160954 | 0.73952644 | 0.69318182
# MobileNetV2                    | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | (2, 2, 2)  | 0.77213709 | 0.75051737 | 0.71111111
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.78102702 | 0.69198629 | 0.70646766
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.81249269 | 0.76978433 | 0.75376884
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.83600421 | 0.76632728 | 0.77528090
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.80067844 | 0.76889196 | 0.75449102
# MobileNetV2                    | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.77552930 | 0.75279450 | 0.72448980
# MobileNetV2                    | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | (0, 1, 1)  | 0.77845362 | 0.76151276 | 0.73298429
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.81916014 | 0.73764578 | 0.77380952
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.82571061 | 0.79511355 | 0.77380952
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.83600421 | 0.76632728 | 0.77528090
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.80067844 | 0.76889196 | 0.75449102
# MobileNetV2                    | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.77552930 | 0.75279450 | 0.72448980
# MobileNetV2                    | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | (0, 2, 2)  | 0.77845362 | 0.76151276 | 0.73298429
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.81916014 | 0.73764578 | 0.77380952
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.82571061 | 0.79511355 | 0.77380952
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.82231840 | 0.74126372 | 0.75862069
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.80559130 | 0.77322937 | 0.74146341
# MobileNetV2                    | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.77704995 | 0.75798494 | 0.71657754
# MobileNetV2                    | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | (1, 0, 0)  | 0.78488712 | 0.76902622 | 0.71957672
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.80757983 | 0.72448100 | 0.76571429
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.83038952 | 0.78958312 | 0.75897436
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.79833899 | 0.71085666 | 0.71657754
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.78827933 | 0.76387628 | 0.72727273
# MobileNetV2                    | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.76160954 | 0.73952644 | 0.69318182
# MobileNetV2                    | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | (2, 0, 0)  | 0.77213709 | 0.75051737 | 0.71111111
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.78102702 | 0.69198629 | 0.70646766
# MobileNetV2                    | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.81249269 | 0.76978433 | 0.75376884
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/0.20                      | None       | 0.86876161 | 0.27802917 | 0.32395833
# MobileNetV2_Block3             | per patch       | BalancedDistributionSVG/500/4/0.30       | None       | 0.83967667 | 0.15250200 | 0.27434093
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.87055945 | 0.25627088 | 0.33029501
# MobileNetV2_Block3             | per patch       | SVG                                      | None       | 0.86419689 | 0.15969785 | 0.27933575
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/0.50                      | None       | 0.87025065 | 0.27049410 | 0.32078375
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.86899279 | 0.28621716 | 0.33385605
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.89172553 | 0.26084462 | 0.31811038
# MobileNetV2_Block3             | per patch       | BalancedDistributionSVG/500/4/0.30       | (1, 1, 1)  | 0.86536739 | 0.14492267 | 0.29040224
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.89461405 | 0.23919467 | 0.32905924
# MobileNetV2_Block3             | per patch       | SVG                                      | (1, 1, 1)  | 0.86368051 | 0.14519818 | 0.28827199
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.89148511 | 0.25168268 | 0.31045337
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.89507661 | 0.27542403 | 0.33615955
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.87944450 | 0.22380497 | 0.27917510
# MobileNetV2_Block3             | per patch       | BalancedDistributionSVG/500/4/0.30       | (2, 2, 2)  | 0.84708233 | 0.13231407 | 0.25266543
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.88245078 | 0.20893471 | 0.28999502
# MobileNetV2_Block3             | per patch       | SVG                                      | (2, 2, 2)  | 0.84689970 | 0.13314599 | 0.25336972
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.87827131 | 0.21233890 | 0.27625815
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.88547989 | 0.23834260 | 0.30220292
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.89994420 | 0.35251994 | 0.36586158
# MobileNetV2_Block3             | per patch       | BalancedDistributionSVG/500/4/0.30       | (0, 1, 1)  | 0.87569670 | 0.15800407 | 0.29763117
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.90185348 | 0.31393132 | 0.37542768
# MobileNetV2_Block3             | per patch       | SVG                                      | (0, 1, 1)  | 0.87707302 | 0.16179832 | 0.29906822
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.90081402 | 0.34266231 | 0.35971223
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.90118507 | 0.36469164 | 0.38211821
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.91419361 | 0.38179553 | 0.38512922
# MobileNetV2_Block3             | per patch       | BalancedDistributionSVG/500/4/0.30       | (0, 2, 2)  | 0.87269479 | 0.15144309 | 0.29790158
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.91611201 | 0.33744957 | 0.39815553
# MobileNetV2_Block3             | per patch       | SVG                                      | (0, 2, 2)  | 0.87600438 | 0.15636532 | 0.30141250
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.91417090 | 0.36765731 | 0.37652308
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.91660718 | 0.40479677 | 0.40715382
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.87132308 | 0.22653854 | 0.29617820
# MobileNetV2_Block3             | per patch       | BalancedDistributionSVG/500/4/0.30       | (1, 0, 0)  | 0.85632267 | 0.14430279 | 0.27747475
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.87451141 | 0.21068833 | 0.30682415
# MobileNetV2_Block3             | per patch       | SVG                                      | (1, 0, 0)  | 0.85813809 | 0.14310156 | 0.27592231
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.87197558 | 0.21977889 | 0.29199540
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.87367336 | 0.23427157 | 0.30861931
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.84786290 | 0.18170447 | 0.24648331
# MobileNetV2_Block3             | per patch       | BalancedDistributionSVG/500/4/0.30       | (2, 0, 0)  | 0.84013806 | 0.13267769 | 0.25595683
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.85223211 | 0.17335197 | 0.25715659
# MobileNetV2_Block3             | per patch       | SVG                                      | (2, 0, 0)  | 0.83841666 | 0.13029876 | 0.24663037
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.84857545 | 0.17715609 | 0.24690645
# MobileNetV2_Block3             | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.85192435 | 0.18615112 | 0.25968074
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.61387297 | 0.59748370 | 0.61855670
# MobileNetV2_Block3             | per frame (max) | BalancedDistributionSVG/500/4/0.30       | None       | 0.49526260 | 0.43256825 | 0.61886792
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.59621008 | 0.51963063 | 0.62406015
# MobileNetV2_Block3             | per frame (max) | SVG                                      | None       | 0.50111124 | 0.43954534 | 0.62121212
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.62545327 | 0.59894232 | 0.63316583
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.58264124 | 0.53762686 | 0.62548263
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.64311615 | 0.59534647 | 0.65587045
# MobileNetV2_Block3             | per frame (max) | BalancedDistributionSVG/500/4/0.30       | (1, 1, 1)  | 0.53550123 | 0.44275413 | 0.66396761
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.64218037 | 0.54203081 | 0.66666667
# MobileNetV2_Block3             | per frame (max) | SVG                                      | (1, 1, 1)  | 0.55480173 | 0.46481548 | 0.65546218
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.63621476 | 0.58668498 | 0.65040650
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.64229734 | 0.59469390 | 0.66400000
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.60276056 | 0.54940205 | 0.66942149
# MobileNetV2_Block3             | per frame (max) | BalancedDistributionSVG/500/4/0.30       | (2, 2, 2)  | 0.56673295 | 0.46343116 | 0.64822134
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.61457480 | 0.51020719 | 0.67811159
# MobileNetV2_Block3             | per frame (max) | SVG                                      | (2, 2, 2)  | 0.57457013 | 0.47595527 | 0.64601770
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.59036145 | 0.52999359 | 0.66129032
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.62767575 | 0.56641992 | 0.67234043
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.72523102 | 0.68315000 | 0.68571429
# MobileNetV2_Block3             | per frame (max) | BalancedDistributionSVG/500/4/0.30       | (0, 1, 1)  | 0.48648965 | 0.42688730 | 0.62406015
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.70733419 | 0.60293332 | 0.68398268
# MobileNetV2_Block3             | per frame (max) | SVG                                      | (0, 1, 1)  | 0.51479705 | 0.44207525 | 0.62406015
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.71458650 | 0.66208367 | 0.67857143
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.71248099 | 0.65474099 | 0.69306931
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.67446485 | 0.65285514 | 0.66063348
# MobileNetV2_Block3             | per frame (max) | BalancedDistributionSVG/500/4/0.30       | (0, 2, 2)  | 0.50660896 | 0.42997276 | 0.62878788
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.66522400 | 0.59488747 | 0.66079295
# MobileNetV2_Block3             | per frame (max) | SVG                                      | (0, 2, 2)  | 0.53292783 | 0.44782599 | 0.62878788
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.65188911 | 0.62949332 | 0.65306122
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.69107498 | 0.66712551 | 0.66956522
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.63340742 | 0.61835272 | 0.66666667
# MobileNetV2_Block3             | per frame (max) | BalancedDistributionSVG/500/4/0.30       | (1, 0, 0)  | 0.57293251 | 0.50493638 | 0.63117871
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.60404726 | 0.52260338 | 0.66666667
# MobileNetV2_Block3             | per frame (max) | SVG                                      | (1, 0, 0)  | 0.58427886 | 0.51394940 | 0.63829787
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.64358404 | 0.61962515 | 0.65863454
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.61469178 | 0.58001733 | 0.66945607
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.59527430 | 0.58015797 | 0.66396761
# MobileNetV2_Block3             | per frame (max) | BalancedDistributionSVG/500/4/0.30       | (2, 0, 0)  | 0.60977892 | 0.51257146 | 0.65354331
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.59258393 | 0.50240473 | 0.66666667
# MobileNetV2_Block3             | per frame (max) | SVG                                      | (2, 0, 0)  | 0.62486841 | 0.52646963 | 0.65612648
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.61200140 | 0.58490250 | 0.65863454
# MobileNetV2_Block3             | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.60779038 | 0.60682045 | 0.67489712
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.64568955 | 0.62193916 | 0.62831858
# MobileNetV2_Block3             | per frame (sum) | BalancedDistributionSVG/500/4/0.30       | None       | 0.69750848 | 0.57451655 | 0.70408163
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.66007720 | 0.60555593 | 0.63849765
# MobileNetV2_Block3             | per frame (sum) | SVG                                      | None       | 0.69271260 | 0.59907355 | 0.69950739
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.65107030 | 0.62348396 | 0.62727273
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.67516669 | 0.66608884 | 0.63636364
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.63527898 | 0.61722057 | 0.62222222
# MobileNetV2_Block3             | per frame (sum) | BalancedDistributionSVG/500/4/0.30       | (1, 1, 1)  | 0.68779974 | 0.56995920 | 0.70243902
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.65013452 | 0.61637213 | 0.62564103
# MobileNetV2_Block3             | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.67832495 | 0.59414387 | 0.68246445
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.63972394 | 0.61704853 | 0.61946903
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.66229968 | 0.65338601 | 0.62727273
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.61621242 | 0.60762056 | 0.61940299
# MobileNetV2_Block3             | per frame (sum) | BalancedDistributionSVG/500/4/0.30       | (2, 2, 2)  | 0.66124693 | 0.56684688 | 0.68493151
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.63223769 | 0.61816652 | 0.62222222
# MobileNetV2_Block3             | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.64545561 | 0.58604707 | 0.66960352
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.61796701 | 0.60587064 | 0.61940299
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.64522166 | 0.63837026 | 0.62162162
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.64568955 | 0.62193916 | 0.62831858
# MobileNetV2_Block3             | per frame (sum) | BalancedDistributionSVG/500/4/0.30       | (0, 1, 1)  | 0.69750848 | 0.57451655 | 0.70408163
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.66007720 | 0.60555593 | 0.63849765
# MobileNetV2_Block3             | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.69271260 | 0.59907355 | 0.69950739
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.65107030 | 0.62348396 | 0.62727273
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.67516669 | 0.66608884 | 0.63636364
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.64568955 | 0.62193916 | 0.62831858
# MobileNetV2_Block3             | per frame (sum) | BalancedDistributionSVG/500/4/0.30       | (0, 2, 2)  | 0.69750848 | 0.57451655 | 0.70408163
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.66007720 | 0.60555593 | 0.63849765
# MobileNetV2_Block3             | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.69271260 | 0.59907355 | 0.69950739
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.65107030 | 0.62348396 | 0.62727273
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.67516669 | 0.66608884 | 0.63636364
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.63527898 | 0.61722057 | 0.62222222
# MobileNetV2_Block3             | per frame (sum) | BalancedDistributionSVG/500/4/0.30       | (1, 0, 0)  | 0.68779974 | 0.56995920 | 0.70243902
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.65013452 | 0.61637213 | 0.62564103
# MobileNetV2_Block3             | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.67832495 | 0.59414387 | 0.68246445
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.63972394 | 0.61704853 | 0.61946903
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.66229968 | 0.65338601 | 0.62727273
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.61621242 | 0.60762056 | 0.61940299
# MobileNetV2_Block3             | per frame (sum) | BalancedDistributionSVG/500/4/0.30       | (2, 0, 0)  | 0.66124693 | 0.56684688 | 0.68493151
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.63223769 | 0.61816652 | 0.62222222
# MobileNetV2_Block3             | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.64545561 | 0.58604707 | 0.66960352
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.61796701 | 0.60587064 | 0.61940299
# MobileNetV2_Block3             | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.64522166 | 0.63837026 | 0.62162162
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/0.20                      | None       | 0.94402756 | 0.53215702 | 0.54907539
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.93872302 | 0.41208857 | 0.49534104
# ResNet50V2_LargeImage          | per patch       | SVG                                      | None       | 0.91194767 | 0.41723184 | 0.43493922
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/0.50                      | None       | 0.94333369 | 0.52482127 | 0.55020693
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.93736717 | 0.41576551 | 0.49205118
# ResNet50V2_LargeImage          | per patch       | BalancedDistributionSVG/500/35/0.30      | None       | 0.90647167 | 0.44273869 | 0.45386158
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.96563489 | 0.56770221 | 0.56131687
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.94139222 | 0.36345845 | 0.46681835
# ResNet50V2_LargeImage          | per patch       | SVG                                      | (1, 1, 1)  | 0.94029787 | 0.51623237 | 0.49368110
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.96698570 | 0.56915098 | 0.57314715
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.94370139 | 0.37737551 | 0.47041086
# ResNet50V2_LargeImage          | per patch       | BalancedDistributionSVG/500/35/0.30      | (1, 1, 1)  | 0.94701843 | 0.57228234 | 0.54276663
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.95071482 | 0.47760575 | 0.48789572
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.91130841 | 0.26442063 | 0.38746719
# ResNet50V2_LargeImage          | per patch       | SVG                                      | (2, 2, 2)  | 0.92870121 | 0.45936338 | 0.45156209
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.95328643 | 0.47194044 | 0.49732977
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.91638664 | 0.27582816 | 0.39462810
# ResNet50V2_LargeImage          | per patch       | BalancedDistributionSVG/500/35/0.30      | (2, 2, 2)  | 0.93817429 | 0.51278936 | 0.49653808
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.96594246 | 0.58711438 | 0.58243650
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.95252769 | 0.42523897 | 0.50952821
# ResNet50V2_LargeImage          | per patch       | SVG                                      | (0, 1, 1)  | 0.94783217 | 0.54906543 | 0.52981969
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.96684810 | 0.59550488 | 0.59257740
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.95335158 | 0.43600113 | 0.50933284
# ResNet50V2_LargeImage          | per patch       | BalancedDistributionSVG/500/35/0.30      | (0, 1, 1)  | 0.95263323 | 0.59699446 | 0.56577737
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.96210747 | 0.54325945 | 0.54814085
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.93825664 | 0.35624396 | 0.47552910
# ResNet50V2_LargeImage          | per patch       | SVG                                      | (0, 2, 2)  | 0.95820822 | 0.60102752 | 0.56264237
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.96394449 | 0.55409079 | 0.55786096
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.94070720 | 0.36800657 | 0.47364888
# ResNet50V2_LargeImage          | per patch       | BalancedDistributionSVG/500/35/0.30      | (0, 2, 2)  | 0.96273657 | 0.64159347 | 0.58591275
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.94720862 | 0.50908224 | 0.52862470
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.93422552 | 0.34831534 | 0.44611092
# ResNet50V2_LargeImage          | per patch       | SVG                                      | (1, 0, 0)  | 0.91570702 | 0.41465760 | 0.42723118
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.94637276 | 0.50199455 | 0.52853758
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.93320708 | 0.35625485 | 0.44678407
# ResNet50V2_LargeImage          | per patch       | BalancedDistributionSVG/500/35/0.30      | (1, 0, 0)  | 0.91541862 | 0.45579298 | 0.45823274
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.92990730 | 0.43076014 | 0.45942117
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.91246284 | 0.27891650 | 0.37873754
# ResNet50V2_LargeImage          | per patch       | SVG                                      | (2, 0, 0)  | 0.89599667 | 0.35519054 | 0.36600135
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.92883689 | 0.42803775 | 0.45730129
# ResNet50V2_LargeImage          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.91307570 | 0.28951799 | 0.38756915
# ResNet50V2_LargeImage          | per patch       | BalancedDistributionSVG/500/35/0.30      | (2, 0, 0)  | 0.89761514 | 0.39741911 | 0.39603504
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.77342379 | 0.74056431 | 0.70129870
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.66066207 | 0.65002062 | 0.62882096
# ResNet50V2_LargeImage          | per frame (max) | SVG                                      | None       | 0.68440753 | 0.67954305 | 0.65573770
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.77213709 | 0.72875623 | 0.69856459
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.66276758 | 0.65626947 | 0.63793103
# ResNet50V2_LargeImage          | per frame (max) | BalancedDistributionSVG/500/35/0.30      | None       | 0.70019885 | 0.69053825 | 0.65573770
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.78547199 | 0.73239832 | 0.73404255
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.64346707 | 0.60366765 | 0.65020576
# ResNet50V2_LargeImage          | per frame (max) | SVG                                      | (1, 1, 1)  | 0.77845362 | 0.76899994 | 0.72081218
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.76710726 | 0.68806058 | 0.72538860
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.63902211 | 0.59545952 | 0.65811966
# ResNet50V2_LargeImage          | per frame (max) | BalancedDistributionSVG/500/35/0.30      | (1, 1, 1)  | 0.79635045 | 0.78819647 | 0.73626374
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.73950170 | 0.68588168 | 0.70103093
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.65107030 | 0.60578111 | 0.62711864
# ResNet50V2_LargeImage          | per frame (max) | SVG                                      | (2, 2, 2)  | 0.77073342 | 0.73765769 | 0.71111111
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.72850626 | 0.65308509 | 0.70142180
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.64592350 | 0.59787773 | 0.63333333
# ResNet50V2_LargeImage          | per frame (max) | BalancedDistributionSVG/500/35/0.30      | (2, 2, 2)  | 0.77880454 | 0.75072623 | 0.73267327
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.78441923 | 0.73207136 | 0.72300469
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.66803135 | 0.65287075 | 0.63768116
# ResNet50V2_LargeImage          | per frame (max) | SVG                                      | (0, 1, 1)  | 0.71996725 | 0.72031730 | 0.67873303
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.79073576 | 0.72233256 | 0.72037915
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.68627910 | 0.66179155 | 0.66094421
# ResNet50V2_LargeImage          | per frame (max) | BalancedDistributionSVG/500/35/0.30      | (0, 1, 1)  | 0.73622646 | 0.73762372 | 0.69306931
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.77447655 | 0.72958133 | 0.70930233
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.67844192 | 0.66443217 | 0.64186047
# ResNet50V2_LargeImage          | per frame (max) | SVG                                      | (0, 2, 2)  | 0.73271728 | 0.71605125 | 0.70642202
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.77880454 | 0.71847141 | 0.71502591
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.69739151 | 0.67090338 | 0.68316832
# ResNet50V2_LargeImage          | per frame (max) | BalancedDistributionSVG/500/35/0.30      | (0, 2, 2)  | 0.75084805 | 0.73639948 | 0.71153846
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.78921511 | 0.74662133 | 0.71111111
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.62989823 | 0.59779899 | 0.62595420
# ResNet50V2_LargeImage          | per frame (max) | SVG                                      | (1, 0, 0)  | 0.73646040 | 0.75617740 | 0.65573770
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.78172886 | 0.70687740 | 0.72081218
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.61913674 | 0.59097890 | 0.63374486
# ResNet50V2_LargeImage          | per frame (max) | BalancedDistributionSVG/500/35/0.30      | (1, 0, 0)  | 0.75540999 | 0.76718508 | 0.67052023
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.74710492 | 0.71065255 | 0.70813397
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.64159551 | 0.59857869 | 0.64285714
# ResNet50V2_LargeImage          | per frame (max) | SVG                                      | (2, 0, 0)  | 0.78044216 | 0.78354161 | 0.70157068
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.74921043 | 0.66787390 | 0.71957672
# ResNet50V2_LargeImage          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.62486841 | 0.59110410 | 0.63114754
# ResNet50V2_LargeImage          | per frame (max) | BalancedDistributionSVG/500/35/0.30      | (2, 0, 0)  | 0.78921511 | 0.78777842 | 0.70967742
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.77704995 | 0.76251714 | 0.71657754
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.72172184 | 0.70850600 | 0.67647059
# ResNet50V2_LargeImage          | per frame (sum) | SVG                                      | None       | 0.78488712 | 0.78770995 | 0.71186441
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.77950638 | 0.75266549 | 0.72826087
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.72757048 | 0.71822849 | 0.68599034
# ResNet50V2_LargeImage          | per frame (sum) | BalancedDistributionSVG/500/35/0.30      | None       | 0.79213943 | 0.79639247 | 0.72432432
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.76757515 | 0.75940989 | 0.69047619
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.68393964 | 0.64089050 | 0.64356436
# ResNet50V2_LargeImage          | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.77903848 | 0.78764230 | 0.70238095
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.74932741 | 0.72932141 | 0.69230769
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.69212773 | 0.65004572 | 0.65094340
# ResNet50V2_LargeImage          | per frame (sum) | BalancedDistributionSVG/500/35/0.30      | (1, 1, 1)  | 0.78746052 | 0.79665975 | 0.71065990
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.74453152 | 0.73877781 | 0.66666667
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.67879284 | 0.62818940 | 0.64406780
# ResNet50V2_LargeImage          | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.75482513 | 0.76845533 | 0.69230769
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.72874020 | 0.70228068 | 0.67346939
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.68522634 | 0.62822797 | 0.65094340
# ResNet50V2_LargeImage          | per frame (sum) | BalancedDistributionSVG/500/35/0.30      | (2, 2, 2)  | 0.76114165 | 0.76654487 | 0.68817204
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.77704995 | 0.76251714 | 0.71657754
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.72172184 | 0.70850600 | 0.67647059
# ResNet50V2_LargeImage          | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.78488712 | 0.78770995 | 0.71186441
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.77950638 | 0.75266549 | 0.72826087
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.72757048 | 0.71822849 | 0.68599034
# ResNet50V2_LargeImage          | per frame (sum) | BalancedDistributionSVG/500/35/0.30      | (0, 1, 1)  | 0.79213943 | 0.79639247 | 0.72432432
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.77704995 | 0.76251714 | 0.71657754
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.72172184 | 0.70850600 | 0.67647059
# ResNet50V2_LargeImage          | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.78488712 | 0.78770995 | 0.71186441
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.77950638 | 0.75266549 | 0.72826087
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.72757048 | 0.71822849 | 0.68599034
# ResNet50V2_LargeImage          | per frame (sum) | BalancedDistributionSVG/500/35/0.30      | (0, 2, 2)  | 0.79213943 | 0.79639247 | 0.72432432
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.76757515 | 0.75940989 | 0.69047619
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.68393964 | 0.64089050 | 0.64356436
# ResNet50V2_LargeImage          | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.77903848 | 0.78764230 | 0.70238095
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.74932741 | 0.72932141 | 0.69230769
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.69212773 | 0.65004572 | 0.65094340
# ResNet50V2_LargeImage          | per frame (sum) | BalancedDistributionSVG/500/35/0.30      | (1, 0, 0)  | 0.78746052 | 0.79665975 | 0.71065990
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.74453152 | 0.73877781 | 0.66666667
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.67879284 | 0.62818940 | 0.64406780
# ResNet50V2_LargeImage          | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.75482513 | 0.76845533 | 0.69230769
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.72874020 | 0.70228068 | 0.67346939
# ResNet50V2_LargeImage          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.68522634 | 0.62822797 | 0.65094340
# ResNet50V2_LargeImage          | per frame (sum) | BalancedDistributionSVG/500/35/0.30      | (2, 0, 0)  | 0.76114165 | 0.76654487 | 0.68817204
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/0.20                      | None       | 0.87926417 | 0.21794777 | 0.29977247
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.87488801 | 0.23462947 | 0.31412429
# ResNet50V2_Stack4_LargeImage   | per patch       | SVG                                      | None       | 0.87886836 | 0.19226414 | 0.29033165
# ResNet50V2_Stack4_LargeImage   | per patch       | BalancedDistributionSVG/500/31/0.30      | None       | 0.85861392 | 0.19437533 | 0.30118494
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/0.50                      | None       | 0.87993499 | 0.21728137 | 0.30010834
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.87315033 | 0.25071029 | 0.32138979
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.89768636 | 0.23491834 | 0.31510832
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.90738084 | 0.26294281 | 0.34945802
# ResNet50V2_Stack4_LargeImage   | per patch       | SVG                                      | (1, 1, 1)  | 0.88302015 | 0.16924161 | 0.29783434
# ResNet50V2_Stack4_LargeImage   | per patch       | BalancedDistributionSVG/500/31/0.30      | (1, 1, 1)  | 0.89735442 | 0.19556005 | 0.33622184
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.89760223 | 0.23010105 | 0.31634717
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.90347763 | 0.30603738 | 0.35728098
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.88241511 | 0.18420264 | 0.29753915
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.90301377 | 0.23403837 | 0.33979592
# ResNet50V2_Stack4_LargeImage   | per patch       | SVG                                      | (2, 2, 2)  | 0.86117776 | 0.15176300 | 0.27092239
# ResNet50V2_Stack4_LargeImage   | per patch       | BalancedDistributionSVG/500/31/0.30      | (2, 2, 2)  | 0.87528529 | 0.17105062 | 0.30816747
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.88186994 | 0.18084834 | 0.29653834
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.90189217 | 0.27671145 | 0.35705734
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.91356402 | 0.28638166 | 0.35411622
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.91623324 | 0.32598036 | 0.39739277
# ResNet50V2_Stack4_LargeImage   | per patch       | SVG                                      | (0, 1, 1)  | 0.89781838 | 0.18885280 | 0.33044567
# ResNet50V2_Stack4_LargeImage   | per patch       | BalancedDistributionSVG/500/31/0.30      | (0, 1, 1)  | 0.90534451 | 0.21278055 | 0.34780420
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.91354521 | 0.27912996 | 0.35389657
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.91430735 | 0.38155311 | 0.40786701
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.91917522 | 0.27682149 | 0.36378958
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.93105811 | 0.33556751 | 0.41702267
# ResNet50V2_Stack4_LargeImage   | per patch       | SVG                                      | (0, 2, 2)  | 0.89476116 | 0.17859645 | 0.33432408
# ResNet50V2_Stack4_LargeImage   | per patch       | BalancedDistributionSVG/500/31/0.30      | (0, 2, 2)  | 0.90649553 | 0.20243659 | 0.35090771
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.91856487 | 0.26838981 | 0.36330762
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.93201170 | 0.40961153 | 0.42968601
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.87563665 | 0.19117804 | 0.27789821
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.87717350 | 0.20993013 | 0.30278884
# ResNet50V2_Stack4_LargeImage   | per patch       | SVG                                      | (1, 0, 0)  | 0.87390281 | 0.17258663 | 0.27848357
# ResNet50V2_Stack4_LargeImage   | per patch       | BalancedDistributionSVG/500/31/0.30      | (1, 0, 0)  | 0.87479623 | 0.19056826 | 0.31141406
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.87623991 | 0.19073537 | 0.27855270
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.87330737 | 0.22432928 | 0.30682353
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.85311610 | 0.16247025 | 0.24881240
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.85818997 | 0.17563956 | 0.26108461
# ResNet50V2_Stack4_LargeImage   | per patch       | SVG                                      | (2, 0, 0)  | 0.85346646 | 0.15262439 | 0.25413700
# ResNet50V2_Stack4_LargeImage   | per patch       | BalancedDistributionSVG/500/31/0.30      | (2, 0, 0)  | 0.86099657 | 0.17304507 | 0.28990626
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.85391250 | 0.16236721 | 0.24891113
# ResNet50V2_Stack4_LargeImage   | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.85355143 | 0.18495951 | 0.26468481
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.55094163 | 0.48408062 | 0.62406015
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.52976956 | 0.45287450 | 0.63157895
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SVG                                      | None       | 0.53292783 | 0.44261344 | 0.63348416
# ResNet50V2_Stack4_LargeImage   | per frame (max) | BalancedDistributionSVG/500/31/0.30      | None       | 0.52719616 | 0.44066341 | 0.62790698
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.54895309 | 0.47887430 | 0.62641509
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.51105392 | 0.44654488 | 0.61940299
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.56497836 | 0.57483678 | 0.63374486
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.54076500 | 0.44550736 | 0.63900415
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SVG                                      | (1, 1, 1)  | 0.55679027 | 0.45051428 | 0.65612648
# ResNet50V2_Stack4_LargeImage   | per frame (max) | BalancedDistributionSVG/500/31/0.30      | (1, 1, 1)  | 0.52052872 | 0.43222522 | 0.64843750
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.56556322 | 0.57364561 | 0.63900415
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.55620540 | 0.49622050 | 0.62857143
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.70873786 | 0.66051065 | 0.67289720
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.63609779 | 0.50022432 | 0.68141593
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SVG                                      | (2, 2, 2)  | 0.65972628 | 0.55638897 | 0.68695652
# ResNet50V2_Stack4_LargeImage   | per frame (max) | BalancedDistributionSVG/500/31/0.30      | (2, 2, 2)  | 0.64627442 | 0.53888602 | 0.68965517
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.70604749 | 0.65741568 | 0.66382979
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.64510469 | 0.54270716 | 0.68493151
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.60732249 | 0.61227508 | 0.62641509
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.60989589 | 0.51635894 | 0.64957265
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SVG                                      | (0, 1, 1)  | 0.54462510 | 0.45913844 | 0.63967611
# ResNet50V2_Stack4_LargeImage   | per frame (max) | BalancedDistributionSVG/500/31/0.30      | (0, 1, 1)  | 0.53468242 | 0.45346664 | 0.63529412
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.60369634 | 0.60692321 | 0.62406015
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.61398994 | 0.53989462 | 0.63492063
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.64639139 | 0.62856255 | 0.65060241
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.67879284 | 0.55853010 | 0.67980296
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SVG                                      | (0, 2, 2)  | 0.61937069 | 0.53693592 | 0.64516129
# ResNet50V2_Stack4_LargeImage   | per frame (max) | BalancedDistributionSVG/500/31/0.30      | (0, 2, 2)  | 0.60720552 | 0.53263758 | 0.64197531
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.64580653 | 0.62860479 | 0.64800000
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.68440753 | 0.59202509 | 0.67000000
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.53035443 | 0.44744565 | 0.63601533
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.48204468 | 0.40118241 | 0.63601533
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SVG                                      | (1, 0, 0)  | 0.52392093 | 0.44052050 | 0.63601533
# ResNet50V2_Stack4_LargeImage   | per frame (max) | BalancedDistributionSVG/500/31/0.30      | (1, 0, 0)  | 0.51280852 | 0.43085208 | 0.63601533
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.52883378 | 0.44404018 | 0.63320463
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.50532226 | 0.43772438 | 0.63117871
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.52848286 | 0.44214428 | 0.63117871
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.43303310 | 0.38231990 | 0.63358779
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SVG                                      | (2, 0, 0)  | 0.55012282 | 0.44830856 | 0.64092664
# ResNet50V2_Stack4_LargeImage   | per frame (max) | BalancedDistributionSVG/500/31/0.30      | (2, 0, 0)  | 0.53152415 | 0.43566793 | 0.64313725
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.52649433 | 0.43852857 | 0.63117871
# ResNet50V2_Stack4_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.47923734 | 0.40925221 | 0.64591440
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.70066674 | 0.65242398 | 0.68062827
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.70710025 | 0.65105890 | 0.67836257
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SVG                                      | None       | 0.76862791 | 0.72542733 | 0.74556213
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | None       | 0.80184817 | 0.77481334 | 0.75000000
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.70546263 | 0.65726492 | 0.68717949
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.71985027 | 0.68442538 | 0.68208092
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.68288689 | 0.64456882 | 0.66326531
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.69154287 | 0.65188324 | 0.67428571
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.75622880 | 0.71126077 | 0.74418605
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | (1, 1, 1)  | 0.79190549 | 0.77294283 | 0.74712644
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.68803369 | 0.65046049 | 0.66331658
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.69739151 | 0.67067013 | 0.66666667
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.64908176 | 0.62426533 | 0.63076923
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.67259329 | 0.63652009 | 0.65671642
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.73377003 | 0.69173159 | 0.72619048
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | (2, 2, 2)  | 0.77506141 | 0.74893502 | 0.73750000
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.65528132 | 0.62988417 | 0.63106796
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.67259329 | 0.64402907 | 0.65365854
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.70066674 | 0.65242398 | 0.68062827
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.70710025 | 0.65105890 | 0.67836257
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.76862791 | 0.72542733 | 0.74556213
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | (0, 1, 1)  | 0.80184817 | 0.77481334 | 0.75000000
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.70546263 | 0.65726492 | 0.68717949
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.71985027 | 0.68442538 | 0.68208092
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.70066674 | 0.65242398 | 0.68062827
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.70710025 | 0.65105890 | 0.67836257
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.76862791 | 0.72542733 | 0.74556213
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | (0, 2, 2)  | 0.80184817 | 0.77481334 | 0.75000000
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.70546263 | 0.65726492 | 0.68717949
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.71985027 | 0.68442538 | 0.68208092
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.68288689 | 0.64456882 | 0.66326531
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.69154287 | 0.65188324 | 0.67428571
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.75622880 | 0.71126077 | 0.74418605
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | (1, 0, 0)  | 0.79190549 | 0.77294283 | 0.74712644
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.68803369 | 0.65046049 | 0.66331658
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.69739151 | 0.67067013 | 0.66666667
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.64908176 | 0.62426533 | 0.63076923
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.67259329 | 0.63652009 | 0.65671642
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.73377003 | 0.69173159 | 0.72619048
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | (2, 0, 0)  | 0.77506141 | 0.74893502 | 0.73750000
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.65528132 | 0.62988417 | 0.63106796
# ResNet50V2_Stack4_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.67259329 | 0.64402907 | 0.65365854
# C3D                            | per patch       | SpatialBin/SVG/0.20                      | None       | 0.91702998 | 0.34623796 | 0.45956765
# C3D                            | per patch       | BalancedDistributionSVG/500/27/0.30      | None       | 0.88258465 | 0.38339277 | 0.44964395
# C3D                            | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.91987711 | 0.36883919 | 0.44366812
# C3D                            | per patch       | SVG                                      | None       | 0.89036480 | 0.36650456 | 0.44166667
# C3D                            | per patch       | SpatialBin/SVG/0.50                      | None       | 0.91516930 | 0.34809024 | 0.45232467
# C3D                            | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.90961672 | 0.35203015 | 0.43044190
# C3D                            | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.92399200 | 0.31739923 | 0.42722372
# C3D                            | per patch       | BalancedDistributionSVG/500/27/0.30      | (1, 1, 1)  | 0.91247951 | 0.46587904 | 0.46861925
# C3D                            | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.92721953 | 0.32074507 | 0.45089903
# C3D                            | per patch       | SVG                                      | (1, 1, 1)  | 0.91387529 | 0.41555718 | 0.45617342
# C3D                            | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.92233538 | 0.32272857 | 0.41631505
# C3D                            | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.92555408 | 0.31912320 | 0.43362241
# C3D                            | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.88695382 | 0.23392197 | 0.32781457
# C3D                            | per patch       | BalancedDistributionSVG/500/27/0.30      | (2, 2, 2)  | 0.90983900 | 0.41667957 | 0.45924453
# C3D                            | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.87226040 | 0.21709475 | 0.32120109
# C3D                            | per patch       | SVG                                      | (2, 2, 2)  | 0.90199378 | 0.37408637 | 0.41067098
# C3D                            | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.88874675 | 0.23998829 | 0.33774834
# C3D                            | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.87740726 | 0.22501950 | 0.32255125
# C3D                            | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.91212066 | 0.32877376 | 0.43116490
# C3D                            | per patch       | BalancedDistributionSVG/500/27/0.30      | (0, 1, 1)  | 0.90122438 | 0.44075234 | 0.47609148
# C3D                            | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.92086971 | 0.34069696 | 0.46840149
# C3D                            | per patch       | SVG                                      | (0, 1, 1)  | 0.90287779 | 0.40989868 | 0.47227191
# C3D                            | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.90923538 | 0.32956083 | 0.41692308
# C3D                            | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.91808271 | 0.33600129 | 0.45109489
# C3D                            | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.88044867 | 0.26610025 | 0.35435057
# C3D                            | per patch       | BalancedDistributionSVG/500/27/0.30      | (0, 2, 2)  | 0.89984735 | 0.42518222 | 0.47311828
# C3D                            | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.88601839 | 0.25821803 | 0.38233515
# C3D                            | per patch       | SVG                                      | (0, 2, 2)  | 0.89720764 | 0.37375532 | 0.45106383
# C3D                            | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.87756821 | 0.26957252 | 0.35000000
# C3D                            | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.88615631 | 0.26309454 | 0.36687752
# C3D                            | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.92935203 | 0.33058963 | 0.45353675
# C3D                            | per patch       | BalancedDistributionSVG/500/27/0.30      | (1, 0, 0)  | 0.89674756 | 0.40726237 | 0.44215349
# C3D                            | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.93371450 | 0.34878413 | 0.46278556
# C3D                            | per patch       | SVG                                      | (1, 0, 0)  | 0.90247850 | 0.38211663 | 0.44727694
# C3D                            | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.92588963 | 0.33516441 | 0.45100865
# C3D                            | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.92652352 | 0.33599613 | 0.44142857
# C3D                            | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.92225103 | 0.29397777 | 0.41893831
# C3D                            | per patch       | BalancedDistributionSVG/500/27/0.30      | (2, 0, 0)  | 0.89360251 | 0.39586674 | 0.43076923
# C3D                            | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.92516014 | 0.29949719 | 0.42922976
# C3D                            | per patch       | SVG                                      | (2, 0, 0)  | 0.89601887 | 0.35109877 | 0.42437923
# C3D                            | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.91856287 | 0.30015276 | 0.42444594
# C3D                            | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.92006738 | 0.29245759 | 0.41374920
# C3D                            | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.82173354 | 0.79410877 | 0.76404494
# C3D                            | per frame (max) | BalancedDistributionSVG/500/27/0.30      | None       | 0.81799041 | 0.83075011 | 0.73619632
# C3D                            | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.84676570 | 0.82368422 | 0.79096045
# C3D                            | per frame (max) | SVG                                      | None       | 0.80231606 | 0.79688918 | 0.71823204
# C3D                            | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.80559130 | 0.78225101 | 0.75000000
# C3D                            | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.85694233 | 0.85894307 | 0.77528090
# C3D                            | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.83752486 | 0.78181809 | 0.76041667
# C3D                            | per frame (max) | BalancedDistributionSVG/500/27/0.30      | (1, 1, 1)  | 0.83998128 | 0.85971210 | 0.75675676
# C3D                            | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.84290560 | 0.80000504 | 0.81176471
# C3D                            | per frame (max) | SVG                                      | (1, 1, 1)  | 0.80804772 | 0.81251206 | 0.71276596
# C3D                            | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.83436659 | 0.78452461 | 0.76237624
# C3D                            | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.85869692 | 0.83464342 | 0.80681818
# C3D                            | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.82664639 | 0.75271458 | 0.78260870
# C3D                            | per frame (max) | BalancedDistributionSVG/500/27/0.30      | (2, 2, 2)  | 0.81927711 | 0.82135965 | 0.71748879
# C3D                            | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.82407299 | 0.75793666 | 0.79532164
# C3D                            | per frame (max) | SVG                                      | (2, 2, 2)  | 0.76125863 | 0.75072288 | 0.69724771
# C3D                            | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.82547666 | 0.75947262 | 0.78260870
# C3D                            | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.83541935 | 0.78082778 | 0.79558011
# C3D                            | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.82442391 | 0.80791574 | 0.78857143
# C3D                            | per frame (max) | BalancedDistributionSVG/500/27/0.30      | (0, 1, 1)  | 0.82430694 | 0.85012539 | 0.74157303
# C3D                            | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.85705931 | 0.84042143 | 0.81761006
# C3D                            | per frame (max) | SVG                                      | (0, 1, 1)  | 0.80956837 | 0.82335058 | 0.71910112
# C3D                            | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.81331150 | 0.79933907 | 0.77714286
# C3D                            | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.86220611 | 0.87413298 | 0.78787879
# C3D                            | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.83085741 | 0.81969825 | 0.77528090
# C3D                            | per frame (max) | BalancedDistributionSVG/500/27/0.30      | (0, 2, 2)  | 0.80968534 | 0.83542292 | 0.72483221
# C3D                            | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.85764417 | 0.84723046 | 0.81250000
# C3D                            | per frame (max) | SVG                                      | (0, 2, 2)  | 0.78044216 | 0.79984853 | 0.69230769
# C3D                            | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.81857527 | 0.81072993 | 0.76666667
# C3D                            | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.86431162 | 0.87698185 | 0.80000000
# C3D                            | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.82805006 | 0.77007903 | 0.74876847
# C3D                            | per frame (max) | BalancedDistributionSVG/500/27/0.30      | (1, 0, 0)  | 0.84770149 | 0.86719054 | 0.76646707
# C3D                            | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.82676336 | 0.77493464 | 0.77647059
# C3D                            | per frame (max) | SVG                                      | (1, 0, 0)  | 0.81143993 | 0.80113093 | 0.75739645
# C3D                            | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.82313721 | 0.77093742 | 0.74876847
# C3D                            | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.83670605 | 0.81657875 | 0.75392670
# C3D                            | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.79775412 | 0.73282317 | 0.71895425
# C3D                            | per frame (max) | BalancedDistributionSVG/500/27/0.30      | (2, 0, 0)  | 0.86875658 | 0.87578170 | 0.77647059
# C3D                            | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.80687800 | 0.74275131 | 0.77094972
# C3D                            | per frame (max) | SVG                                      | (2, 0, 0)  | 0.78746052 | 0.74869635 | 0.72316384
# C3D                            | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.79974266 | 0.74041074 | 0.73103448
# C3D                            | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.82126565 | 0.78070482 | 0.76086957
# C3D                            | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.84173588 | 0.81943492 | 0.77966102
# C3D                            | per frame (sum) | BalancedDistributionSVG/500/27/0.30      | None       | 0.79190549 | 0.80435788 | 0.70238095
# C3D                            | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.86887355 | 0.85383992 | 0.82424242
# C3D                            | per frame (sum) | SVG                                      | None       | 0.76617148 | 0.77599657 | 0.69841270
# C3D                            | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.83191016 | 0.81570048 | 0.77647059
# C3D                            | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.87799743 | 0.88063120 | 0.82716049
# C3D                            | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.84980699 | 0.79295806 | 0.79041916
# C3D                            | per frame (sum) | BalancedDistributionSVG/500/27/0.30      | (1, 1, 1)  | 0.79307521 | 0.79924801 | 0.69523810
# C3D                            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.86033454 | 0.81340844 | 0.79761905
# C3D                            | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.75973798 | 0.75914746 | 0.69523810
# C3D                            | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.84629781 | 0.79490066 | 0.79012346
# C3D                            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.87039420 | 0.83297408 | 0.81142857
# C3D                            | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.83939642 | 0.75870763 | 0.78313253
# C3D                            | per frame (sum) | BalancedDistributionSVG/500/27/0.30      | (2, 2, 2)  | 0.79295824 | 0.78368586 | 0.71568627
# C3D                            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.82547666 | 0.75608425 | 0.75935829
# C3D                            | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.75225173 | 0.72821845 | 0.69198312
# C3D                            | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.83740788 | 0.76428521 | 0.77456647
# C3D                            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.84360744 | 0.78643476 | 0.78453039
# C3D                            | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.84173588 | 0.81943492 | 0.77966102
# C3D                            | per frame (sum) | BalancedDistributionSVG/500/27/0.30      | (0, 1, 1)  | 0.79190549 | 0.80435788 | 0.70238095
# C3D                            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.86887355 | 0.85383992 | 0.82424242
# C3D                            | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.76617148 | 0.77599657 | 0.69841270
# C3D                            | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.83191016 | 0.81570048 | 0.77647059
# C3D                            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.87799743 | 0.88063120 | 0.82716049
# C3D                            | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.84173588 | 0.81943492 | 0.77966102
# C3D                            | per frame (sum) | BalancedDistributionSVG/500/27/0.30      | (0, 2, 2)  | 0.79190549 | 0.80435788 | 0.70238095
# C3D                            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.86887355 | 0.85383992 | 0.82424242
# C3D                            | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.76617148 | 0.77599657 | 0.69841270
# C3D                            | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.83191016 | 0.81570048 | 0.77647059
# C3D                            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.87799743 | 0.88063120 | 0.82716049
# C3D                            | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.84980699 | 0.79295806 | 0.79041916
# C3D                            | per frame (sum) | BalancedDistributionSVG/500/27/0.30      | (1, 0, 0)  | 0.79307521 | 0.79924801 | 0.69523810
# C3D                            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.86033454 | 0.81340844 | 0.79761905
# C3D                            | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.75973798 | 0.75914746 | 0.69523810
# C3D                            | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.84629781 | 0.79490066 | 0.79012346
# C3D                            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.87039420 | 0.83297408 | 0.81142857
# C3D                            | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.83939642 | 0.75870763 | 0.78313253
# C3D                            | per frame (sum) | BalancedDistributionSVG/500/27/0.30      | (2, 0, 0)  | 0.79295824 | 0.78368586 | 0.71568627
# C3D                            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.82547666 | 0.75608425 | 0.75935829
# C3D                            | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.75225173 | 0.72821845 | 0.69198312
# C3D                            | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.83740788 | 0.76428521 | 0.77456647
# C3D                            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.84360744 | 0.78643476 | 0.78453039
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/0.20                      | None       | 0.65285026 | 0.06281404 | 0.14532030
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.63856901 | 0.06050228 | 0.13665633
# EfficientNetB0_Block3          | per patch       | SVG                                      | None       | 0.60368467 | 0.05987792 | 0.13512366
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/0.50                      | None       | 0.65248909 | 0.06282798 | 0.14588788
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.63882126 | 0.06047499 | 0.13530814
# EfficientNetB0_Block3          | per patch       | BalancedDistributionSVG/500/3/0.30       | None       | 0.50304987 | 0.05252532 | 0.12091162
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.62783867 | 0.05772304 | 0.13292637
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.61850906 | 0.05741070 | 0.12638492
# EfficientNetB0_Block3          | per patch       | SVG                                      | (1, 1, 1)  | 0.62233975 | 0.05858826 | 0.13717374
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.62816657 | 0.05782088 | 0.13317571
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.61404064 | 0.05625444 | 0.12505562
# EfficientNetB0_Block3          | per patch       | BalancedDistributionSVG/500/3/0.30       | (1, 1, 1)  | 0.51688748 | 0.05051504 | 0.10933812
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.57677894 | 0.05210959 | 0.11532598
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.57231933 | 0.05289116 | 0.11236468
# EfficientNetB0_Block3          | per patch       | SVG                                      | (2, 2, 2)  | 0.56304215 | 0.05198844 | 0.11264742
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.57662550 | 0.05215508 | 0.11542654
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.56361159 | 0.05039027 | 0.11187883
# EfficientNetB0_Block3          | per patch       | BalancedDistributionSVG/500/3/0.30       | (2, 2, 2)  | 0.52324905 | 0.04918281 | 0.09864792
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.63151170 | 0.05819825 | 0.13680036
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.61946155 | 0.05648678 | 0.13046017
# EfficientNetB0_Block3          | per patch       | SVG                                      | (0, 1, 1)  | 0.61785009 | 0.05868964 | 0.13717058
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.63181869 | 0.05829980 | 0.13739918
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.61865085 | 0.05636246 | 0.12963372
# EfficientNetB0_Block3          | per patch       | BalancedDistributionSVG/500/3/0.30       | (0, 1, 1)  | 0.51097780 | 0.05058871 | 0.11238122
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.58195907 | 0.05251712 | 0.11791939
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.57507875 | 0.05139997 | 0.11514476
# EfficientNetB0_Block3          | per patch       | SVG                                      | (0, 2, 2)  | 0.56901395 | 0.05324475 | 0.11409213
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.58188010 | 0.05257810 | 0.11794689
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.57531067 | 0.05138389 | 0.11511551
# EfficientNetB0_Block3          | per patch       | BalancedDistributionSVG/500/3/0.30       | (0, 2, 2)  | 0.52190461 | 0.04960919 | 0.10052419
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.65828643 | 0.06290730 | 0.14287167
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.64362684 | 0.06156420 | 0.13346393
# EfficientNetB0_Block3          | per patch       | SVG                                      | (1, 0, 0)  | 0.63442247 | 0.06156083 | 0.14243331
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.65828923 | 0.06294514 | 0.14380334
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.64044737 | 0.06064927 | 0.13197431
# EfficientNetB0_Block3          | per patch       | BalancedDistributionSVG/500/3/0.30       | (1, 0, 0)  | 0.50708871 | 0.05151539 | 0.11636187
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.65205283 | 0.06198578 | 0.13919481
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.63411375 | 0.06087880 | 0.12953406
# EfficientNetB0_Block3          | per patch       | SVG                                      | (2, 0, 0)  | 0.63900253 | 0.06115090 | 0.14234048
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.65196762 | 0.06199932 | 0.13978423
# EfficientNetB0_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.62798167 | 0.05909365 | 0.12803226
# EfficientNetB0_Block3          | per patch       | BalancedDistributionSVG/500/3/0.30       | (2, 0, 0)  | 0.51132486 | 0.05101016 | 0.11468684
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.33103287 | 0.34123569 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.34822786 | 0.34856506 | 0.62172285
# EfficientNetB0_Block3          | per frame (max) | SVG                                      | None       | 0.46824190 | 0.49818037 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.33851913 | 0.35138908 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.40238624 | 0.38529470 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | BalancedDistributionSVG/500/3/0.30       | None       | 0.46859282 | 0.50005890 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.39408118 | 0.37860901 | 0.62406015
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.41232893 | 0.38016519 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SVG                                      | (1, 1, 1)  | 0.55047374 | 0.53679805 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.40028073 | 0.38835160 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.40846883 | 0.38442040 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | BalancedDistributionSVG/500/3/0.30       | (1, 1, 1)  | 0.54193473 | 0.53581169 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.43279916 | 0.38612352 | 0.63601533
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.44519827 | 0.39236407 | 0.63117871
# EfficientNetB0_Block3          | per frame (max) | SVG                                      | (2, 2, 2)  | 0.60767341 | 0.56754372 | 0.65843621
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.45443912 | 0.39739804 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.40109954 | 0.38219969 | 0.63529412
# EfficientNetB0_Block3          | per frame (max) | BalancedDistributionSVG/500/3/0.30       | (2, 2, 2)  | 0.56837057 | 0.55189257 | 0.63846154
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.36717745 | 0.36298996 | 0.62406015
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.36916598 | 0.35596985 | 0.62172285
# EfficientNetB0_Block3          | per frame (max) | SVG                                      | (0, 1, 1)  | 0.54392327 | 0.53015497 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.37922564 | 0.37721509 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.41338168 | 0.38534293 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | BalancedDistributionSVG/500/3/0.30       | (0, 1, 1)  | 0.53386361 | 0.52369573 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.38893438 | 0.36803334 | 0.63035019
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.38390455 | 0.35754676 | 0.61886792
# EfficientNetB0_Block3          | per frame (max) | SVG                                      | (0, 2, 2)  | 0.59644403 | 0.57619722 | 0.61776062
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.40262019 | 0.37519784 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.43806293 | 0.38926695 | 0.63934426
# EfficientNetB0_Block3          | per frame (max) | BalancedDistributionSVG/500/3/0.30       | (0, 2, 2)  | 0.56614809 | 0.54594142 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.35454439 | 0.34878069 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.41092525 | 0.38217443 | 0.61940299
# EfficientNetB0_Block3          | per frame (max) | SVG                                      | (1, 0, 0)  | 0.49502866 | 0.51321198 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.35501228 | 0.35702556 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.41806059 | 0.39289398 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | BalancedDistributionSVG/500/3/0.30       | (1, 0, 0)  | 0.49573049 | 0.51952695 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.37583343 | 0.35519604 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.46145748 | 0.40961445 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SVG                                      | (2, 0, 0)  | 0.49783600 | 0.51053929 | 0.61710037
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.37326003 | 0.35882885 | 0.61940299
# EfficientNetB0_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.41923032 | 0.38855672 | 0.62121212
# EfficientNetB0_Block3          | per frame (max) | BalancedDistributionSVG/500/3/0.30       | (2, 0, 0)  | 0.49830390 | 0.51268581 | 0.61710037
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.50146216 | 0.41379909 | 0.63157895
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.43069365 | 0.37652483 | 0.63358779
# EfficientNetB0_Block3          | per frame (sum) | SVG                                      | None       | 0.63258861 | 0.59198925 | 0.63673469
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.50497134 | 0.41499370 | 0.62650602
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.42075097 | 0.37557274 | 0.63358779
# EfficientNetB0_Block3          | per frame (sum) | BalancedDistributionSVG/500/3/0.30       | None       | 0.32576910 | 0.33857211 | 0.61710037
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.51994385 | 0.42410164 | 0.63157895
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.46917768 | 0.39478985 | 0.63601533
# EfficientNetB0_Block3          | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.62954732 | 0.58901643 | 0.64843750
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.52473974 | 0.42520381 | 0.63157895
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.43221429 | 0.38528763 | 0.63358779
# EfficientNetB0_Block3          | per frame (sum) | BalancedDistributionSVG/500/3/0.30       | (1, 1, 1)  | 0.31266815 | 0.33249899 | 0.61940299
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.54462510 | 0.43432741 | 0.64777328
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.49537958 | 0.40964063 | 0.63813230
# EfficientNetB0_Block3          | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.60884314 | 0.57122225 | 0.65612648
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.54439116 | 0.43247171 | 0.65040650
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.43946660 | 0.39079929 | 0.63358779
# EfficientNetB0_Block3          | per frame (sum) | BalancedDistributionSVG/500/3/0.30       | (2, 2, 2)  | 0.31009475 | 0.33134589 | 0.62406015
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.50146216 | 0.41379909 | 0.63157895
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.43069365 | 0.37652483 | 0.63358779
# EfficientNetB0_Block3          | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.63258861 | 0.59198925 | 0.63673469
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.50497134 | 0.41499370 | 0.62650602
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.42075097 | 0.37557274 | 0.63358779
# EfficientNetB0_Block3          | per frame (sum) | BalancedDistributionSVG/500/3/0.30       | (0, 1, 1)  | 0.32576910 | 0.33857211 | 0.61710037
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.50146216 | 0.41379909 | 0.63157895
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.43069365 | 0.37652483 | 0.63358779
# EfficientNetB0_Block3          | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.63258861 | 0.59198925 | 0.63673469
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.50497134 | 0.41499370 | 0.62650602
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.42075097 | 0.37557274 | 0.63358779
# EfficientNetB0_Block3          | per frame (sum) | BalancedDistributionSVG/500/3/0.30       | (0, 2, 2)  | 0.32576910 | 0.33857211 | 0.61710037
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.51994385 | 0.42410164 | 0.63157895
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.46917768 | 0.39478985 | 0.63601533
# EfficientNetB0_Block3          | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.62954732 | 0.58901643 | 0.64843750
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.52473974 | 0.42520381 | 0.63157895
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.43221429 | 0.38528763 | 0.63358779
# EfficientNetB0_Block3          | per frame (sum) | BalancedDistributionSVG/500/3/0.30       | (1, 0, 0)  | 0.31266815 | 0.33249899 | 0.61940299
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.54462510 | 0.43432741 | 0.64777328
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.49537958 | 0.40964063 | 0.63813230
# EfficientNetB0_Block3          | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.60884314 | 0.57122225 | 0.65612648
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.54439116 | 0.43247171 | 0.65040650
# EfficientNetB0_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.43946660 | 0.39079929 | 0.63358779
# EfficientNetB0_Block3          | per frame (sum) | BalancedDistributionSVG/500/3/0.30       | (2, 0, 0)  | 0.31009475 | 0.33134589 | 0.62406015
# EfficientNetB3                 | per patch       | SpatialBin/SVG/0.20                      | None       | 0.64252277 | 0.08816395 | 0.16345556
# EfficientNetB3                 | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.65861164 | 0.09817570 | 0.17858168
# EfficientNetB3                 | per patch       | SVG                                      | None       | 0.62336567 | 0.08057183 | 0.14889815
# EfficientNetB3                 | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.65522141 | 0.09804521 | 0.17716535
# EfficientNetB3                 | per patch       | SpatialBin/SVG/0.50                      | None       | 0.64267126 | 0.08754410 | 0.16311399
# EfficientNetB3                 | per patch       | BalancedDistributionSVG/500/39/0.30      | None       | 0.61857738 | 0.08238906 | 0.16218236
# EfficientNetB3                 | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.69784024 | 0.10642658 | 0.17381738
# EfficientNetB3                 | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.71474633 | 0.09983711 | 0.17542214
# EfficientNetB3                 | per patch       | SVG                                      | (1, 1, 1)  | 0.67048259 | 0.09359572 | 0.14659197
# EfficientNetB3                 | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.71301341 | 0.10514505 | 0.17746479
# EfficientNetB3                 | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.69763017 | 0.10574305 | 0.17340426
# EfficientNetB3                 | per patch       | BalancedDistributionSVG/500/39/0.30      | (1, 1, 1)  | 0.66945562 | 0.09818363 | 0.16129032
# EfficientNetB3                 | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.74741955 | 0.12685720 | 0.19374431
# EfficientNetB3                 | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.74032428 | 0.09581727 | 0.17193676
# EfficientNetB3                 | per patch       | SVG                                      | (2, 2, 2)  | 0.71120671 | 0.10800317 | 0.15000938
# EfficientNetB3                 | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.75333160 | 0.11440476 | 0.20201137
# EfficientNetB3                 | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.74713200 | 0.12586319 | 0.19282640
# EfficientNetB3                 | per patch       | BalancedDistributionSVG/500/39/0.30      | (2, 2, 2)  | 0.72203641 | 0.11771300 | 0.17799241
# EfficientNetB3                 | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.65986489 | 0.09500697 | 0.17783858
# EfficientNetB3                 | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.68140787 | 0.09799869 | 0.19295559
# EfficientNetB3                 | per patch       | SVG                                      | (0, 1, 1)  | 0.64068007 | 0.08686766 | 0.16294028
# EfficientNetB3                 | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.67480215 | 0.10324229 | 0.18935837
# EfficientNetB3                 | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.65981701 | 0.09414788 | 0.17576664
# EfficientNetB3                 | per patch       | BalancedDistributionSVG/500/39/0.30      | (0, 1, 1)  | 0.63173125 | 0.08963055 | 0.18468146
# EfficientNetB3                 | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.67782741 | 0.10351126 | 0.20654045
# EfficientNetB3                 | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.70264074 | 0.10305575 | 0.21508962
# EfficientNetB3                 | per patch       | SVG                                      | (0, 2, 2)  | 0.66130134 | 0.09927458 | 0.18970409
# EfficientNetB3                 | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.68897704 | 0.10932422 | 0.20563991
# EfficientNetB3                 | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.67833207 | 0.10284631 | 0.20550760
# EfficientNetB3                 | per patch       | BalancedDistributionSVG/500/39/0.30      | (0, 2, 2)  | 0.64612197 | 0.09997087 | 0.20384163
# EfficientNetB3                 | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.67263507 | 0.09718719 | 0.15739034
# EfficientNetB3                 | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.69483604 | 0.10430904 | 0.17085020
# EfficientNetB3                 | per patch       | SVG                                      | (1, 0, 0)  | 0.64609153 | 0.08639101 | 0.13479574
# EfficientNetB3                 | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.69455463 | 0.10600200 | 0.17098233
# EfficientNetB3                 | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.67250053 | 0.09669587 | 0.15667030
# EfficientNetB3                 | per patch       | BalancedDistributionSVG/500/39/0.30      | (1, 0, 0)  | 0.63999777 | 0.08770585 | 0.14019417
# EfficientNetB3                 | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.67328389 | 0.09471222 | 0.14928124
# EfficientNetB3                 | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.70421586 | 0.10481700 | 0.18438603
# EfficientNetB3                 | per patch       | SVG                                      | (2, 0, 0)  | 0.64204350 | 0.08277153 | 0.13118313
# EfficientNetB3                 | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.70402298 | 0.10647278 | 0.18360141
# EfficientNetB3                 | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.67295964 | 0.09419343 | 0.14936520
# EfficientNetB3                 | per patch       | BalancedDistributionSVG/500/39/0.30      | (2, 0, 0)  | 0.63254692 | 0.08355272 | 0.12984925
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.62147620 | 0.62144830 | 0.64573991
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.58790502 | 0.49565072 | 0.64220183
# EfficientNetB3                 | per frame (max) | SVG                                      | None       | 0.50871447 | 0.57528335 | 0.61710037
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.56684992 | 0.48495624 | 0.63348416
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.61387297 | 0.61706899 | 0.63478261
# EfficientNetB3                 | per frame (max) | BalancedDistributionSVG/500/39/0.30      | None       | 0.51280852 | 0.57794282 | 0.61710037
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.68592818 | 0.68151995 | 0.66292135
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.60814130 | 0.46845692 | 0.68161435
# EfficientNetB3                 | per frame (max) | SVG                                      | (1, 1, 1)  | 0.67306118 | 0.64753410 | 0.67027027
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.62112528 | 0.53218146 | 0.67555556
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.67645339 | 0.67946438 | 0.66292135
# EfficientNetB3                 | per frame (max) | BalancedDistributionSVG/500/39/0.30      | (1, 1, 1)  | 0.72640075 | 0.70663099 | 0.68544601
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.70932273 | 0.67678401 | 0.66960352
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.58334308 | 0.44881355 | 0.67241379
# EfficientNetB3                 | per frame (max) | SVG                                      | (2, 2, 2)  | 0.70967365 | 0.65277830 | 0.67248908
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.61305416 | 0.51472576 | 0.67841410
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.70546263 | 0.67496324 | 0.66666667
# EfficientNetB3                 | per frame (max) | BalancedDistributionSVG/500/39/0.30      | (2, 2, 2)  | 0.74967832 | 0.72219978 | 0.68599034
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.52111358 | 0.58526854 | 0.61710037
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.57573985 | 0.49788279 | 0.62500000
# EfficientNetB3                 | per frame (max) | SVG                                      | (0, 1, 1)  | 0.55562054 | 0.59592352 | 0.61710037
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.56521231 | 0.50452606 | 0.62595420
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.51093695 | 0.58233064 | 0.61710037
# EfficientNetB3                 | per frame (max) | BalancedDistributionSVG/500/39/0.30      | (0, 1, 1)  | 0.55924670 | 0.59983565 | 0.61710037
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.50309978 | 0.57671586 | 0.62641509
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.55222833 | 0.50279692 | 0.62641509
# EfficientNetB3                 | per frame (max) | SVG                                      | (0, 2, 2)  | 0.51397824 | 0.58090563 | 0.61710037
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.52965259 | 0.49815896 | 0.62406015
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.49795298 | 0.57497978 | 0.62121212
# EfficientNetB3                 | per frame (max) | BalancedDistributionSVG/500/39/0.30      | (0, 2, 2)  | 0.51982688 | 0.58378814 | 0.61710037
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.71751082 | 0.68349238 | 0.67811159
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.60322845 | 0.46377934 | 0.68907563
# EfficientNetB3                 | per frame (max) | SVG                                      | (1, 0, 0)  | 0.68896947 | 0.64209658 | 0.66666667
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.56954030 | 0.48450342 | 0.67219917
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.71669201 | 0.68597715 | 0.67521368
# EfficientNetB3                 | per frame (max) | BalancedDistributionSVG/500/39/0.30      | (1, 0, 0)  | 0.71879752 | 0.68997470 | 0.68449198
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.75388934 | 0.69231492 | 0.69827586
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.56532928 | 0.43411113 | 0.70742358
# EfficientNetB3                 | per frame (max) | SVG                                      | (2, 0, 0)  | 0.73540765 | 0.65558479 | 0.70531401
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.57035911 | 0.48002122 | 0.68644068
# EfficientNetB3                 | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.75342145 | 0.69051314 | 0.70129870
# EfficientNetB3                 | per frame (max) | BalancedDistributionSVG/500/39/0.30      | (2, 0, 0)  | 0.76207744 | 0.70944987 | 0.71921182
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.53678793 | 0.59488975 | 0.62641509
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.58006784 | 0.53725638 | 0.62641509
# EfficientNetB3                 | per frame (sum) | SVG                                      | None       | 0.61820096 | 0.63498747 | 0.62121212
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.54673061 | 0.52289284 | 0.62641509
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.51432916 | 0.58653907 | 0.62406015
# EfficientNetB3                 | per frame (sum) | BalancedDistributionSVG/500/39/0.30      | None       | 0.63457714 | 0.64318527 | 0.62121212
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.69189379 | 0.68219510 | 0.66981132
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.63293953 | 0.51596048 | 0.66079295
# EfficientNetB3                 | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.71470347 | 0.65654223 | 0.70000000
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.64335010 | 0.56896202 | 0.64935065
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.68768277 | 0.67971366 | 0.66976744
# EfficientNetB3                 | per frame (sum) | BalancedDistributionSVG/500/39/0.30      | (1, 1, 1)  | 0.74020353 | 0.71444493 | 0.69406393
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.72335946 | 0.67399886 | 0.68156425
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.61094865 | 0.47558109 | 0.67841410
# EfficientNetB3                 | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.72441221 | 0.63701799 | 0.68686869
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.64218037 | 0.54794172 | 0.67826087
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.72043514 | 0.67009845 | 0.67841410
# EfficientNetB3                 | per frame (sum) | BalancedDistributionSVG/500/39/0.30      | (2, 2, 2)  | 0.75728155 | 0.71612913 | 0.69306931
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.53678793 | 0.59488975 | 0.62641509
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.58006784 | 0.53725638 | 0.62641509
# EfficientNetB3                 | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.61820096 | 0.63498747 | 0.62121212
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.54673061 | 0.52289284 | 0.62641509
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.51432916 | 0.58653907 | 0.62406015
# EfficientNetB3                 | per frame (sum) | BalancedDistributionSVG/500/39/0.30      | (0, 1, 1)  | 0.63457714 | 0.64318527 | 0.62121212
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.53678793 | 0.59488975 | 0.62641509
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.58006784 | 0.53725638 | 0.62641509
# EfficientNetB3                 | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.61820096 | 0.63498747 | 0.62121212
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.54673061 | 0.52289284 | 0.62641509
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.51432916 | 0.58653907 | 0.62406015
# EfficientNetB3                 | per frame (sum) | BalancedDistributionSVG/500/39/0.30      | (0, 2, 2)  | 0.63457714 | 0.64318527 | 0.62121212
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.69189379 | 0.68219510 | 0.66981132
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.63293953 | 0.51596048 | 0.66079295
# EfficientNetB3                 | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.71470347 | 0.65654223 | 0.70000000
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.64335010 | 0.56896202 | 0.64935065
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.68768277 | 0.67971366 | 0.66976744
# EfficientNetB3                 | per frame (sum) | BalancedDistributionSVG/500/39/0.30      | (1, 0, 0)  | 0.74020353 | 0.71444493 | 0.69406393
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.72335946 | 0.67399886 | 0.68156425
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.61094865 | 0.47558109 | 0.67841410
# EfficientNetB3                 | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.72441221 | 0.63701799 | 0.68686869
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.64218037 | 0.54794172 | 0.67826087
# EfficientNetB3                 | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.72043514 | 0.67009845 | 0.67841410
# EfficientNetB3                 | per frame (sum) | BalancedDistributionSVG/500/39/0.30      | (2, 0, 0)  | 0.75728155 | 0.71612913 | 0.69306931
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/0.20                      | None       | 0.65185930 | 0.08660308 | 0.16855412
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.67433858 | 0.09433399 | 0.16585704
# EfficientNetB3_Block6          | per patch       | SVG                                      | None       | 0.64319996 | 0.08212562 | 0.16140777
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/0.50                      | None       | 0.65152232 | 0.08650636 | 0.16877637
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.67172015 | 0.09123775 | 0.16122623
# EfficientNetB3_Block6          | per patch       | BalancedDistributionSVG/500/14/0.30      | None       | 0.62961606 | 0.08165099 | 0.15767417
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.68724186 | 0.09215591 | 0.17217788
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.71876741 | 0.09932088 | 0.18137580
# EfficientNetB3_Block6          | per patch       | SVG                                      | (1, 1, 1)  | 0.67434737 | 0.08628177 | 0.16101878
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.68700743 | 0.09193853 | 0.17121729
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.71713963 | 0.10315006 | 0.17582418
# EfficientNetB3_Block6          | per patch       | BalancedDistributionSVG/500/14/0.30      | (1, 1, 1)  | 0.68869253 | 0.09550573 | 0.17976760
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.71444631 | 0.09876040 | 0.17452007
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.75002661 | 0.10212177 | 0.18301887
# EfficientNetB3_Block6          | per patch       | SVG                                      | (2, 2, 2)  | 0.69760465 | 0.08996697 | 0.16048144
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.71418280 | 0.09840329 | 0.17363572
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.76488044 | 0.11618348 | 0.19090327
# EfficientNetB3_Block6          | per patch       | BalancedDistributionSVG/500/14/0.30      | (2, 2, 2)  | 0.73826059 | 0.10523248 | 0.18415283
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.67014348 | 0.09231145 | 0.17108793
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.69974452 | 0.09672827 | 0.17541692
# EfficientNetB3_Block6          | per patch       | SVG                                      | (0, 1, 1)  | 0.66148311 | 0.08728244 | 0.16047633
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.66988733 | 0.09216212 | 0.17068094
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.69136117 | 0.09903902 | 0.16290357
# EfficientNetB3_Block6          | per patch       | BalancedDistributionSVG/500/14/0.30      | (0, 1, 1)  | 0.66550601 | 0.09343850 | 0.18457060
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.69200418 | 0.10103640 | 0.18598455
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.71373377 | 0.10019185 | 0.17413114
# EfficientNetB3_Block6          | per patch       | SVG                                      | (0, 2, 2)  | 0.67949887 | 0.09468998 | 0.17144463
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.69173615 | 0.10071661 | 0.18641390
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.69470513 | 0.10845299 | 0.16049383
# EfficientNetB3_Block6          | per patch       | BalancedDistributionSVG/500/14/0.30      | (0, 2, 2)  | 0.68346930 | 0.10036581 | 0.17578980
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.66441016 | 0.08687099 | 0.17137861
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.69351784 | 0.09707160 | 0.17041261
# EfficientNetB3_Block6          | per patch       | SVG                                      | (1, 0, 0)  | 0.65259749 | 0.08225237 | 0.15850448
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.66412746 | 0.08676359 | 0.17064648
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.69250871 | 0.09571452 | 0.16861827
# EfficientNetB3_Block6          | per patch       | BalancedDistributionSVG/500/14/0.30      | (1, 0, 0)  | 0.64620274 | 0.08346607 | 0.15692175
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.66595814 | 0.08497850 | 0.15690799
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.70451511 | 0.09507476 | 0.17170552
# EfficientNetB3_Block6          | per patch       | SVG                                      | (2, 0, 0)  | 0.65228571 | 0.08107963 | 0.15315061
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.66560048 | 0.08487243 | 0.15657143
# EfficientNetB3_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.70542893 | 0.09530409 | 0.16971945
# EfficientNetB3_Block6          | per patch       | BalancedDistributionSVG/500/14/0.30      | (2, 0, 0)  | 0.64671787 | 0.08213414 | 0.15117158
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.59971927 | 0.52335953 | 0.63677130
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.56942332 | 0.47338915 | 0.62439024
# EfficientNetB3_Block6          | per frame (max) | SVG                                      | None       | 0.66311849 | 0.56923649 | 0.67010309
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.59890046 | 0.52247173 | 0.63507109
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.51163879 | 0.44977297 | 0.61710037
# EfficientNetB3_Block6          | per frame (max) | BalancedDistributionSVG/500/14/0.30      | None       | 0.65820564 | 0.56528289 | 0.66666667
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.64229734 | 0.52812694 | 0.68141593
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.55690724 | 0.44759672 | 0.62790698
# EfficientNetB3_Block6          | per frame (max) | SVG                                      | (1, 1, 1)  | 0.64884782 | 0.51377858 | 0.67219917
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.64604047 | 0.53106363 | 0.68224299
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.48181074 | 0.44339911 | 0.62172285
# EfficientNetB3_Block6          | per frame (max) | BalancedDistributionSVG/500/14/0.30      | (1, 1, 1)  | 0.65493040 | 0.52690034 | 0.67206478
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.63562990 | 0.52176477 | 0.69166667
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.55105860 | 0.45092737 | 0.65236052
# EfficientNetB3_Block6          | per frame (max) | SVG                                      | (2, 2, 2)  | 0.63141888 | 0.49601047 | 0.67889908
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.63668265 | 0.52238823 | 0.69166667
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.50602410 | 0.46007211 | 0.62878788
# EfficientNetB3_Block6          | per frame (max) | BalancedDistributionSVG/500/14/0.30      | (2, 2, 2)  | 0.64077670 | 0.52231949 | 0.68202765
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.61972160 | 0.54184290 | 0.66960352
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.57457013 | 0.47574950 | 0.62616822
# EfficientNetB3_Block6          | per frame (max) | SVG                                      | (0, 1, 1)  | 0.65376067 | 0.56457361 | 0.67841410
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.62182711 | 0.54190009 | 0.66375546
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.53257691 | 0.46556655 | 0.61710037
# EfficientNetB3_Block6          | per frame (max) | BalancedDistributionSVG/500/14/0.30      | (0, 1, 1)  | 0.65387765 | 0.56407863 | 0.67841410
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.63188677 | 0.55303532 | 0.66960352
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.58626740 | 0.48092214 | 0.63302752
# EfficientNetB3_Block6          | per frame (max) | SVG                                      | (0, 2, 2)  | 0.65586618 | 0.56568686 | 0.67543860
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.63247163 | 0.55267510 | 0.66371681
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.55690724 | 0.48138784 | 0.62184874
# EfficientNetB3_Block6          | per frame (max) | BalancedDistributionSVG/500/14/0.30      | (0, 2, 2)  | 0.65773775 | 0.56693650 | 0.67543860
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.62545327 | 0.51294418 | 0.68695652
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.55573751 | 0.44654854 | 0.64114833
# EfficientNetB3_Block6          | per frame (max) | SVG                                      | (1, 0, 0)  | 0.63796935 | 0.50600308 | 0.67241379
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.62720786 | 0.51384434 | 0.68965517
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.45233361 | 0.42111009 | 0.62641509
# EfficientNetB3_Block6          | per frame (max) | BalancedDistributionSVG/500/14/0.30      | (1, 0, 0)  | 0.63504503 | 0.50508090 | 0.67532468
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.60498304 | 0.48420963 | 0.69456067
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.52824892 | 0.42525503 | 0.63461538
# EfficientNetB3_Block6          | per frame (max) | SVG                                      | (2, 0, 0)  | 0.60533396 | 0.47643963 | 0.66968326
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.60545093 | 0.48452919 | 0.69456067
# EfficientNetB3_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.45631068 | 0.42958241 | 0.62406015
# EfficientNetB3_Block6          | per frame (max) | BalancedDistributionSVG/500/14/0.30      | (2, 0, 0)  | 0.59925137 | 0.47437165 | 0.67264574
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.64744415 | 0.58136136 | 0.66960352
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.61083168 | 0.51868570 | 0.64485981
# EfficientNetB3_Block6          | per frame (sum) | SVG                                      | None       | 0.66861621 | 0.59448252 | 0.66666667
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.64732717 | 0.57896266 | 0.66960352
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.57492104 | 0.51833445 | 0.62809917
# EfficientNetB3_Block6          | per frame (sum) | BalancedDistributionSVG/500/14/0.30      | None       | 0.67996257 | 0.63645133 | 0.66382979
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.67633641 | 0.58567699 | 0.67489712
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.62615511 | 0.51253240 | 0.67510549
# EfficientNetB3_Block6          | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.67118961 | 0.55444433 | 0.67647059
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.67481577 | 0.58395102 | 0.67755102
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.60217569 | 0.53092421 | 0.64220183
# EfficientNetB3_Block6          | per frame (sum) | BalancedDistributionSVG/500/14/0.30      | (1, 1, 1)  | 0.68838461 | 0.59914440 | 0.67317073
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.66510703 | 0.56046387 | 0.68444444
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.59515733 | 0.49535806 | 0.68468468
# EfficientNetB3_Block6          | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.66077904 | 0.53652136 | 0.67826087
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.66323547 | 0.55883643 | 0.68141593
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.59714587 | 0.51517509 | 0.66115702
# EfficientNetB3_Block6          | per frame (sum) | BalancedDistributionSVG/500/14/0.30      | (2, 2, 2)  | 0.68663002 | 0.60220597 | 0.68161435
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.64744415 | 0.58136136 | 0.66960352
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.61083168 | 0.51868570 | 0.64485981
# EfficientNetB3_Block6          | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.66861621 | 0.59448252 | 0.66666667
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.64732717 | 0.57896266 | 0.66960352
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.57492104 | 0.51833445 | 0.62809917
# EfficientNetB3_Block6          | per frame (sum) | BalancedDistributionSVG/500/14/0.30      | (0, 1, 1)  | 0.67996257 | 0.63645133 | 0.66382979
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.64744415 | 0.58136136 | 0.66960352
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.61083168 | 0.51868570 | 0.64485981
# EfficientNetB3_Block6          | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.66861621 | 0.59448252 | 0.66666667
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.64732717 | 0.57896266 | 0.66960352
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.57492104 | 0.51833445 | 0.62809917
# EfficientNetB3_Block6          | per frame (sum) | BalancedDistributionSVG/500/14/0.30      | (0, 2, 2)  | 0.67996257 | 0.63645133 | 0.66382979
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.67633641 | 0.58567699 | 0.67489712
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.62615511 | 0.51253240 | 0.67510549
# EfficientNetB3_Block6          | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.67118961 | 0.55444433 | 0.67647059
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.67481577 | 0.58395102 | 0.67755102
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.60217569 | 0.53092421 | 0.64220183
# EfficientNetB3_Block6          | per frame (sum) | BalancedDistributionSVG/500/14/0.30      | (1, 0, 0)  | 0.68838461 | 0.59914440 | 0.67317073
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.66510703 | 0.56046387 | 0.68444444
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.59515733 | 0.49535806 | 0.68468468
# EfficientNetB3_Block6          | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.66077904 | 0.53652136 | 0.67826087
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.66323547 | 0.55883643 | 0.68141593
# EfficientNetB3_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.59714587 | 0.51517509 | 0.66115702
# EfficientNetB3_Block6          | per frame (sum) | BalancedDistributionSVG/500/14/0.30      | (2, 0, 0)  | 0.68663002 | 0.60220597 | 0.68161435
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/0.20                      | None       | 0.53762649 | 0.04864920 | 0.10112710
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.54729253 | 0.04877646 | 0.10323917
# EfficientNetB6_Block5          | per patch       | SVG                                      | None       | 0.53763538 | 0.04869377 | 0.10139920
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/0.50                      | None       | 0.53769866 | 0.04863195 | 0.10114059
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.54749441 | 0.04886940 | 0.10284140
# EfficientNetB6_Block5          | per patch       | BalancedDistributionSVG/500/13/0.30      | None       | 0.51718111 | 0.04864509 | 0.09514644
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.53524817 | 0.04825078 | 0.10117868
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.54605618 | 0.05127961 | 0.10357614
# EfficientNetB6_Block5          | per patch       | SVG                                      | (1, 1, 1)  | 0.53531691 | 0.04826773 | 0.10113458
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.53534645 | 0.04833531 | 0.10116531
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.54692508 | 0.05145494 | 0.10379054
# EfficientNetB6_Block5          | per patch       | BalancedDistributionSVG/500/13/0.30      | (1, 1, 1)  | 0.50822761 | 0.04703628 | 0.09656144
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.52874750 | 0.04730290 | 0.10063512
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.53161476 | 0.04941707 | 0.10208326
# EfficientNetB6_Block5          | per patch       | SVG                                      | (2, 2, 2)  | 0.52906731 | 0.04737010 | 0.10060399
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.52887040 | 0.04734470 | 0.10063370
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.53529333 | 0.05090433 | 0.10210210
# EfficientNetB6_Block5          | per patch       | BalancedDistributionSVG/500/13/0.30      | (2, 2, 2)  | 0.50200951 | 0.04652841 | 0.09736161
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.53493013 | 0.04831476 | 0.10120970
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.54105601 | 0.04792096 | 0.10274544
# EfficientNetB6_Block5          | per patch       | SVG                                      | (0, 1, 1)  | 0.53493768 | 0.04830282 | 0.10116777
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.53503355 | 0.04836870 | 0.10120409
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.54064756 | 0.04786791 | 0.10296658
# EfficientNetB6_Block5          | per patch       | BalancedDistributionSVG/500/13/0.30      | (0, 1, 1)  | 0.50831404 | 0.04723076 | 0.09671893
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.52987705 | 0.04768532 | 0.10106061
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.53366271 | 0.04695965 | 0.10260040
# EfficientNetB6_Block5          | per patch       | SVG                                      | (0, 2, 2)  | 0.53012082 | 0.04775393 | 0.10115472
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.52998992 | 0.04769564 | 0.10103892
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.53388953 | 0.04715930 | 0.10258487
# EfficientNetB6_Block5          | per patch       | BalancedDistributionSVG/500/13/0.30      | (0, 2, 2)  | 0.50494414 | 0.04758939 | 0.09780061
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.53807544 | 0.04838488 | 0.10111562
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.55224568 | 0.05173548 | 0.10311522
# EfficientNetB6_Block5          | per patch       | SVG                                      | (1, 0, 0)  | 0.53820925 | 0.04838389 | 0.10140931
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.53816130 | 0.04842094 | 0.10111086
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.55313082 | 0.05176087 | 0.10291555
# EfficientNetB6_Block5          | per patch       | BalancedDistributionSVG/500/13/0.30      | (1, 0, 0)  | 0.51759253 | 0.04853292 | 0.09499704
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.53748765 | 0.04827517 | 0.10067998
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.54776581 | 0.05103888 | 0.10292168
# EfficientNetB6_Block5          | per patch       | SVG                                      | (2, 0, 0)  | 0.53760468 | 0.04826864 | 0.10087059
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.53757940 | 0.04830609 | 0.10066520
# EfficientNetB6_Block5          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.54858771 | 0.05115988 | 0.10259303
# EfficientNetB6_Block5          | per patch       | BalancedDistributionSVG/500/13/0.30      | (2, 0, 0)  | 0.51600895 | 0.04822201 | 0.09480994
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.41525325 | 0.41491875 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.40449175 | 0.38519569 | 0.62172285
# EfficientNetB6_Block5          | per frame (max) | SVG                                      | None       | 0.41034039 | 0.41458380 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.42344134 | 0.43569651 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.39314540 | 0.42362035 | 0.61940299
# EfficientNetB6_Block5          | per frame (max) | BalancedDistributionSVG/500/13/0.30      | None       | 0.47034741 | 0.41709113 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.51269154 | 0.41435565 | 0.63076923
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.48181074 | 0.41374135 | 0.63358779
# EfficientNetB6_Block5          | per frame (max) | SVG                                      | (1, 1, 1)  | 0.50532226 | 0.41784877 | 0.63414634
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.55094163 | 0.46794229 | 0.63565891
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.56848754 | 0.53292998 | 0.61940299
# EfficientNetB6_Block5          | per frame (max) | BalancedDistributionSVG/500/13/0.30      | (1, 1, 1)  | 0.54404024 | 0.44856019 | 0.62601626
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.55316411 | 0.55098047 | 0.63601533
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.39244356 | 0.37629361 | 0.62406015
# EfficientNetB6_Block5          | per frame (max) | SVG                                      | (2, 2, 2)  | 0.58579951 | 0.56921712 | 0.63601533
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.56263891 | 0.56700410 | 0.64705882
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.56100129 | 0.52269403 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | BalancedDistributionSVG/500/13/0.30      | (2, 2, 2)  | 0.57807931 | 0.57245883 | 0.64347826
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.51818926 | 0.43295559 | 0.63247863
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.49245526 | 0.42300157 | 0.62406015
# EfficientNetB6_Block5          | per frame (max) | SVG                                      | (0, 1, 1)  | 0.49970757 | 0.42735735 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.54322143 | 0.47379237 | 0.63755459
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.46695520 | 0.44088993 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | BalancedDistributionSVG/500/13/0.30      | (0, 1, 1)  | 0.54766639 | 0.49057741 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.54029711 | 0.53582374 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.44917534 | 0.39798419 | 0.62641509
# EfficientNetB6_Block5          | per frame (max) | SVG                                      | (0, 2, 2)  | 0.53585215 | 0.53197065 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.55737513 | 0.55297026 | 0.62831858
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.43584045 | 0.42814860 | 0.61940299
# EfficientNetB6_Block5          | per frame (max) | BalancedDistributionSVG/500/13/0.30      | (0, 2, 2)  | 0.53269388 | 0.53567942 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.38647795 | 0.38100316 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.44812259 | 0.40543295 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | SVG                                      | (1, 0, 0)  | 0.39759036 | 0.39044226 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.38448941 | 0.38176607 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.56626506 | 0.55261455 | 0.63846154
# EfficientNetB6_Block5          | per frame (max) | BalancedDistributionSVG/500/13/0.30      | (1, 0, 0)  | 0.44040239 | 0.39960272 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.39326237 | 0.39191178 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.44040239 | 0.39915664 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | SVG                                      | (2, 0, 0)  | 0.42040005 | 0.41326902 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.38308574 | 0.38717340 | 0.61710037
# EfficientNetB6_Block5          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.57281553 | 0.55057556 | 0.62172285
# EfficientNetB6_Block5          | per frame (max) | BalancedDistributionSVG/500/13/0.30      | (2, 0, 0)  | 0.44262487 | 0.39726915 | 0.61710037
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.55901275 | 0.55736566 | 0.63900415
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.50824658 | 0.42022569 | 0.62878788
# EfficientNetB6_Block5          | per frame (sum) | SVG                                      | None       | 0.49409288 | 0.54936573 | 0.62595420
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.57609077 | 0.56344731 | 0.64435146
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.50567318 | 0.44667491 | 0.62172285
# EfficientNetB6_Block5          | per frame (sum) | BalancedDistributionSVG/500/13/0.30      | None       | 0.50134519 | 0.52142214 | 0.62406015
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.57468710 | 0.57726984 | 0.64062500
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.52532460 | 0.42752449 | 0.63255814
# EfficientNetB6_Block5          | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.52181542 | 0.56185461 | 0.61940299
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.59562522 | 0.58262462 | 0.65178571
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.57340040 | 0.49425786 | 0.64102564
# EfficientNetB6_Block5          | per frame (sum) | BalancedDistributionSVG/500/13/0.30      | (1, 1, 1)  | 0.52626038 | 0.55264099 | 0.61710037
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.57948298 | 0.58409214 | 0.64566929
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.52029477 | 0.42704422 | 0.63677130
# EfficientNetB6_Block5          | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.58556556 | 0.61005072 | 0.62962963
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.60170780 | 0.59142431 | 0.64092664
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.53667096 | 0.46893613 | 0.63967611
# EfficientNetB6_Block5          | per frame (sum) | BalancedDistributionSVG/500/13/0.30      | (2, 2, 2)  | 0.57807931 | 0.59559409 | 0.62385321
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.55901275 | 0.55736566 | 0.63900415
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.50824658 | 0.42022569 | 0.62878788
# EfficientNetB6_Block5          | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.49409288 | 0.54936573 | 0.62595420
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.57609077 | 0.56344731 | 0.64435146
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.50567318 | 0.44667491 | 0.62172285
# EfficientNetB6_Block5          | per frame (sum) | BalancedDistributionSVG/500/13/0.30      | (0, 1, 1)  | 0.50134519 | 0.52142214 | 0.62406015
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.55901275 | 0.55736566 | 0.63900415
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.50824658 | 0.42022569 | 0.62878788
# EfficientNetB6_Block5          | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.49409288 | 0.54936573 | 0.62595420
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.57609077 | 0.56344731 | 0.64435146
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.50567318 | 0.44667491 | 0.62172285
# EfficientNetB6_Block5          | per frame (sum) | BalancedDistributionSVG/500/13/0.30      | (0, 2, 2)  | 0.50134519 | 0.52142214 | 0.62406015
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.57468710 | 0.57726984 | 0.64062500
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.52532460 | 0.42752449 | 0.63255814
# EfficientNetB6_Block5          | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.52181542 | 0.56185461 | 0.61940299
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.59562522 | 0.58262462 | 0.65178571
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.57340040 | 0.49425786 | 0.64102564
# EfficientNetB6_Block5          | per frame (sum) | BalancedDistributionSVG/500/13/0.30      | (1, 0, 0)  | 0.52626038 | 0.55264099 | 0.61710037
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.57948298 | 0.58409214 | 0.64566929
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.52029477 | 0.42704422 | 0.63677130
# EfficientNetB6_Block5          | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.58556556 | 0.61005072 | 0.62962963
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.60170780 | 0.59142431 | 0.64092664
# EfficientNetB6_Block5          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.53667096 | 0.46893613 | 0.63967611
# EfficientNetB6_Block5          | per frame (sum) | BalancedDistributionSVG/500/13/0.30      | (2, 0, 0)  | 0.57807931 | 0.59559409 | 0.62385321
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/0.20                      | None       | 0.84041113 | 0.12646602 | 0.24840183
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.87858143 | 0.17036961 | 0.30802369
# MobileNetV2_Block14            | per patch       | BalancedDistributionSVG/500/11/0.30      | None       | 0.84151929 | 0.12986487 | 0.26011824
# MobileNetV2_Block14            | per patch       | SVG                                      | None       | 0.82975052 | 0.11861917 | 0.23721141
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/0.50                      | None       | 0.84054476 | 0.12655611 | 0.24895105
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.87913565 | 0.18449823 | 0.31905076
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.82710545 | 0.12150116 | 0.22550053
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.88146417 | 0.18608467 | 0.30265849
# MobileNetV2_Block14            | per patch       | BalancedDistributionSVG/500/11/0.30      | (1, 1, 1)  | 0.83174911 | 0.12569841 | 0.22798635
# MobileNetV2_Block14            | per patch       | SVG                                      | (1, 1, 1)  | 0.81291094 | 0.11105615 | 0.21618283
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.82754866 | 0.12191475 | 0.22603219
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.87860781 | 0.21960285 | 0.30098599
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.87919670 | 0.29369014 | 0.34088849
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.89912938 | 0.32965418 | 0.35564054
# MobileNetV2_Block14            | per patch       | BalancedDistributionSVG/500/11/0.30      | (2, 2, 2)  | 0.87645550 | 0.27622317 | 0.32203390
# MobileNetV2_Block14            | per patch       | SVG                                      | (2, 2, 2)  | 0.86623301 | 0.22609654 | 0.29778096
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.87942675 | 0.29540975 | 0.34154930
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.87972588 | 0.32611330 | 0.34206897
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.84271502 | 0.13463931 | 0.24676259
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.89644280 | 0.21910340 | 0.34634146
# MobileNetV2_Block14            | per patch       | BalancedDistributionSVG/500/11/0.30      | (0, 1, 1)  | 0.84495945 | 0.14019685 | 0.25000000
# MobileNetV2_Block14            | per patch       | SVG                                      | (0, 1, 1)  | 0.82776344 | 0.12139517 | 0.23272727
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.84298978 | 0.13502694 | 0.24747475
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.88960826 | 0.24151220 | 0.34679335
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.90042527 | 0.36700724 | 0.41651032
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.92571556 | 0.41461508 | 0.47389558
# MobileNetV2_Block14            | per patch       | BalancedDistributionSVG/500/11/0.30      | (0, 2, 2)  | 0.89488286 | 0.34112534 | 0.38294314
# MobileNetV2_Block14            | per patch       | SVG                                      | (0, 2, 2)  | 0.88817097 | 0.28485755 | 0.35961680
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.90042393 | 0.36812452 | 0.41747573
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.90899462 | 0.37244272 | 0.43618090
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.82620617 | 0.11636074 | 0.23106192
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.86505629 | 0.15065233 | 0.28267800
# MobileNetV2_Block14            | per patch       | BalancedDistributionSVG/500/11/0.30      | (1, 0, 0)  | 0.83204396 | 0.12080464 | 0.24306874
# MobileNetV2_Block14            | per patch       | SVG                                      | (1, 0, 0)  | 0.81610098 | 0.10986453 | 0.22360248
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.82641372 | 0.11650055 | 0.23116973
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.86630130 | 0.16394025 | 0.29434698
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.80057845 | 0.10365195 | 0.21134021
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.83675511 | 0.12831228 | 0.24730354
# MobileNetV2_Block14            | per patch       | BalancedDistributionSVG/500/11/0.30      | (2, 0, 0)  | 0.80737952 | 0.10786212 | 0.21595598
# MobileNetV2_Block14            | per patch       | SVG                                      | (2, 0, 0)  | 0.79180851 | 0.09904902 | 0.20422347
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.80086928 | 0.10384828 | 0.21100656
# MobileNetV2_Block14            | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.83701381 | 0.14166417 | 0.25590872
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.57620774 | 0.47683116 | 0.64197531
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.48918002 | 0.41485329 | 0.63414634
# MobileNetV2_Block14            | per frame (max) | BalancedDistributionSVG/500/11/0.30      | None       | 0.54708153 | 0.47673478 | 0.63200000
# MobileNetV2_Block14            | per frame (max) | SVG                                      | None       | 0.55269622 | 0.47776192 | 0.63414634
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.57901509 | 0.48147055 | 0.64462810
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.52427184 | 0.49509122 | 0.62295082
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.58100363 | 0.47899497 | 0.65354331
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.51970991 | 0.43929659 | 0.63813230
# MobileNetV2_Block14            | per frame (max) | BalancedDistributionSVG/500/11/0.30      | (1, 1, 1)  | 0.54345537 | 0.46954112 | 0.63241107
# MobileNetV2_Block14            | per frame (max) | SVG                                      | (1, 1, 1)  | 0.55445081 | 0.47381887 | 0.63846154
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.58229033 | 0.48388901 | 0.65098039
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.55737513 | 0.53374568 | 0.63813230
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.72078606 | 0.65452112 | 0.69945355
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.67633641 | 0.60828940 | 0.66666667
# MobileNetV2_Block14            | per frame (max) | BalancedDistributionSVG/500/11/0.30      | (2, 2, 2)  | 0.66534098 | 0.58894844 | 0.66666667
# MobileNetV2_Block14            | per frame (max) | SVG                                      | (2, 2, 2)  | 0.65691894 | 0.53997120 | 0.66304348
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.72031817 | 0.65454910 | 0.70270270
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.66007720 | 0.63913125 | 0.67256637
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.56673295 | 0.47525052 | 0.64566929
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.52427184 | 0.43601792 | 0.63358779
# MobileNetV2_Block14            | per frame (max) | BalancedDistributionSVG/500/11/0.30      | (0, 1, 1)  | 0.54684758 | 0.46443714 | 0.62878788
# MobileNetV2_Block14            | per frame (max) | SVG                                      | (0, 1, 1)  | 0.54287051 | 0.46443036 | 0.62835249
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.56918938 | 0.47544423 | 0.64313725
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.55924670 | 0.50273382 | 0.63846154
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.70476079 | 0.65722606 | 0.68571429
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.71259796 | 0.64014461 | 0.68493151
# MobileNetV2_Block14            | per frame (max) | BalancedDistributionSVG/500/11/0.30      | (0, 2, 2)  | 0.64416891 | 0.59236844 | 0.65641026
# MobileNetV2_Block14            | per frame (max) | SVG                                      | (0, 2, 2)  | 0.63703357 | 0.55899360 | 0.66009852
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.70499474 | 0.66210560 | 0.68899522
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.70791905 | 0.65570506 | 0.69035533
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.58088665 | 0.45924701 | 0.65338645
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.47970523 | 0.41653366 | 0.63117871
# MobileNetV2_Block14            | per frame (max) | BalancedDistributionSVG/500/11/0.30      | (1, 0, 0)  | 0.56065037 | 0.46539913 | 0.64591440
# MobileNetV2_Block14            | per frame (max) | SVG                                      | (1, 0, 0)  | 0.56240496 | 0.46344835 | 0.64843750
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.57889812 | 0.45794965 | 0.65338645
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.53129021 | 0.51639096 | 0.62835249
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.59094631 | 0.47367936 | 0.66666667
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.49596444 | 0.42898850 | 0.61940299
# MobileNetV2_Block14            | per frame (max) | BalancedDistributionSVG/500/11/0.30      | (2, 0, 0)  | 0.56720084 | 0.47418094 | 0.64843750
# MobileNetV2_Block14            | per frame (max) | SVG                                      | (2, 0, 0)  | 0.57340040 | 0.47489322 | 0.65863454
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.58895777 | 0.47158351 | 0.66129032
# MobileNetV2_Block14            | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.57328342 | 0.58928665 | 0.61832061
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.75400632 | 0.76128887 | 0.70329670
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.74207510 | 0.69674718 | 0.71962617
# MobileNetV2_Block14            | per frame (sum) | BalancedDistributionSVG/500/11/0.30      | None       | 0.71002456 | 0.70913974 | 0.66094421
# MobileNetV2_Block14            | per frame (sum) | SVG                                      | None       | 0.70429290 | 0.69435318 | 0.66666667
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.75201778 | 0.75900845 | 0.69230769
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.73423792 | 0.69600365 | 0.70476190
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.74687098 | 0.75765041 | 0.67281106
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.72698561 | 0.69241348 | 0.70370370
# MobileNetV2_Block14            | per frame (sum) | BalancedDistributionSVG/500/11/0.30      | (1, 1, 1)  | 0.70522868 | 0.70660315 | 0.67000000
# MobileNetV2_Block14            | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.70546263 | 0.69199337 | 0.66310160
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.74839162 | 0.75774486 | 0.67281106
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.72125395 | 0.67992717 | 0.71770335
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.71318283 | 0.73158306 | 0.64000000
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.69446719 | 0.66350221 | 0.67857143
# MobileNetV2_Block14            | per frame (sum) | BalancedDistributionSVG/500/11/0.30      | (2, 2, 2)  | 0.66265060 | 0.67972161 | 0.63967611
# MobileNetV2_Block14            | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.66335244 | 0.66479100 | 0.63492063
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.71622412 | 0.73126040 | 0.64341085
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.67972862 | 0.63915478 | 0.68122271
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.75400632 | 0.76128887 | 0.70329670
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.74207510 | 0.69674718 | 0.71962617
# MobileNetV2_Block14            | per frame (sum) | BalancedDistributionSVG/500/11/0.30      | (0, 1, 1)  | 0.71002456 | 0.70913974 | 0.66094421
# MobileNetV2_Block14            | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.70429290 | 0.69435318 | 0.66666667
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.75201778 | 0.75900845 | 0.69230769
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.73423792 | 0.69600365 | 0.70476190
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.75400632 | 0.76128887 | 0.70329670
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.74207510 | 0.69674718 | 0.71962617
# MobileNetV2_Block14            | per frame (sum) | BalancedDistributionSVG/500/11/0.30      | (0, 2, 2)  | 0.71002456 | 0.70913974 | 0.66094421
# MobileNetV2_Block14            | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.70429290 | 0.69435318 | 0.66666667
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.75201778 | 0.75900845 | 0.69230769
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.73423792 | 0.69600365 | 0.70476190
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.74687098 | 0.75765041 | 0.67281106
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.72698561 | 0.69241348 | 0.70370370
# MobileNetV2_Block14            | per frame (sum) | BalancedDistributionSVG/500/11/0.30      | (1, 0, 0)  | 0.70522868 | 0.70660315 | 0.67000000
# MobileNetV2_Block14            | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.70546263 | 0.69199337 | 0.66310160
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.74839162 | 0.75774486 | 0.67281106
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.72125395 | 0.67992717 | 0.71770335
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.71318283 | 0.73158306 | 0.64000000
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.69446719 | 0.66350221 | 0.67857143
# MobileNetV2_Block14            | per frame (sum) | BalancedDistributionSVG/500/11/0.30      | (2, 0, 0)  | 0.66265060 | 0.67972161 | 0.63967611
# MobileNetV2_Block14            | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.66335244 | 0.66479100 | 0.63492063
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.71622412 | 0.73126040 | 0.64341085
# MobileNetV2_Block14            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.67972862 | 0.63915478 | 0.68122271
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/0.20                      | None       | 0.85697810 | 0.14961513 | 0.25847225
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.87232022 | 0.20683737 | 0.27904578
# MobileNetV2_Block9             | per patch       | SVG                                      | None       | 0.83774735 | 0.12254966 | 0.25259345
# MobileNetV2_Block9             | per patch       | BalancedDistributionSVG/500/7/0.30       | None       | 0.83079459 | 0.12365944 | 0.23740498
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/0.50                      | None       | 0.85664899 | 0.14824207 | 0.25879703
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.87035829 | 0.21584001 | 0.28100890
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.85883098 | 0.14496283 | 0.26681128
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.88747236 | 0.21455877 | 0.29356789
# MobileNetV2_Block9             | per patch       | SVG                                      | (1, 1, 1)  | 0.83629545 | 0.12179316 | 0.24700279
# MobileNetV2_Block9             | per patch       | BalancedDistributionSVG/500/7/0.30       | (1, 1, 1)  | 0.83976473 | 0.12521233 | 0.24837733
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.85858669 | 0.14371588 | 0.26606601
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.88724648 | 0.23432138 | 0.29293791
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.86115939 | 0.15847087 | 0.27079935
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.88685921 | 0.23282081 | 0.31902206
# MobileNetV2_Block9             | per patch       | SVG                                      | (2, 2, 2)  | 0.83704415 | 0.13546029 | 0.23790596
# MobileNetV2_Block9             | per patch       | BalancedDistributionSVG/500/7/0.30       | (2, 2, 2)  | 0.84049181 | 0.14304751 | 0.24191029
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.86076919 | 0.15779847 | 0.27009572
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.88595374 | 0.25550337 | 0.30968834
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.87634567 | 0.17724083 | 0.28081700
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.90009208 | 0.28288481 | 0.32102431
# MobileNetV2_Block9             | per patch       | SVG                                      | (0, 1, 1)  | 0.84911259 | 0.12919995 | 0.26673799
# MobileNetV2_Block9             | per patch       | BalancedDistributionSVG/500/7/0.30       | (0, 1, 1)  | 0.84940118 | 0.13170332 | 0.26130128
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.87591385 | 0.17401302 | 0.28108756
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.89801620 | 0.30510778 | 0.33090770
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.89944021 | 0.22307833 | 0.33241866
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.91980050 | 0.37115357 | 0.39146919
# MobileNetV2_Block9             | per patch       | SVG                                      | (0, 2, 2)  | 0.86971536 | 0.15071867 | 0.28687840
# MobileNetV2_Block9             | per patch       | BalancedDistributionSVG/500/7/0.30       | (0, 2, 2)  | 0.87234893 | 0.15746267 | 0.28804410
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.89885164 | 0.22067966 | 0.33096927
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.91717335 | 0.40039773 | 0.41622964
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.84480693 | 0.12957357 | 0.25374150
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.86840623 | 0.16900647 | 0.27122070
# MobileNetV2_Block9             | per patch       | SVG                                      | (1, 0, 0)  | 0.82680120 | 0.11430417 | 0.24212660
# MobileNetV2_Block9             | per patch       | BalancedDistributionSVG/500/7/0.30       | (1, 0, 0)  | 0.82916135 | 0.11798679 | 0.24030320
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.84456572 | 0.12899354 | 0.25336567
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.86780755 | 0.17884428 | 0.27250446
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.82377074 | 0.11606041 | 0.23145588
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.84779141 | 0.14121013 | 0.24947652
# MobileNetV2_Block9             | per patch       | SVG                                      | (2, 0, 0)  | 0.80957297 | 0.10748823 | 0.21891934
# MobileNetV2_Block9             | per patch       | BalancedDistributionSVG/500/7/0.30       | (2, 0, 0)  | 0.81308879 | 0.11073398 | 0.22258091
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.82374644 | 0.11596051 | 0.23150817
# MobileNetV2_Block9             | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.84750212 | 0.14716933 | 0.25043445
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.55468476 | 0.52772295 | 0.62172285
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.58720318 | 0.53085126 | 0.64197531
# MobileNetV2_Block9             | per frame (max) | SVG                                      | None       | 0.56041642 | 0.46834614 | 0.62641509
# MobileNetV2_Block9             | per frame (max) | BalancedDistributionSVG/500/7/0.30       | None       | 0.56076734 | 0.47670806 | 0.63241107
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.56111826 | 0.52838046 | 0.62835249
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.57737747 | 0.51685412 | 0.65338645
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.61480875 | 0.52664059 | 0.65198238
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.57597380 | 0.50739209 | 0.63900415
# MobileNetV2_Block9             | per frame (max) | SVG                                      | (1, 1, 1)  | 0.57550591 | 0.46940990 | 0.67782427
# MobileNetV2_Block9             | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (1, 1, 1)  | 0.57258159 | 0.45907633 | 0.67782427
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.61036379 | 0.51600422 | 0.66379310
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.59024447 | 0.58582144 | 0.64705882
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.59281787 | 0.46147633 | 0.68398268
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.56380863 | 0.52784406 | 0.62406015
# MobileNetV2_Block9             | per frame (max) | SVG                                      | (2, 2, 2)  | 0.59036145 | 0.46212676 | 0.69298246
# MobileNetV2_Block9             | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (2, 2, 2)  | 0.60989589 | 0.47298728 | 0.69527897
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.59199906 | 0.46021089 | 0.68995633
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.59539127 | 0.55508916 | 0.63768116
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.57632472 | 0.47901191 | 0.64935065
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.60053807 | 0.54202998 | 0.62711864
# MobileNetV2_Block9             | per frame (max) | SVG                                      | (0, 1, 1)  | 0.56427652 | 0.45738853 | 0.66666667
# MobileNetV2_Block9             | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (0, 1, 1)  | 0.54731548 | 0.44601488 | 0.64035088
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.57538893 | 0.47636552 | 0.66371681
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.60439818 | 0.55902812 | 0.63745020
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.54228565 | 0.47911678 | 0.62641509
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.61515967 | 0.58112066 | 0.63157895
# MobileNetV2_Block9             | per frame (max) | SVG                                      | (0, 2, 2)  | 0.53023745 | 0.45159278 | 0.63967611
# MobileNetV2_Block9             | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (0, 2, 2)  | 0.54064803 | 0.45951681 | 0.63601533
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.54345537 | 0.48906377 | 0.63673469
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.63516201 | 0.63399295 | 0.62641509
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.61129957 | 0.52801550 | 0.65000000
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.57152883 | 0.48778612 | 0.64092664
# MobileNetV2_Block9             | per frame (max) | SVG                                      | (1, 0, 0)  | 0.60615277 | 0.50052432 | 0.64313725
# MobileNetV2_Block9             | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (1, 0, 0)  | 0.59843257 | 0.50434704 | 0.64822134
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.61188443 | 0.52602977 | 0.65289256
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.55819394 | 0.47424585 | 0.64680851
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.64065973 | 0.52914074 | 0.66666667
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.57106094 | 0.45774959 | 0.66666667
# MobileNetV2_Block9             | per frame (max) | SVG                                      | (2, 0, 0)  | 0.61328810 | 0.49532066 | 0.64843750
# MobileNetV2_Block9             | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (2, 0, 0)  | 0.58942566 | 0.49362638 | 0.64843750
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.63399228 | 0.52448591 | 0.67213115
# MobileNetV2_Block9             | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.57047608 | 0.46979613 | 0.66108787
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.68148321 | 0.64388252 | 0.66666667
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.68967131 | 0.66977826 | 0.64485981
# MobileNetV2_Block9             | per frame (sum) | SVG                                      | None       | 0.72488010 | 0.66172655 | 0.71153846
# MobileNetV2_Block9             | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | None       | 0.76242835 | 0.69785197 | 0.72906404
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.68756580 | 0.64899906 | 0.66666667
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.68288689 | 0.66725780 | 0.63783784
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.66569189 | 0.63307902 | 0.66063348
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.67294420 | 0.64876192 | 0.63636364
# MobileNetV2_Block9             | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.71318283 | 0.65109851 | 0.68722467
# MobileNetV2_Block9             | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (1, 1, 1)  | 0.77143526 | 0.69668232 | 0.75829384
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.67072172 | 0.63770576 | 0.66055046
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.66499006 | 0.64535453 | 0.63529412
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.63481109 | 0.61184042 | 0.64843750
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.64990057 | 0.62412974 | 0.63636364
# MobileNetV2_Block9             | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.67762311 | 0.62788492 | 0.68619247
# MobileNetV2_Block9             | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (2, 2, 2)  | 0.74757282 | 0.67564835 | 0.75799087
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.63937303 | 0.61566339 | 0.64655172
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.64019184 | 0.62130671 | 0.63241107
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.68148321 | 0.64388252 | 0.66666667
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.68967131 | 0.66977826 | 0.64485981
# MobileNetV2_Block9             | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.72488010 | 0.66172655 | 0.71153846
# MobileNetV2_Block9             | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (0, 1, 1)  | 0.76242835 | 0.69785197 | 0.72906404
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.68756580 | 0.64899906 | 0.66666667
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.68288689 | 0.66725780 | 0.63783784
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.68148321 | 0.64388252 | 0.66666667
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.68967131 | 0.66977826 | 0.64485981
# MobileNetV2_Block9             | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.72488010 | 0.66172655 | 0.71153846
# MobileNetV2_Block9             | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (0, 2, 2)  | 0.76242835 | 0.69785197 | 0.72906404
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.68756580 | 0.64899906 | 0.66666667
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.68288689 | 0.66725780 | 0.63783784
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.66569189 | 0.63307902 | 0.66063348
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.67294420 | 0.64876192 | 0.63636364
# MobileNetV2_Block9             | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.71318283 | 0.65109851 | 0.68722467
# MobileNetV2_Block9             | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (1, 0, 0)  | 0.77143526 | 0.69668232 | 0.75829384
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.67072172 | 0.63770576 | 0.66055046
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.66499006 | 0.64535453 | 0.63529412
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.63481109 | 0.61184042 | 0.64843750
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.64990057 | 0.62412974 | 0.63636364
# MobileNetV2_Block9             | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.67762311 | 0.62788492 | 0.68619247
# MobileNetV2_Block9             | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (2, 0, 0)  | 0.74757282 | 0.67564835 | 0.75799087
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.63937303 | 0.61566339 | 0.64655172
# MobileNetV2_Block9             | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.64019184 | 0.62130671 | 0.63241107
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/0.20                      | None       | 0.89769447 | 0.30812327 | 0.34568134
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.89848370 | 0.29495119 | 0.35749182
# ResNet50V2_Stack3_LargeImage   | per patch       | SVG                                      | None       | 0.88818480 | 0.20896704 | 0.31044401
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/0.50                      | None       | 0.89855960 | 0.30343250 | 0.34471891
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.89831590 | 0.32874305 | 0.36167005
# ResNet50V2_Stack3_LargeImage   | per patch       | BalancedDistributionSVG/500/21/0.30      | None       | 0.89698829 | 0.22785721 | 0.33636520
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.91296915 | 0.28998077 | 0.34333441
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.91798491 | 0.28167750 | 0.36622797
# ResNet50V2_Stack3_LargeImage   | per patch       | SVG                                      | (1, 1, 1)  | 0.88428183 | 0.18269499 | 0.30862122
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.91163618 | 0.27984976 | 0.33435937
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.91885161 | 0.32314607 | 0.37776022
# ResNet50V2_Stack3_LargeImage   | per patch       | BalancedDistributionSVG/500/21/0.30      | (1, 1, 1)  | 0.90196882 | 0.21635041 | 0.33340948
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.89462049 | 0.22766224 | 0.30742876
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.90148950 | 0.23511782 | 0.32283894
# ResNet50V2_Stack3_LargeImage   | per patch       | SVG                                      | (2, 2, 2)  | 0.86513621 | 0.16028355 | 0.26810500
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.89286126 | 0.21688187 | 0.30420363
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.90332190 | 0.25969309 | 0.33690631
# ResNet50V2_Stack3_LargeImage   | per patch       | BalancedDistributionSVG/500/21/0.30      | (2, 2, 2)  | 0.88003533 | 0.19814811 | 0.29073160
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.92991800 | 0.40907836 | 0.40438820
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.93196136 | 0.38249965 | 0.42593545
# ResNet50V2_Stack3_LargeImage   | per patch       | SVG                                      | (0, 1, 1)  | 0.90291545 | 0.20633745 | 0.33795968
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.92976155 | 0.39770629 | 0.40382434
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.93309340 | 0.44715865 | 0.43554798
# ResNet50V2_Stack3_LargeImage   | per patch       | BalancedDistributionSVG/500/21/0.30      | (0, 1, 1)  | 0.91892416 | 0.24624867 | 0.38147420
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.93810342 | 0.42117804 | 0.42954804
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.94123522 | 0.39989312 | 0.45841210
# ResNet50V2_Stack3_LargeImage   | per patch       | SVG                                      | (0, 2, 2)  | 0.90300443 | 0.20219108 | 0.34610195
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.93722018 | 0.40629088 | 0.42279208
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.94367636 | 0.47570733 | 0.47167178
# ResNet50V2_Stack3_LargeImage   | per patch       | BalancedDistributionSVG/500/21/0.30      | (0, 2, 2)  | 0.91839217 | 0.24824762 | 0.37853741
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.89527360 | 0.24124241 | 0.32065358
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.89895307 | 0.24464566 | 0.33773150
# ResNet50V2_Stack3_LargeImage   | per patch       | SVG                                      | (1, 0, 0)  | 0.87747422 | 0.17451805 | 0.30104892
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.89518903 | 0.23551700 | 0.31760089
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.89829686 | 0.26315577 | 0.34078116
# ResNet50V2_Stack3_LargeImage   | per patch       | BalancedDistributionSVG/500/21/0.30      | (1, 0, 0)  | 0.89325359 | 0.20210735 | 0.32292611
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.86970934 | 0.18998455 | 0.27343750
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.87580785 | 0.19456143 | 0.28973896
# ResNet50V2_Stack3_LargeImage   | per patch       | SVG                                      | (2, 0, 0)  | 0.85563397 | 0.15531369 | 0.26743333
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.86965298 | 0.18694592 | 0.27383192
# ResNet50V2_Stack3_LargeImage   | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.87458372 | 0.20264185 | 0.29565102
# ResNet50V2_Stack3_LargeImage   | per patch       | BalancedDistributionSVG/500/21/0.30      | (2, 0, 0)  | 0.87324194 | 0.17782152 | 0.28808378
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.64580653 | 0.64882766 | 0.64285714
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.64604047 | 0.54776502 | 0.64480874
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SVG                                      | None       | 0.57995087 | 0.50064713 | 0.65587045
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.63644871 | 0.64770222 | 0.63492063
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.65750380 | 0.60330680 | 0.64804469
# ResNet50V2_Stack3_LargeImage   | per frame (max) | BalancedDistributionSVG/500/21/0.30      | None       | 0.58720318 | 0.48195355 | 0.64069264
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.69060709 | 0.66827311 | 0.66666667
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.67364604 | 0.54974011 | 0.66666667
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SVG                                      | (1, 1, 1)  | 0.63598082 | 0.57862955 | 0.65587045
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.69212773 | 0.66665696 | 0.66094421
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.69458416 | 0.67303210 | 0.66964286
# ResNet50V2_Stack3_LargeImage   | per frame (max) | BalancedDistributionSVG/500/21/0.30      | (1, 1, 1)  | 0.60533396 | 0.59047047 | 0.65306122
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.64533864 | 0.63195997 | 0.65560166
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.60346239 | 0.51305980 | 0.66945607
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SVG                                      | (2, 2, 2)  | 0.66580887 | 0.55216028 | 0.68246445
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.62814364 | 0.61319124 | 0.65271967
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.66393730 | 0.65570136 | 0.68049793
# ResNet50V2_Stack3_LargeImage   | per frame (max) | BalancedDistributionSVG/500/21/0.30      | (2, 2, 2)  | 0.68499240 | 0.64787442 | 0.66115702
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.74113931 | 0.71975744 | 0.68085106
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.70300620 | 0.57720602 | 0.67592593
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SVG                                      | (0, 1, 1)  | 0.61176746 | 0.54774849 | 0.64541833
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.74464850 | 0.72444181 | 0.68341709
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.74207510 | 0.70529972 | 0.69444444
# ResNet50V2_Stack3_LargeImage   | per frame (max) | BalancedDistributionSVG/500/21/0.30      | (0, 1, 1)  | 0.58065271 | 0.50394244 | 0.65338645
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.69938004 | 0.69121518 | 0.63694268
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.68241900 | 0.58730511 | 0.64566929
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SVG                                      | (0, 2, 2)  | 0.64311615 | 0.56062870 | 0.64462810
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.68990525 | 0.68019655 | 0.63414634
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.73868289 | 0.71693819 | 0.67428571
# ResNet50V2_Stack3_LargeImage   | per frame (max) | BalancedDistributionSVG/500/21/0.30      | (0, 2, 2)  | 0.64896479 | 0.58568078 | 0.64135021
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.72499708 | 0.67819757 | 0.69868996
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.70324015 | 0.56611389 | 0.68376068
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SVG                                      | (1, 0, 0)  | 0.60626974 | 0.53724862 | 0.66386555
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.72371038 | 0.67712917 | 0.69264069
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.72172184 | 0.65677123 | 0.68067227
# ResNet50V2_Stack3_LargeImage   | per frame (max) | BalancedDistributionSVG/500/21/0.30      | (1, 0, 0)  | 0.59574219 | 0.53200589 | 0.65354331
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.70534565 | 0.65262493 | 0.69603524
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.67961165 | 0.54213044 | 0.68421053
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SVG                                      | (2, 0, 0)  | 0.64990057 | 0.63024555 | 0.64516129
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.70522868 | 0.64736929 | 0.69955157
# ResNet50V2_Stack3_LargeImage   | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.70721722 | 0.66065502 | 0.67532468
# ResNet50V2_Stack3_LargeImage   | per frame (max) | BalancedDistributionSVG/500/21/0.30      | (2, 0, 0)  | 0.62966429 | 0.65174128 | 0.64516129
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.70031583 | 0.65898818 | 0.69090909
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.72008422 | 0.65215404 | 0.70440252
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SVG                                      | None       | 0.72979296 | 0.65754422 | 0.69230769
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.70534565 | 0.66187176 | 0.69090909
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.73903381 | 0.71468655 | 0.71337580
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | BalancedDistributionSVG/500/21/0.30      | None       | 0.71879752 | 0.67469894 | 0.68539326
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.67972862 | 0.64176468 | 0.67403315
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.70557960 | 0.66105387 | 0.68108108
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.71131126 | 0.64340121 | 0.67924528
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.68651304 | 0.64535670 | 0.67455621
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.72031817 | 0.69731488 | 0.68292683
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | BalancedDistributionSVG/500/21/0.30      | (1, 1, 1)  | 0.70394198 | 0.65962334 | 0.68292683
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.66323547 | 0.63061499 | 0.65240642
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.69177682 | 0.66509945 | 0.65384615
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.68323781 | 0.63005627 | 0.67873303
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.66732951 | 0.63298845 | 0.65284974
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.70195344 | 0.68095156 | 0.66009852
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | BalancedDistributionSVG/500/21/0.30      | (2, 2, 2)  | 0.68148321 | 0.64192757 | 0.67298578
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.70031583 | 0.65898818 | 0.69090909
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.72008422 | 0.65215404 | 0.70440252
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.72979296 | 0.65754422 | 0.69230769
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.70534565 | 0.66187176 | 0.69090909
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.73903381 | 0.71468655 | 0.71337580
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | BalancedDistributionSVG/500/21/0.30      | (0, 1, 1)  | 0.71879752 | 0.67469894 | 0.68539326
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.70031583 | 0.65898818 | 0.69090909
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.72008422 | 0.65215404 | 0.70440252
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.72979296 | 0.65754422 | 0.69230769
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.70534565 | 0.66187176 | 0.69090909
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.73903381 | 0.71468655 | 0.71337580
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | BalancedDistributionSVG/500/21/0.30      | (0, 2, 2)  | 0.71879752 | 0.67469894 | 0.68539326
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.67972862 | 0.64176468 | 0.67403315
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.70557960 | 0.66105387 | 0.68108108
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.71131126 | 0.64340121 | 0.67924528
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.68651304 | 0.64535670 | 0.67455621
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.72031817 | 0.69731488 | 0.68292683
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | BalancedDistributionSVG/500/21/0.30      | (1, 0, 0)  | 0.70394198 | 0.65962334 | 0.68292683
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.66323547 | 0.63061499 | 0.65240642
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.69177682 | 0.66509945 | 0.65384615
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.68323781 | 0.63005627 | 0.67873303
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.66732951 | 0.63298845 | 0.65284974
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.70195344 | 0.68095156 | 0.66009852
# ResNet50V2_Stack3_LargeImage   | per frame (sum) | BalancedDistributionSVG/500/21/0.30      | (2, 0, 0)  | 0.68148321 | 0.64192757 | 0.67298578
# VGG16_Block3                   | per patch       | SpatialBin/SVG/0.20                      | None       | 0.92249120 | 0.41698501 | 0.42874540
# VGG16_Block3                   | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.92406934 | 0.46078327 | 0.45284290
# VGG16_Block3                   | per patch       | BalancedDistributionSVG/500/11/0.30      | None       | 0.90123466 | 0.19982670 | 0.34313864
# VGG16_Block3                   | per patch       | SVG                                      | None       | 0.90138256 | 0.19527282 | 0.33588450
# VGG16_Block3                   | per patch       | SpatialBin/SVG/0.50                      | None       | 0.92385959 | 0.41551827 | 0.42675745
# VGG16_Block3                   | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.92316148 | 0.47146145 | 0.45868428
# VGG16_Block3                   | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.92364648 | 0.35296130 | 0.39031725
# VGG16_Block3                   | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.92866638 | 0.40664259 | 0.42363656
# VGG16_Block3                   | per patch       | BalancedDistributionSVG/500/11/0.30      | (1, 1, 1)  | 0.89507868 | 0.18104305 | 0.33023871
# VGG16_Block3                   | per patch       | SVG                                      | (1, 1, 1)  | 0.88968729 | 0.17265601 | 0.31610900
# VGG16_Block3                   | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.92480705 | 0.35962281 | 0.38630411
# VGG16_Block3                   | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.92764001 | 0.39492798 | 0.42677322
# VGG16_Block3                   | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.90660934 | 0.29863526 | 0.33322808
# VGG16_Block3                   | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.91471438 | 0.33548296 | 0.37374367
# VGG16_Block3                   | per patch       | BalancedDistributionSVG/500/11/0.30      | (2, 2, 2)  | 0.87584621 | 0.16151034 | 0.29010597
# VGG16_Block3                   | per patch       | SVG                                      | (2, 2, 2)  | 0.87013632 | 0.15469837 | 0.27844683
# VGG16_Block3                   | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.90787189 | 0.30251878 | 0.32911159
# VGG16_Block3                   | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.91320592 | 0.32305199 | 0.36957970
# VGG16_Block3                   | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.92852266 | 0.42999377 | 0.44451995
# VGG16_Block3                   | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.93136441 | 0.48896700 | 0.47372030
# VGG16_Block3                   | per patch       | BalancedDistributionSVG/500/11/0.30      | (0, 1, 1)  | 0.90449650 | 0.19932275 | 0.35484912
# VGG16_Block3                   | per patch       | SVG                                      | (0, 1, 1)  | 0.90263920 | 0.19343110 | 0.34450692
# VGG16_Block3                   | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.93015552 | 0.43719865 | 0.44294017
# VGG16_Block3                   | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.93019580 | 0.48492028 | 0.47632451
# VGG16_Block3                   | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.93480181 | 0.44366291 | 0.45490509
# VGG16_Block3                   | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.93929172 | 0.51250944 | 0.49231839
# VGG16_Block3                   | per patch       | BalancedDistributionSVG/500/11/0.30      | (0, 2, 2)  | 0.90538064 | 0.19643413 | 0.35961904
# VGG16_Block3                   | per patch       | SVG                                      | (0, 2, 2)  | 0.90269246 | 0.19035368 | 0.35054146
# VGG16_Block3                   | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.93657243 | 0.45438420 | 0.45668118
# VGG16_Block3                   | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.93770995 | 0.49806354 | 0.49346525
# VGG16_Block3                   | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.92057602 | 0.34755647 | 0.38510336
# VGG16_Block3                   | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.92422009 | 0.39122006 | 0.41289056
# VGG16_Block3                   | per patch       | BalancedDistributionSVG/500/11/0.30      | (1, 0, 0)  | 0.89419274 | 0.18176730 | 0.32595900
# VGG16_Block3                   | per patch       | SVG                                      | (1, 0, 0)  | 0.88952904 | 0.17370912 | 0.31272615
# VGG16_Block3                   | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.92132803 | 0.34870124 | 0.38127219
# VGG16_Block3                   | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.92384902 | 0.39244211 | 0.41986369
# VGG16_Block3                   | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.90061024 | 0.27926510 | 0.32653681
# VGG16_Block3                   | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.90594637 | 0.30945338 | 0.35404605
# VGG16_Block3                   | per patch       | BalancedDistributionSVG/500/11/0.30      | (2, 0, 0)  | 0.87578702 | 0.16286723 | 0.29140005
# VGG16_Block3                   | per patch       | SVG                                      | (2, 0, 0)  | 0.87032506 | 0.15560455 | 0.27803229
# VGG16_Block3                   | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.90125939 | 0.28365644 | 0.32215985
# VGG16_Block3                   | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.90572865 | 0.30723878 | 0.35953663
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.62510235 | 0.54973072 | 0.63559322
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.60849222 | 0.58033744 | 0.64591440
# VGG16_Block3                   | per frame (max) | BalancedDistributionSVG/500/11/0.30      | None       | 0.61059773 | 0.51703700 | 0.63358779
# VGG16_Block3                   | per frame (max) | SVG                                      | None       | 0.59831559 | 0.51075394 | 0.63934426
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.64346707 | 0.60447208 | 0.63709677
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.58790502 | 0.51483081 | 0.64092664
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.64814598 | 0.55047199 | 0.64777328
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.62907942 | 0.59459571 | 0.64591440
# VGG16_Block3                   | per frame (max) | BalancedDistributionSVG/500/11/0.30      | (1, 1, 1)  | 0.64065973 | 0.50116953 | 0.70754717
# VGG16_Block3                   | per frame (max) | SVG                                      | (1, 1, 1)  | 0.63562990 | 0.49603371 | 0.70422535
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.65083636 | 0.59371668 | 0.65198238
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.61644637 | 0.51341757 | 0.65098039
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.60966195 | 0.56453131 | 0.65079365
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.61551059 | 0.58622915 | 0.66400000
# VGG16_Block3                   | per frame (max) | BalancedDistributionSVG/500/11/0.30      | (2, 2, 2)  | 0.65984326 | 0.53299506 | 0.72037915
# VGG16_Block3                   | per frame (max) | SVG                                      | (2, 2, 2)  | 0.65586618 | 0.53811090 | 0.71844660
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.61738215 | 0.59093528 | 0.65338645
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.60217569 | 0.52202472 | 0.66135458
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.64206340 | 0.56307571 | 0.65040650
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.63761843 | 0.62312510 | 0.65338645
# VGG16_Block3                   | per frame (max) | BalancedDistributionSVG/500/11/0.30      | (0, 1, 1)  | 0.61129957 | 0.51860196 | 0.64566929
# VGG16_Block3                   | per frame (max) | SVG                                      | (0, 1, 1)  | 0.60708855 | 0.51459555 | 0.64566929
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.65399462 | 0.62545959 | 0.64062500
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.63527898 | 0.55606712 | 0.65098039
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.66019417 | 0.60128443 | 0.65079365
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.65843958 | 0.64885223 | 0.64822134
# VGG16_Block3                   | per frame (max) | BalancedDistributionSVG/500/11/0.30      | (0, 2, 2)  | 0.60732249 | 0.50984436 | 0.65322581
# VGG16_Block3                   | per frame (max) | SVG                                      | (0, 2, 2)  | 0.59433852 | 0.50458531 | 0.64566929
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.65785472 | 0.62391658 | 0.64541833
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.65352673 | 0.59715966 | 0.65354331
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.63340742 | 0.52892507 | 0.66666667
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.57468710 | 0.55698244 | 0.63453815
# VGG16_Block3                   | per frame (max) | BalancedDistributionSVG/500/11/0.30      | (1, 0, 0)  | 0.62872851 | 0.48986712 | 0.71153846
# VGG16_Block3                   | per frame (max) | SVG                                      | (1, 0, 0)  | 0.62545327 | 0.48842161 | 0.70192308
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.64487075 | 0.58334299 | 0.64601770
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.58123757 | 0.47746977 | 0.63967611
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.57117792 | 0.49016969 | 0.64822134
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.57878114 | 0.55005459 | 0.64341085
# VGG16_Block3                   | per frame (max) | BalancedDistributionSVG/500/11/0.30      | (2, 0, 0)  | 0.65036846 | 0.50485001 | 0.71962617
# VGG16_Block3                   | per frame (max) | SVG                                      | (2, 0, 0)  | 0.64884782 | 0.49542834 | 0.71962617
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.60556790 | 0.56181819 | 0.65863454
# VGG16_Block3                   | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.52906773 | 0.44254603 | 0.65040650
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.71973330 | 0.66019116 | 0.68131868
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.74429758 | 0.74270674 | 0.69047619
# VGG16_Block3                   | per frame (sum) | BalancedDistributionSVG/500/11/0.30      | None       | 0.80652708 | 0.75784322 | 0.76470588
# VGG16_Block3                   | per frame (sum) | SVG                                      | None       | 0.79494678 | 0.73677065 | 0.75903614
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.73318517 | 0.69432948 | 0.69430052
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.73236636 | 0.72410841 | 0.68965517
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.69762545 | 0.66333098 | 0.65957447
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.72885718 | 0.72922176 | 0.67052023
# VGG16_Block3                   | per frame (sum) | BalancedDistributionSVG/500/11/0.30      | (1, 1, 1)  | 0.79623348 | 0.75729838 | 0.78481013
# VGG16_Block3                   | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.77693297 | 0.72146577 | 0.74853801
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.71107732 | 0.68403004 | 0.66990291
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.71423558 | 0.70815632 | 0.66666667
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.67540063 | 0.64997957 | 0.65811966
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.71306586 | 0.70209460 | 0.66666667
# VGG16_Block3                   | per frame (sum) | BalancedDistributionSVG/500/11/0.30      | (2, 2, 2)  | 0.77786876 | 0.74508963 | 0.75000000
# VGG16_Block3                   | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.75447421 | 0.70615811 | 0.73333333
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.68896947 | 0.67170399 | 0.65811966
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.69516903 | 0.68702386 | 0.66285714
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.71973330 | 0.66019116 | 0.68131868
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.74429758 | 0.74270674 | 0.69047619
# VGG16_Block3                   | per frame (sum) | BalancedDistributionSVG/500/11/0.30      | (0, 1, 1)  | 0.80652708 | 0.75784322 | 0.76470588
# VGG16_Block3                   | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.79494678 | 0.73677065 | 0.75903614
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.73318517 | 0.69432948 | 0.69430052
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.73236636 | 0.72410841 | 0.68965517
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.71973330 | 0.66019116 | 0.68131868
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.74429758 | 0.74270674 | 0.69047619
# VGG16_Block3                   | per frame (sum) | BalancedDistributionSVG/500/11/0.30      | (0, 2, 2)  | 0.80652708 | 0.75784322 | 0.76470588
# VGG16_Block3                   | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.79494678 | 0.73677065 | 0.75903614
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.73318517 | 0.69432948 | 0.69430052
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.73236636 | 0.72410841 | 0.68965517
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.69762545 | 0.66333098 | 0.65957447
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.72885718 | 0.72922176 | 0.67052023
# VGG16_Block3                   | per frame (sum) | BalancedDistributionSVG/500/11/0.30      | (1, 0, 0)  | 0.79623348 | 0.75729838 | 0.78481013
# VGG16_Block3                   | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.77693297 | 0.72146577 | 0.74853801
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.71107732 | 0.68403004 | 0.66990291
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.71423558 | 0.70815632 | 0.66666667
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.67540063 | 0.64997957 | 0.65811966
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.71306586 | 0.70209460 | 0.66666667
# VGG16_Block3                   | per frame (sum) | BalancedDistributionSVG/500/11/0.30      | (2, 0, 0)  | 0.77786876 | 0.74508963 | 0.75000000
# VGG16_Block3                   | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.75447421 | 0.70615811 | 0.73333333
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.68896947 | 0.67170399 | 0.65811966
# VGG16_Block3                   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.69516903 | 0.68702386 | 0.66285714
# C3D_Block3                     | per patch       | SpatialBin/SVG/0.20                      | None       | 0.88239994 | 0.29666452 | 0.44214302
# C3D_Block3                     | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.88920504 | 0.30216784 | 0.45570820
# C3D_Block3                     | per patch       | SVG                                      | None       | 0.84877370 | 0.19994591 | 0.29995231
# C3D_Block3                     | per patch       | SpatialBin/SVG/0.50                      | None       | 0.88215043 | 0.29454469 | 0.43575719
# C3D_Block3                     | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.89161026 | 0.30801050 | 0.46508641
# C3D_Block3                     | per patch       | BalancedDistributionSVG/500/12/0.30      | None       | 0.81056696 | 0.19883768 | 0.30369411
# C3D_Block3                     | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.91060648 | 0.31212007 | 0.46818923
# C3D_Block3                     | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.92011414 | 0.32295106 | 0.48623004
# C3D_Block3                     | per patch       | SVG                                      | (1, 1, 1)  | 0.86741510 | 0.19980661 | 0.30505878
# C3D_Block3                     | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.91014124 | 0.30993190 | 0.46302405
# C3D_Block3                     | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.92342841 | 0.33236974 | 0.49483827
# C3D_Block3                     | per patch       | BalancedDistributionSVG/500/12/0.30      | (1, 1, 1)  | 0.85526861 | 0.20623688 | 0.31096658
# C3D_Block3                     | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.91750468 | 0.29886667 | 0.45311620
# C3D_Block3                     | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.92850505 | 0.31531809 | 0.47563537
# C3D_Block3                     | per patch       | SVG                                      | (2, 2, 2)  | 0.87027630 | 0.20109674 | 0.29814250
# C3D_Block3                     | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.91743894 | 0.30045475 | 0.44990921
# C3D_Block3                     | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.93162368 | 0.32354679 | 0.48090176
# C3D_Block3                     | per patch       | BalancedDistributionSVG/500/12/0.30      | (2, 2, 2)  | 0.86887071 | 0.21053657 | 0.29858577
# C3D_Block3                     | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.89684451 | 0.31551646 | 0.47398014
# C3D_Block3                     | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.90494073 | 0.31934533 | 0.48650245
# C3D_Block3                     | per patch       | SVG                                      | (0, 1, 1)  | 0.85903368 | 0.20402977 | 0.30998459
# C3D_Block3                     | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.89664880 | 0.31117702 | 0.46892755
# C3D_Block3                     | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.90832701 | 0.32713355 | 0.49561848
# C3D_Block3                     | per patch       | BalancedDistributionSVG/500/12/0.30      | (0, 1, 1)  | 0.83555581 | 0.20797590 | 0.31402502
# C3D_Block3                     | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.91065136 | 0.33546333 | 0.49227324
# C3D_Block3                     | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.91948960 | 0.33831070 | 0.50877805
# C3D_Block3                     | per patch       | SVG                                      | (0, 2, 2)  | 0.86903956 | 0.21715725 | 0.31908832
# C3D_Block3                     | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.91042158 | 0.32949019 | 0.48921620
# C3D_Block3                     | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.92336761 | 0.34783737 | 0.51527149
# C3D_Block3                     | per patch       | BalancedDistributionSVG/500/12/0.30      | (0, 2, 2)  | 0.85744404 | 0.22322273 | 0.32370531
# C3D_Block3                     | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.90074175 | 0.29774085 | 0.44308659
# C3D_Block3                     | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.90944279 | 0.30939015 | 0.46287520
# C3D_Block3                     | per patch       | SVG                                      | (1, 0, 0)  | 0.86020650 | 0.19562540 | 0.29672638
# C3D_Block3                     | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.90016311 | 0.29639763 | 0.43735763
# C3D_Block3                     | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.91199573 | 0.31669540 | 0.47130834
# C3D_Block3                     | per patch       | BalancedDistributionSVG/500/12/0.30      | (1, 0, 0)  | 0.83544895 | 0.19832365 | 0.30136631
# C3D_Block3                     | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.89943659 | 0.27515554 | 0.41446725
# C3D_Block3                     | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.90969179 | 0.28969046 | 0.43736423
# C3D_Block3                     | per patch       | SVG                                      | (2, 0, 0)  | 0.85712900 | 0.18678650 | 0.28992154
# C3D_Block3                     | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.89889607 | 0.27561745 | 0.40934940
# C3D_Block3                     | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.91220715 | 0.29606280 | 0.44554617
# C3D_Block3                     | per patch       | BalancedDistributionSVG/500/12/0.30      | (2, 0, 0)  | 0.83831088 | 0.19141690 | 0.29259315
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.81448123 | 0.68534387 | 0.76836158
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.74780676 | 0.61293876 | 0.72316384
# C3D_Block3                     | per frame (max) | SVG                                      | None       | 0.64299918 | 0.59916841 | 0.65777778
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.81857527 | 0.68439208 | 0.75151515
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.77073342 | 0.63930439 | 0.73631841
# C3D_Block3                     | per frame (max) | BalancedDistributionSVG/500/12/0.30      | None       | 0.64721020 | 0.59963434 | 0.65454545
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.77681600 | 0.68005638 | 0.74576271
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.79611650 | 0.68760797 | 0.76543210
# C3D_Block3                     | per frame (max) | SVG                                      | (1, 1, 1)  | 0.59960229 | 0.55504104 | 0.63559322
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.76628845 | 0.66795472 | 0.74157303
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.80091239 | 0.69410831 | 0.74534161
# C3D_Block3                     | per frame (max) | BalancedDistributionSVG/500/12/0.30      | (1, 1, 1)  | 0.60872617 | 0.55162541 | 0.64406780
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.76336414 | 0.67940682 | 0.75000000
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.79506375 | 0.72165115 | 0.75824176
# C3D_Block3                     | per frame (max) | SVG                                      | (2, 2, 2)  | 0.59059539 | 0.56873287 | 0.65040650
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.75810036 | 0.67180967 | 0.74251497
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.79763715 | 0.69925722 | 0.75000000
# C3D_Block3                     | per frame (max) | BalancedDistributionSVG/500/12/0.30      | (2, 2, 2)  | 0.60896011 | 0.57881336 | 0.65843621
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.78722658 | 0.68922329 | 0.74534161
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.78582290 | 0.67471255 | 0.72955975
# C3D_Block3                     | per frame (max) | SVG                                      | (0, 1, 1)  | 0.58965961 | 0.54004861 | 0.63436123
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.77658206 | 0.67384101 | 0.73417722
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.80079541 | 0.69426727 | 0.74117647
# C3D_Block3                     | per frame (max) | BalancedDistributionSVG/500/12/0.30      | (0, 1, 1)  | 0.59913440 | 0.53678716 | 0.64573991
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.79190549 | 0.72496662 | 0.78313253
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.80325184 | 0.71051599 | 0.77419355
# C3D_Block3                     | per frame (max) | SVG                                      | (0, 2, 2)  | 0.61656334 | 0.59280998 | 0.63013699
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.78055913 | 0.69831991 | 0.76129032
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.80231606 | 0.71613123 | 0.77011494
# C3D_Block3                     | per frame (max) | BalancedDistributionSVG/500/12/0.30      | (0, 2, 2)  | 0.62603813 | 0.58789639 | 0.63677130
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.78418528 | 0.66818098 | 0.74444444
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.74546731 | 0.62596388 | 0.70588235
# C3D_Block3                     | per frame (max) | SVG                                      | (1, 0, 0)  | 0.61831793 | 0.56388504 | 0.65686275
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.78909814 | 0.66139099 | 0.74285714
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.76827699 | 0.65612591 | 0.72392638
# C3D_Block3                     | per frame (max) | BalancedDistributionSVG/500/12/0.30      | (1, 0, 0)  | 0.63130191 | 0.56675315 | 0.65365854
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.76699029 | 0.64620960 | 0.74074074
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.74991227 | 0.63262243 | 0.71856287
# C3D_Block3                     | per frame (max) | SVG                                      | (2, 0, 0)  | 0.59983624 | 0.56491592 | 0.64628821
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.77108434 | 0.63989994 | 0.72727273
# C3D_Block3                     | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.77096736 | 0.65849620 | 0.75000000
# C3D_Block3                     | per frame (max) | BalancedDistributionSVG/500/12/0.30      | (2, 0, 0)  | 0.61340508 | 0.57295052 | 0.65338645
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.74032051 | 0.70825948 | 0.67357513
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.77775178 | 0.74173844 | 0.70270270
# C3D_Block3                     | per frame (sum) | SVG                                      | None       | 0.65212306 | 0.61554861 | 0.66086957
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.73634343 | 0.70221607 | 0.67796610
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.78839630 | 0.74299034 | 0.71028037
# C3D_Block3                     | per frame (sum) | BalancedDistributionSVG/500/12/0.30      | None       | 0.67130659 | 0.63727147 | 0.66968326
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.74359574 | 0.71567895 | 0.68246445
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.77681600 | 0.74233659 | 0.71428571
# C3D_Block3                     | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.65165516 | 0.60812913 | 0.66666667
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.74067142 | 0.71047357 | 0.68000000
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.79096970 | 0.74608153 | 0.73148148
# C3D_Block3                     | per frame (sum) | BalancedDistributionSVG/500/12/0.30      | (1, 1, 1)  | 0.66709557 | 0.62630124 | 0.67256637
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.73809802 | 0.71537466 | 0.70370370
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.76991461 | 0.73905264 | 0.72645740
# C3D_Block3                     | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.64451983 | 0.59487234 | 0.65811966
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.73950170 | 0.71285740 | 0.70697674
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.78020821 | 0.74098448 | 0.73148148
# C3D_Block3                     | per frame (sum) | BalancedDistributionSVG/500/12/0.30      | (2, 2, 2)  | 0.66171482 | 0.61777265 | 0.66666667
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.74032051 | 0.70825948 | 0.67357513
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.77775178 | 0.74173844 | 0.70270270
# C3D_Block3                     | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.65212306 | 0.61554861 | 0.66086957
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.73634343 | 0.70221607 | 0.67796610
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.78839630 | 0.74299034 | 0.71028037
# C3D_Block3                     | per frame (sum) | BalancedDistributionSVG/500/12/0.30      | (0, 1, 1)  | 0.67130659 | 0.63727147 | 0.66968326
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.74032051 | 0.70825948 | 0.67357513
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.77775178 | 0.74173844 | 0.70270270
# C3D_Block3                     | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.65212306 | 0.61554861 | 0.66086957
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.73634343 | 0.70221607 | 0.67796610
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.78839630 | 0.74299034 | 0.71028037
# C3D_Block3                     | per frame (sum) | BalancedDistributionSVG/500/12/0.30      | (0, 2, 2)  | 0.67130659 | 0.63727147 | 0.66968326
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.74359574 | 0.71567895 | 0.68246445
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.77681600 | 0.74233659 | 0.71428571
# C3D_Block3                     | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.65165516 | 0.60812913 | 0.66666667
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.74067142 | 0.71047357 | 0.68000000
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.79096970 | 0.74608153 | 0.73148148
# C3D_Block3                     | per frame (sum) | BalancedDistributionSVG/500/12/0.30      | (1, 0, 0)  | 0.66709557 | 0.62630124 | 0.67256637
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.73809802 | 0.71537466 | 0.70370370
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.76991461 | 0.73905264 | 0.72645740
# C3D_Block3                     | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.64451983 | 0.59487234 | 0.65811966
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.73950170 | 0.71285740 | 0.70697674
# C3D_Block3                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.78020821 | 0.74098448 | 0.73148148
# C3D_Block3                     | per frame (sum) | BalancedDistributionSVG/500/12/0.30      | (2, 0, 0)  | 0.66171482 | 0.61777265 | 0.66666667
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/0.20                      | None       | 0.57580411 | 0.05086573 | 0.11617186
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.57777552 | 0.05136139 | 0.11434642
# EfficientNetB0_Block4          | per patch       | SVG                                      | None       | 0.50880025 | 0.04617288 | 0.10016669
# EfficientNetB0_Block4          | per patch       | BalancedDistributionSVG/500/7/0.30       | None       | 0.45172746 | 0.04226893 | 0.09025256
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/0.50                      | None       | 0.57391899 | 0.05068354 | 0.11575325
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.58066643 | 0.05188828 | 0.11471553
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.55191050 | 0.04877186 | 0.10817811
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.55200811 | 0.04866628 | 0.10728796
# EfficientNetB0_Block4          | per patch       | SVG                                      | (1, 1, 1)  | 0.52574716 | 0.04688322 | 0.09867696
# EfficientNetB0_Block4          | per patch       | BalancedDistributionSVG/500/7/0.30       | (1, 1, 1)  | 0.45996092 | 0.04094808 | 0.09005894
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.55146383 | 0.04873318 | 0.10808477
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.55341870 | 0.04846389 | 0.10748766
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.52442049 | 0.04623109 | 0.09843254
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.52772214 | 0.04650325 | 0.09966058
# EfficientNetB0_Block4          | per patch       | SVG                                      | (2, 2, 2)  | 0.50425560 | 0.04398341 | 0.09743399
# EfficientNetB0_Block4          | per patch       | BalancedDistributionSVG/500/7/0.30       | (2, 2, 2)  | 0.45034223 | 0.03906061 | 0.09224541
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.52312779 | 0.04604807 | 0.09838430
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.52292256 | 0.04536082 | 0.10008943
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.55273809 | 0.04889610 | 0.10836744
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.55806895 | 0.04980586 | 0.10773481
# EfficientNetB0_Block4          | per patch       | SVG                                      | (0, 1, 1)  | 0.52469568 | 0.04694754 | 0.09816332
# EfficientNetB0_Block4          | per patch       | BalancedDistributionSVG/500/7/0.30       | (0, 1, 1)  | 0.45607568 | 0.04074720 | 0.09005894
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.55218129 | 0.04884417 | 0.10794543
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.55966432 | 0.04951523 | 0.10818348
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.52793271 | 0.04666148 | 0.09852253
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.54248243 | 0.04923056 | 0.10141211
# EfficientNetB0_Block4          | per patch       | SVG                                      | (0, 2, 2)  | 0.50528992 | 0.04415974 | 0.09656247
# EfficientNetB0_Block4          | per patch       | BalancedDistributionSVG/500/7/0.30       | (0, 2, 2)  | 0.44532501 | 0.03876343 | 0.09119001
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.52649396 | 0.04645645 | 0.09847926
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.53972998 | 0.04764409 | 0.10179533
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.57700109 | 0.05087474 | 0.11633789
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.57437052 | 0.05076206 | 0.11373237
# EfficientNetB0_Block4          | per patch       | SVG                                      | (1, 0, 0)  | 0.51323661 | 0.04623486 | 0.09970567
# EfficientNetB0_Block4          | per patch       | BalancedDistributionSVG/500/7/0.30       | (1, 0, 0)  | 0.45597642 | 0.04246351 | 0.09145113
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.57555142 | 0.05073780 | 0.11586824
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.57798907 | 0.05132639 | 0.11422719
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.57486413 | 0.05066486 | 0.11524518
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.57014421 | 0.05034286 | 0.11276441
# EfficientNetB0_Block4          | per patch       | SVG                                      | (2, 0, 0)  | 0.51610470 | 0.04626871 | 0.09954488
# EfficientNetB0_Block4          | per patch       | BalancedDistributionSVG/500/7/0.30       | (2, 0, 0)  | 0.46140192 | 0.04273880 | 0.09085392
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.57357077 | 0.05053873 | 0.11513534
# EfficientNetB0_Block4          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.57289849 | 0.05072750 | 0.11328110
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.61048076 | 0.58435655 | 0.63601533
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.46157445 | 0.39788945 | 0.64257028
# EfficientNetB0_Block4          | per frame (max) | SVG                                      | None       | 0.68183413 | 0.58733447 | 0.66666667
# EfficientNetB0_Block4          | per frame (max) | BalancedDistributionSVG/500/7/0.30       | None       | 0.67341209 | 0.56852190 | 0.66985646
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.61527664 | 0.55226598 | 0.64341085
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.46192537 | 0.39490902 | 0.63203463
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.61223535 | 0.62202559 | 0.64591440
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.46403088 | 0.39419571 | 0.64062500
# EfficientNetB0_Block4          | per frame (max) | SVG                                      | (1, 1, 1)  | 0.68569423 | 0.61432543 | 0.67961165
# EfficientNetB0_Block4          | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (1, 1, 1)  | 0.67434788 | 0.57250574 | 0.67661692
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.61597848 | 0.57540402 | 0.64489796
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.37361095 | 0.35092587 | 0.62601626
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.59925137 | 0.61939611 | 0.63601533
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.42426015 | 0.37129350 | 0.63565891
# EfficientNetB0_Block4          | per frame (max) | SVG                                      | (2, 2, 2)  | 0.61504270 | 0.54801344 | 0.63392857
# EfficientNetB0_Block4          | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (2, 2, 2)  | 0.60428120 | 0.50704300 | 0.64516129
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.59890046 | 0.61873248 | 0.63358779
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.34904667 | 0.34615968 | 0.62878788
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.60896011 | 0.58217109 | 0.64777328
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.47163411 | 0.40221355 | 0.64541833
# EfficientNetB0_Block4          | per frame (max) | SVG                                      | (0, 1, 1)  | 0.69891215 | 0.62592154 | 0.67555556
# EfficientNetB0_Block4          | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (0, 1, 1)  | 0.69189379 | 0.59385595 | 0.67555556
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.61445783 | 0.55543853 | 0.64566929
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.41279682 | 0.36616396 | 0.62948207
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.59901743 | 0.60573036 | 0.63967611
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.47292081 | 0.40484093 | 0.64257028
# EfficientNetB0_Block4          | per frame (max) | SVG                                      | (0, 2, 2)  | 0.66265060 | 0.59005900 | 0.66382979
# EfficientNetB0_Block4          | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (0, 2, 2)  | 0.65422856 | 0.54051538 | 0.66379310
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.59995321 | 0.55634737 | 0.63601533
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.40320505 | 0.36087114 | 0.62551440
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.61071470 | 0.62043485 | 0.64031621
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.46356299 | 0.39842566 | 0.65040650
# EfficientNetB0_Block4          | per frame (max) | SVG                                      | (1, 0, 0)  | 0.67235934 | 0.56475923 | 0.67692308
# EfficientNetB0_Block4          | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (1, 0, 0)  | 0.66627676 | 0.55704983 | 0.67289720
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.61796701 | 0.56381524 | 0.65020576
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.41642297 | 0.37069352 | 0.61940299
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.60662066 | 0.61922616 | 0.63601533
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.45923500 | 0.39240362 | 0.64489796
# EfficientNetB0_Block4          | per frame (max) | SVG                                      | (2, 0, 0)  | 0.63071704 | 0.54542370 | 0.66346154
# EfficientNetB0_Block4          | per frame (max) | BalancedDistributionSVG/500/7/0.30       | (2, 0, 0)  | 0.62662300 | 0.53517586 | 0.66346154
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.61469178 | 0.61514262 | 0.63934426
# EfficientNetB0_Block4          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.38378758 | 0.36004130 | 0.62040816
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.66522400 | 0.66575880 | 0.65863454
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.58158849 | 0.49642199 | 0.62406015
# EfficientNetB0_Block4          | per frame (sum) | SVG                                      | None       | 0.57854720 | 0.51714528 | 0.63492063
# EfficientNetB0_Block4          | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | None       | 0.47104925 | 0.40479966 | 0.61710037
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.66615978 | 0.66715998 | 0.65338645
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.56111826 | 0.45321405 | 0.63111111
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.68288689 | 0.69708672 | 0.65587045
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.55983156 | 0.46925052 | 0.61710037
# EfficientNetB0_Block4          | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.59059539 | 0.50852009 | 0.64092664
# EfficientNetB0_Block4          | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (1, 1, 1)  | 0.49327407 | 0.41161011 | 0.61886792
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.69072406 | 0.70175103 | 0.65573770
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.52976956 | 0.43328967 | 0.63291139
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.67797403 | 0.69611479 | 0.65853659
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.53047140 | 0.44052868 | 0.61710037
# EfficientNetB0_Block4          | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.57152883 | 0.49627159 | 0.63529412
# EfficientNetB0_Block4          | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (2, 2, 2)  | 0.48929699 | 0.40768123 | 0.62790698
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.68569423 | 0.70226461 | 0.66122449
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.49315709 | 0.41622623 | 0.62809917
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.66522400 | 0.66575880 | 0.65863454
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.58158849 | 0.49642199 | 0.62406015
# EfficientNetB0_Block4          | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.57854720 | 0.51714528 | 0.63492063
# EfficientNetB0_Block4          | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (0, 1, 1)  | 0.47104925 | 0.40479966 | 0.61710037
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.66615978 | 0.66715998 | 0.65338645
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.56111826 | 0.45321405 | 0.63111111
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.66522400 | 0.66575880 | 0.65863454
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.58158849 | 0.49642199 | 0.62406015
# EfficientNetB0_Block4          | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.57854720 | 0.51714528 | 0.63492063
# EfficientNetB0_Block4          | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (0, 2, 2)  | 0.47104925 | 0.40479966 | 0.61710037
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.66615978 | 0.66715998 | 0.65338645
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.56111826 | 0.45321405 | 0.63111111
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.68288689 | 0.69708672 | 0.65587045
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.55983156 | 0.46925052 | 0.61710037
# EfficientNetB0_Block4          | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.59059539 | 0.50852009 | 0.64092664
# EfficientNetB0_Block4          | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (1, 0, 0)  | 0.49327407 | 0.41161011 | 0.61886792
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.69072406 | 0.70175103 | 0.65573770
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.52976956 | 0.43328967 | 0.63291139
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.67797403 | 0.69611479 | 0.65853659
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.53047140 | 0.44052868 | 0.61710037
# EfficientNetB0_Block4          | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.57152883 | 0.49627159 | 0.63529412
# EfficientNetB0_Block4          | per frame (sum) | BalancedDistributionSVG/500/7/0.30       | (2, 0, 0)  | 0.48929699 | 0.40768123 | 0.62790698
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.68569423 | 0.70226461 | 0.66122449
# EfficientNetB0_Block4          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.49315709 | 0.41622623 | 0.62809917
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/0.20                      | None       | 0.58432349 | 0.05867607 | 0.10980998
# EfficientNetB3_Block3          | per patch       | BalancedDistributionSVG/500/4/0.30       | None       | 0.63947084 | 0.07927490 | 0.15229119
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.55194190 | 0.05456356 | 0.10016115
# EfficientNetB3_Block3          | per patch       | SVG                                      | None       | 0.63177311 | 0.07639721 | 0.14216370
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/0.50                      | None       | 0.58619836 | 0.05909751 | 0.11046181
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.54371824 | 0.05320970 | 0.09783442
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.57902610 | 0.05860165 | 0.10772394
# EfficientNetB3_Block3          | per patch       | BalancedDistributionSVG/500/4/0.30       | (1, 1, 1)  | 0.65225220 | 0.07735709 | 0.13864100
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.55834426 | 0.05833694 | 0.09956411
# EfficientNetB3_Block3          | per patch       | SVG                                      | (1, 1, 1)  | 0.61901493 | 0.07158102 | 0.13036127
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.58048091 | 0.05890548 | 0.10823453
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.54668311 | 0.05472847 | 0.09765531
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.57306899 | 0.05834158 | 0.10528102
# EfficientNetB3_Block3          | per patch       | BalancedDistributionSVG/500/4/0.30       | (2, 2, 2)  | 0.61670523 | 0.07358623 | 0.13695262
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.55330131 | 0.05672764 | 0.09838351
# EfficientNetB3_Block3          | per patch       | SVG                                      | (2, 2, 2)  | 0.58805500 | 0.06710487 | 0.12692108
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.57421466 | 0.05873696 | 0.10574400
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.53671372 | 0.05146358 | 0.09659400
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.57986535 | 0.05871482 | 0.10786884
# EfficientNetB3_Block3          | per patch       | BalancedDistributionSVG/500/4/0.30       | (0, 1, 1)  | 0.65555391 | 0.07824315 | 0.14129759
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.55187955 | 0.05442841 | 0.09882084
# EfficientNetB3_Block3          | per patch       | SVG                                      | (0, 1, 1)  | 0.62182256 | 0.07219447 | 0.13143728
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.58151203 | 0.05903675 | 0.10836631
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.54371224 | 0.05309174 | 0.09710646
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.57419206 | 0.05837626 | 0.10540503
# EfficientNetB3_Block3          | per patch       | BalancedDistributionSVG/500/4/0.30       | (0, 2, 2)  | 0.62333569 | 0.07502020 | 0.13813164
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.55198076 | 0.05421088 | 0.09869395
# EfficientNetB3_Block3          | per patch       | SVG                                      | (0, 2, 2)  | 0.59245871 | 0.06803320 | 0.12794300
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.57562117 | 0.05880749 | 0.10581969
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.54510415 | 0.05318412 | 0.09753142
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.58359293 | 0.05852544 | 0.10975305
# EfficientNetB3_Block3          | per patch       | BalancedDistributionSVG/500/4/0.30       | (1, 0, 0)  | 0.64866689 | 0.07932653 | 0.15309559
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.55772992 | 0.05824338 | 0.10085303
# EfficientNetB3_Block3          | per patch       | SVG                                      | (1, 0, 0)  | 0.64103152 | 0.07644462 | 0.14360119
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.58539382 | 0.05894359 | 0.11067931
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.54878592 | 0.05556737 | 0.09910462
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.58238730 | 0.05846491 | 0.10968575
# EfficientNetB3_Block3          | per patch       | BalancedDistributionSVG/500/4/0.30       | (2, 0, 0)  | 0.64660554 | 0.07807019 | 0.14823354
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.55327706 | 0.05663142 | 0.09959683
# EfficientNetB3_Block3          | per patch       | SVG                                      | (2, 0, 0)  | 0.63783531 | 0.07525976 | 0.13998238
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.58390211 | 0.05885419 | 0.11056462
# EfficientNetB3_Block3          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.54076685 | 0.05306613 | 0.09769815
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.49853784 | 0.42450531 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | BalancedDistributionSVG/500/4/0.30       | None       | 0.47537724 | 0.42251770 | 0.62878788
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.51351035 | 0.43761148 | 0.61886792
# EfficientNetB3_Block3          | per frame (max) | SVG                                      | None       | 0.47830156 | 0.42529852 | 0.62878788
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.47046438 | 0.41765837 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.50157913 | 0.44966020 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.51444613 | 0.43777242 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | BalancedDistributionSVG/500/4/0.30       | (1, 1, 1)  | 0.45794830 | 0.39670115 | 0.62878788
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.56111826 | 0.48719305 | 0.62100457
# EfficientNetB3_Block3          | per frame (max) | SVG                                      | (1, 1, 1)  | 0.45829922 | 0.39692426 | 0.62878788
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.48695754 | 0.43338891 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.54836823 | 0.48019543 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.54731548 | 0.48957581 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | BalancedDistributionSVG/500/4/0.30       | (2, 2, 2)  | 0.44964323 | 0.39964845 | 0.62641509
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.57164581 | 0.48285943 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | SVG                                      | (2, 2, 2)  | 0.44730378 | 0.39792433 | 0.62641509
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.54778337 | 0.50016303 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.54170078 | 0.47638713 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.51128787 | 0.43473504 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | BalancedDistributionSVG/500/4/0.30       | (0, 1, 1)  | 0.47046438 | 0.40684861 | 0.63117871
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.51210668 | 0.43627925 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | SVG                                      | (0, 1, 1)  | 0.47128319 | 0.40777045 | 0.63117871
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.49070067 | 0.43009264 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.51935899 | 0.45931961 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.53526728 | 0.45956030 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | BalancedDistributionSVG/500/4/0.30       | (0, 2, 2)  | 0.47023044 | 0.40790623 | 0.63117871
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.52836589 | 0.43986127 | 0.61940299
# EfficientNetB3_Block3          | per frame (max) | SVG                                      | (0, 2, 2)  | 0.46835887 | 0.40546718 | 0.63117871
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.53339572 | 0.47336904 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.54134987 | 0.47653715 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.49935665 | 0.42568394 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | BalancedDistributionSVG/500/4/0.30       | (1, 0, 0)  | 0.46215932 | 0.40026507 | 0.62878788
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.56766873 | 0.50177410 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | SVG                                      | (1, 0, 0)  | 0.46379694 | 0.40117447 | 0.62878788
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.46859282 | 0.41655578 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.53795766 | 0.47274161 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.48695754 | 0.41859180 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | BalancedDistributionSVG/500/4/0.30       | (2, 0, 0)  | 0.45490701 | 0.39936162 | 0.62641509
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.57047608 | 0.52325233 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | SVG                                      | (2, 0, 0)  | 0.45748041 | 0.40202175 | 0.62641509
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.46625336 | 0.41946503 | 0.61710037
# EfficientNetB3_Block3          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.52707919 | 0.46080503 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.49877179 | 0.42242187 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | BalancedDistributionSVG/500/4/0.30       | None       | 0.66908410 | 0.64024895 | 0.67811159
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.45081296 | 0.39705915 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SVG                                      | None       | 0.65726986 | 0.63226992 | 0.67532468
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.49222131 | 0.41861397 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.47748275 | 0.42842975 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.49444379 | 0.42261012 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | BalancedDistributionSVG/500/4/0.30       | (1, 1, 1)  | 0.67446485 | 0.63111089 | 0.69527897
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.49397590 | 0.43680436 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.66545795 | 0.62609392 | 0.69264069
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.48789332 | 0.41846106 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.48438414 | 0.43945720 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.48040707 | 0.41888650 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | BalancedDistributionSVG/500/4/0.30       | (2, 2, 2)  | 0.66112996 | 0.61996276 | 0.68333333
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.50801263 | 0.44771134 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.65305884 | 0.61543913 | 0.67510549
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.47444146 | 0.41564467 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.46344602 | 0.42724546 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.49877179 | 0.42242187 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | BalancedDistributionSVG/500/4/0.30       | (0, 1, 1)  | 0.66908410 | 0.64024895 | 0.67811159
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.45081296 | 0.39705915 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.65726986 | 0.63226992 | 0.67532468
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.49222131 | 0.41861397 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.47748275 | 0.42842975 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.49877179 | 0.42242187 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | BalancedDistributionSVG/500/4/0.30       | (0, 2, 2)  | 0.66908410 | 0.64024895 | 0.67811159
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.45081296 | 0.39705915 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.65726986 | 0.63226992 | 0.67532468
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.49222131 | 0.41861397 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.47748275 | 0.42842975 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.49444379 | 0.42261012 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | BalancedDistributionSVG/500/4/0.30       | (1, 0, 0)  | 0.67446485 | 0.63111089 | 0.69527897
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.49397590 | 0.43680436 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.66545795 | 0.62609392 | 0.69264069
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.48789332 | 0.41846106 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.48438414 | 0.43945720 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.48040707 | 0.41888650 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | BalancedDistributionSVG/500/4/0.30       | (2, 0, 0)  | 0.66112996 | 0.61996276 | 0.68333333
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.50801263 | 0.44771134 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.65305884 | 0.61543913 | 0.67510549
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.47444146 | 0.41564467 | 0.61710037
# EfficientNetB3_Block3          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.46344602 | 0.42724546 | 0.61710037
# EfficientNetB6                 | per patch       | SpatialBin/SVG/0.20                      | None       | 0.57404304 | 0.06869695 | 0.14573460
# EfficientNetB6                 | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.58927698 | 0.06949493 | 0.15168967
# EfficientNetB6                 | per patch       | SVG                                      | None       | 0.58654121 | 0.08318312 | 0.18517548
# EfficientNetB6                 | per patch       | SpatialBin/SVG/0.50                      | None       | 0.57355059 | 0.06816024 | 0.14445400
# EfficientNetB6                 | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.59207383 | 0.07040142 | 0.15553501
# EfficientNetB6                 | per patch       | BalancedDistributionSVG/500/45/0.30      | None       | 0.57861811 | 0.07601431 | 0.14230322
# EfficientNetB6                 | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.59860896 | 0.07473189 | 0.12621150
# EfficientNetB6                 | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.62450453 | 0.07516197 | 0.13855244
# EfficientNetB6                 | per patch       | SVG                                      | (1, 1, 1)  | 0.61609007 | 0.10360977 | 0.18188264
# EfficientNetB6                 | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.59864885 | 0.07352661 | 0.12536935
# EfficientNetB6                 | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.62610807 | 0.07452665 | 0.13687839
# EfficientNetB6                 | per patch       | BalancedDistributionSVG/500/45/0.30      | (1, 1, 1)  | 0.60031480 | 0.09478083 | 0.15715309
# EfficientNetB6                 | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.60936357 | 0.09462451 | 0.14547964
# EfficientNetB6                 | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.62857514 | 0.07589059 | 0.14652015
# EfficientNetB6                 | per patch       | SVG                                      | (2, 2, 2)  | 0.62461238 | 0.12671792 | 0.21583851
# EfficientNetB6                 | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.60988732 | 0.09306479 | 0.14476190
# EfficientNetB6                 | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.63076947 | 0.08023467 | 0.15167394
# EfficientNetB6                 | per patch       | BalancedDistributionSVG/500/45/0.30      | (2, 2, 2)  | 0.60647909 | 0.11964861 | 0.19287671
# EfficientNetB6                 | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.57693786 | 0.07107248 | 0.16079418
# EfficientNetB6                 | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.59346587 | 0.07034628 | 0.16425909
# EfficientNetB6                 | per patch       | SVG                                      | (0, 1, 1)  | 0.58622483 | 0.08423210 | 0.19847890
# EfficientNetB6                 | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.57626749 | 0.07022635 | 0.16003435
# EfficientNetB6                 | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.59647982 | 0.07166390 | 0.16757785
# EfficientNetB6                 | per patch       | BalancedDistributionSVG/500/45/0.30      | (0, 1, 1)  | 0.58161237 | 0.08129778 | 0.16242538
# EfficientNetB6                 | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.56326767 | 0.07461375 | 0.19469457
# EfficientNetB6                 | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.58362128 | 0.06981038 | 0.19078434
# EfficientNetB6                 | per patch       | SVG                                      | (0, 2, 2)  | 0.56929097 | 0.08305481 | 0.20031149
# EfficientNetB6                 | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.56220147 | 0.07350729 | 0.19458935
# EfficientNetB6                 | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.58805786 | 0.07196112 | 0.19314571
# EfficientNetB6                 | per patch       | BalancedDistributionSVG/500/45/0.30      | (0, 2, 2)  | 0.56874371 | 0.08661101 | 0.19805353
# EfficientNetB6                 | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.58088312 | 0.06810708 | 0.11261209
# EfficientNetB6                 | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.60920904 | 0.07301615 | 0.12710466
# EfficientNetB6                 | per patch       | SVG                                      | (1, 0, 0)  | 0.59954379 | 0.09480642 | 0.16042289
# EfficientNetB6                 | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.58081494 | 0.06707346 | 0.11271905
# EfficientNetB6                 | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.60945819 | 0.07002678 | 0.12446945
# EfficientNetB6                 | per patch       | BalancedDistributionSVG/500/45/0.30      | (1, 0, 0)  | 0.58526404 | 0.08673140 | 0.15097160
# EfficientNetB6                 | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.56823548 | 0.06707383 | 0.10846405
# EfficientNetB6                 | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.59438573 | 0.07000182 | 0.12769752
# EfficientNetB6                 | per patch       | SVG                                      | (2, 0, 0)  | 0.59202451 | 0.10013080 | 0.15520728
# EfficientNetB6                 | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.56817833 | 0.06571065 | 0.10831700
# EfficientNetB6                 | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.59320443 | 0.06700687 | 0.12324930
# EfficientNetB6                 | per patch       | BalancedDistributionSVG/500/45/0.30      | (2, 0, 0)  | 0.57831638 | 0.09115874 | 0.13856146
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.49058369 | 0.44238163 | 0.63453815
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.43104457 | 0.39298279 | 0.61710037
# EfficientNetB6                 | per frame (max) | SVG                                      | None       | 0.51631770 | 0.52388102 | 0.61940299
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.48005615 | 0.43525582 | 0.63200000
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.39057200 | 0.36734209 | 0.61710037
# EfficientNetB6                 | per frame (max) | BalancedDistributionSVG/500/45/0.30      | None       | 0.51678559 | 0.54210364 | 0.62172285
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.54333840 | 0.50151301 | 0.62406015
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.45268452 | 0.42044176 | 0.61710037
# EfficientNetB6                 | per frame (max) | SVG                                      | (1, 1, 1)  | 0.59644403 | 0.57049406 | 0.63967611
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.53526728 | 0.48909573 | 0.62172285
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.36437010 | 0.35706966 | 0.62172285
# EfficientNetB6                 | per frame (max) | BalancedDistributionSVG/500/45/0.30      | (1, 1, 1)  | 0.57550591 | 0.55238117 | 0.63967611
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.52824892 | 0.51740198 | 0.62406015
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.48742543 | 0.43560903 | 0.62172285
# EfficientNetB6                 | per frame (max) | SVG                                      | (2, 2, 2)  | 0.56182010 | 0.57691826 | 0.64313725
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.52883378 | 0.51508548 | 0.62406015
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.41887940 | 0.38905712 | 0.62172285
# EfficientNetB6                 | per frame (max) | BalancedDistributionSVG/500/45/0.30      | (2, 2, 2)  | 0.55374898 | 0.56896569 | 0.64800000
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.51842321 | 0.46709075 | 0.62992126
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.40215230 | 0.39186192 | 0.61710037
# EfficientNetB6                 | per frame (max) | SVG                                      | (0, 1, 1)  | 0.59936835 | 0.56581271 | 0.61710037
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.50450345 | 0.45561579 | 0.62500000
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.37033571 | 0.36638825 | 0.62406015
# EfficientNetB6                 | per frame (max) | BalancedDistributionSVG/500/45/0.30      | (0, 1, 1)  | 0.59819862 | 0.57061772 | 0.61710037
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.56065037 | 0.51965476 | 0.62450593
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.46204234 | 0.41456102 | 0.61710037
# EfficientNetB6                 | per frame (max) | SVG                                      | (0, 2, 2)  | 0.60755644 | 0.57995962 | 0.61710037
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.54836823 | 0.50301933 | 0.62348178
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.44005147 | 0.39783614 | 0.62172285
# EfficientNetB6                 | per frame (max) | BalancedDistributionSVG/500/45/0.30      | (0, 2, 2)  | 0.59667797 | 0.56619807 | 0.61710037
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.54871915 | 0.47374445 | 0.65612648
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.43513861 | 0.40497524 | 0.62172285
# EfficientNetB6                 | per frame (max) | SVG                                      | (1, 0, 0)  | 0.55140952 | 0.54803409 | 0.62172285
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.54251959 | 0.46835713 | 0.65612648
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.34413382 | 0.34307725 | 0.61940299
# EfficientNetB6                 | per frame (max) | BalancedDistributionSVG/500/45/0.30      | (1, 0, 0)  | 0.53971225 | 0.53434821 | 0.62172285
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.54813428 | 0.48642650 | 0.65873016
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.40051468 | 0.39833405 | 0.61940299
# EfficientNetB6                 | per frame (max) | SVG                                      | (2, 0, 0)  | 0.55784302 | 0.56541593 | 0.61940299
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.53971225 | 0.48007402 | 0.65612648
# EfficientNetB6                 | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.26120014 | 0.31703097 | 0.61710037
# EfficientNetB6                 | per frame (max) | BalancedDistributionSVG/500/45/0.30      | (2, 0, 0)  | 0.54228565 | 0.54759196 | 0.61940299
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.56954030 | 0.49595129 | 0.64516129
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.53901041 | 0.46980986 | 0.62641509
# EfficientNetB6                 | per frame (sum) | SVG                                      | None       | 0.60077202 | 0.56740777 | 0.61710037
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.55421687 | 0.48798819 | 0.62978723
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.52719616 | 0.48530777 | 0.62172285
# EfficientNetB6                 | per frame (sum) | BalancedDistributionSVG/500/45/0.30      | None       | 0.60708855 | 0.57238769 | 0.62280702
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.60860919 | 0.55856810 | 0.65853659
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.57913206 | 0.49228809 | 0.62745098
# EfficientNetB6                 | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.59024447 | 0.59102526 | 0.64227642
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.59597614 | 0.54911092 | 0.66400000
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.56065037 | 0.51270827 | 0.62172285
# EfficientNetB6                 | per frame (sum) | BalancedDistributionSVG/500/45/0.30      | (1, 1, 1)  | 0.59585916 | 0.60476055 | 0.64516129
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.59866651 | 0.59237746 | 0.65098039
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.57035911 | 0.48493943 | 0.62878788
# EfficientNetB6                 | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.57176278 | 0.60488668 | 0.64566929
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.58720318 | 0.58711728 | 0.65098039
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.56275588 | 0.52151319 | 0.61940299
# EfficientNetB6                 | per frame (sum) | BalancedDistributionSVG/500/45/0.30      | (2, 2, 2)  | 0.57187975 | 0.60871655 | 0.65098039
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.56954030 | 0.49595129 | 0.64516129
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.53901041 | 0.46980986 | 0.62641509
# EfficientNetB6                 | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.60077202 | 0.56740777 | 0.61710037
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.55421687 | 0.48798819 | 0.62978723
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.52719616 | 0.48530777 | 0.62172285
# EfficientNetB6                 | per frame (sum) | BalancedDistributionSVG/500/45/0.30      | (0, 1, 1)  | 0.60708855 | 0.57238769 | 0.62280702
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.56954030 | 0.49595129 | 0.64516129
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.53901041 | 0.46980986 | 0.62641509
# EfficientNetB6                 | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.60077202 | 0.56740777 | 0.61710037
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.55421687 | 0.48798819 | 0.62978723
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.52719616 | 0.48530777 | 0.62172285
# EfficientNetB6                 | per frame (sum) | BalancedDistributionSVG/500/45/0.30      | (0, 2, 2)  | 0.60708855 | 0.57238769 | 0.62280702
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.60860919 | 0.55856810 | 0.65853659
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.57913206 | 0.49228809 | 0.62745098
# EfficientNetB6                 | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.59024447 | 0.59102526 | 0.64227642
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.59597614 | 0.54911092 | 0.66400000
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.56065037 | 0.51270827 | 0.62172285
# EfficientNetB6                 | per frame (sum) | BalancedDistributionSVG/500/45/0.30      | (1, 0, 0)  | 0.59585916 | 0.60476055 | 0.64516129
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.59866651 | 0.59237746 | 0.65098039
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.57035911 | 0.48493943 | 0.62878788
# EfficientNetB6                 | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.57176278 | 0.60488668 | 0.64566929
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.58720318 | 0.58711728 | 0.65098039
# EfficientNetB6                 | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.56275588 | 0.52151319 | 0.61940299
# EfficientNetB6                 | per frame (sum) | BalancedDistributionSVG/500/45/0.30      | (2, 0, 0)  | 0.57187975 | 0.60871655 | 0.65098039
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/0.20                      | None       | 0.60501100 | 0.07790247 | 0.18023933
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.61981958 | 0.07712178 | 0.17254421
# EfficientNetB6_Block6          | per patch       | SVG                                      | None       | 0.61300714 | 0.08884669 | 0.18957529
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/0.50                      | None       | 0.60403364 | 0.07771827 | 0.18032454
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.62216898 | 0.07825959 | 0.17400045
# EfficientNetB6_Block6          | per patch       | BalancedDistributionSVG/500/18/0.30      | None       | 0.57350816 | 0.08965861 | 0.19685834
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.66747039 | 0.10400002 | 0.18651713
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.67638815 | 0.09506549 | 0.18763531
# EfficientNetB6_Block6          | per patch       | SVG                                      | (1, 1, 1)  | 0.68296419 | 0.13135596 | 0.21996818
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.66689427 | 0.10350590 | 0.18601298
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.67880716 | 0.10309391 | 0.19355684
# EfficientNetB6_Block6          | per patch       | BalancedDistributionSVG/500/18/0.30      | (1, 1, 1)  | 0.61418946 | 0.12239764 | 0.22204370
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.68954985 | 0.12362778 | 0.19631700
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.68336694 | 0.08834310 | 0.18284574
# EfficientNetB6_Block6          | per patch       | SVG                                      | (2, 2, 2)  | 0.70392085 | 0.14647534 | 0.24207728
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.68957020 | 0.12299291 | 0.19584212
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.69554193 | 0.11773132 | 0.20237968
# EfficientNetB6_Block6          | per patch       | BalancedDistributionSVG/500/18/0.30      | (2, 2, 2)  | 0.61816812 | 0.13553843 | 0.24318038
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.61552404 | 0.08163530 | 0.18857722
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.62697234 | 0.07690487 | 0.18388294
# EfficientNetB6_Block6          | per patch       | SVG                                      | (0, 1, 1)  | 0.62043333 | 0.08905436 | 0.20304818
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.61445863 | 0.08138775 | 0.18917392
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.63101431 | 0.08046434 | 0.18708647
# EfficientNetB6_Block6          | per patch       | BalancedDistributionSVG/500/18/0.30      | (0, 1, 1)  | 0.56227934 | 0.09135151 | 0.20898669
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.61044530 | 0.07947328 | 0.20290209
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.62093737 | 0.07318983 | 0.19074010
# EfficientNetB6_Block6          | per patch       | SVG                                      | (0, 2, 2)  | 0.61673664 | 0.08565917 | 0.20476132
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.60909919 | 0.07919304 | 0.20295158
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.62921337 | 0.08014388 | 0.20082564
# EfficientNetB6_Block6          | per patch       | BalancedDistributionSVG/500/18/0.30      | (0, 2, 2)  | 0.54686558 | 0.09113141 | 0.20820268
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.64480777 | 0.08409056 | 0.17032290
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.66481961 | 0.09199845 | 0.17640095
# EfficientNetB6_Block6          | per patch       | SVG                                      | (1, 0, 0)  | 0.66300788 | 0.11664204 | 0.19706912
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.64432221 | 0.08382637 | 0.17037216
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.66397446 | 0.09077764 | 0.17573302
# EfficientNetB6_Block6          | per patch       | BalancedDistributionSVG/500/18/0.30      | (1, 0, 0)  | 0.60682989 | 0.10968558 | 0.20456438
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.64391887 | 0.07575167 | 0.14717833
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.66826888 | 0.08609949 | 0.17059651
# EfficientNetB6_Block6          | per patch       | SVG                                      | (2, 0, 0)  | 0.66739450 | 0.10549066 | 0.19108876
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.64375072 | 0.07549503 | 0.14680711
# EfficientNetB6_Block6          | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.66617867 | 0.08629618 | 0.17250812
# EfficientNetB6_Block6          | per patch       | BalancedDistributionSVG/500/18/0.30      | (2, 0, 0)  | 0.60274858 | 0.10336212 | 0.18995030
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.51994385 | 0.47072817 | 0.62903226
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.39817523 | 0.37890975 | 0.61940299
# EfficientNetB6_Block6          | per frame (max) | SVG                                      | None       | 0.62557024 | 0.57190615 | 0.65517241
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.50684291 | 0.46517760 | 0.62650602
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.37864078 | 0.36694302 | 0.61710037
# EfficientNetB6_Block6          | per frame (max) | BalancedDistributionSVG/500/18/0.30      | None       | 0.61843490 | 0.57244502 | 0.64978903
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.51081998 | 0.49678742 | 0.63745020
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.43724412 | 0.40667543 | 0.61940299
# EfficientNetB6_Block6          | per frame (max) | SVG                                      | (1, 1, 1)  | 0.57223067 | 0.59227871 | 0.61940299
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.50169610 | 0.49591386 | 0.64031621
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.33407416 | 0.33859769 | 0.61940299
# EfficientNetB6_Block6          | per frame (max) | BalancedDistributionSVG/500/18/0.30      | (1, 1, 1)  | 0.59527430 | 0.62005563 | 0.62698413
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.56696690 | 0.57119928 | 0.65863454
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.44402854 | 0.41128015 | 0.61710037
# EfficientNetB6_Block6          | per frame (max) | SVG                                      | (2, 2, 2)  | 0.57878114 | 0.61646690 | 0.62172285
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.55316411 | 0.56669403 | 0.65612648
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.38133115 | 0.35751819 | 0.61710037
# EfficientNetB6_Block6          | per frame (max) | BalancedDistributionSVG/500/18/0.30      | (2, 2, 2)  | 0.58813896 | 0.63418438 | 0.62641509
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.48672359 | 0.46722993 | 0.62903226
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.42133583 | 0.39306583 | 0.61940299
# EfficientNetB6_Block6          | per frame (max) | SVG                                      | (0, 1, 1)  | 0.48754240 | 0.50589548 | 0.61940299
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.47993917 | 0.46413871 | 0.62903226
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.38729676 | 0.36871443 | 0.61710037
# EfficientNetB6_Block6          | per frame (max) | BalancedDistributionSVG/500/18/0.30      | (0, 1, 1)  | 0.62252895 | 0.59424809 | 0.63241107
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.53795766 | 0.49105723 | 0.62931034
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.43864780 | 0.40347995 | 0.61710037
# EfficientNetB6_Block6          | per frame (max) | SVG                                      | (0, 2, 2)  | 0.64101064 | 0.60438287 | 0.65833333
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.51900807 | 0.48184821 | 0.62903226
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.40449175 | 0.38090250 | 0.61710037
# EfficientNetB6_Block6          | per frame (max) | BalancedDistributionSVG/500/18/0.30      | (0, 2, 2)  | 0.62054041 | 0.60403412 | 0.61710037
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.48122587 | 0.43997391 | 0.62903226
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.43513861 | 0.40716807 | 0.61940299
# EfficientNetB6_Block6          | per frame (max) | SVG                                      | (1, 0, 0)  | 0.58521464 | 0.59035162 | 0.62790698
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.47292081 | 0.43083223 | 0.62903226
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.31091356 | 0.33103460 | 0.61710037
# EfficientNetB6_Block6          | per frame (max) | BalancedDistributionSVG/500/18/0.30      | (1, 0, 0)  | 0.59878348 | 0.61432224 | 0.62641509
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.44192303 | 0.38854988 | 0.63320463
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.38378758 | 0.39040379 | 0.62172285
# EfficientNetB6_Block6          | per frame (max) | SVG                                      | (2, 0, 0)  | 0.57187975 | 0.61123743 | 0.61940299
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.43876477 | 0.39321408 | 0.63358779
# EfficientNetB6_Block6          | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.24236753 | 0.30984141 | 0.62878788
# EfficientNetB6_Block6          | per frame (max) | BalancedDistributionSVG/500/18/0.30      | (2, 0, 0)  | 0.58954264 | 0.62450685 | 0.62015504
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.54368932 | 0.49203948 | 0.63900415
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.53772371 | 0.46966234 | 0.63320463
# EfficientNetB6_Block6          | per frame (sum) | SVG                                      | None       | 0.58708621 | 0.56884859 | 0.61710037
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.52357001 | 0.48018144 | 0.63200000
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.53491636 | 0.50256046 | 0.62790698
# EfficientNetB6_Block6          | per frame (sum) | BalancedDistributionSVG/500/18/0.30      | None       | 0.59188209 | 0.57604262 | 0.61710037
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.57913206 | 0.55474445 | 0.66400000
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.57796233 | 0.48688173 | 0.63846154
# EfficientNetB6_Block6          | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.56778571 | 0.59330734 | 0.61960784
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.56287285 | 0.54849130 | 0.66400000
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.56965727 | 0.55020432 | 0.63117871
# EfficientNetB6_Block6          | per frame (sum) | BalancedDistributionSVG/500/18/0.30      | (1, 1, 1)  | 0.56942332 | 0.60446626 | 0.62307692
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.58872383 | 0.59827709 | 0.65873016
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.55035677 | 0.46973748 | 0.63358779
# EfficientNetB6_Block6          | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.55725816 | 0.61052845 | 0.61710037
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.57340040 | 0.59072367 | 0.66666667
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.57843023 | 0.58741431 | 0.62641509
# EfficientNetB6_Block6          | per frame (sum) | BalancedDistributionSVG/500/18/0.30      | (2, 2, 2)  | 0.56790268 | 0.62605544 | 0.62068966
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.54368932 | 0.49203948 | 0.63900415
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.53772371 | 0.46966234 | 0.63320463
# EfficientNetB6_Block6          | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.58708621 | 0.56884859 | 0.61710037
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.52357001 | 0.48018144 | 0.63200000
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.53491636 | 0.50256046 | 0.62790698
# EfficientNetB6_Block6          | per frame (sum) | BalancedDistributionSVG/500/18/0.30      | (0, 1, 1)  | 0.59188209 | 0.57604262 | 0.61710037
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.54368932 | 0.49203948 | 0.63900415
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.53772371 | 0.46966234 | 0.63320463
# EfficientNetB6_Block6          | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.58708621 | 0.56884859 | 0.61710037
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.52357001 | 0.48018144 | 0.63200000
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.53491636 | 0.50256046 | 0.62790698
# EfficientNetB6_Block6          | per frame (sum) | BalancedDistributionSVG/500/18/0.30      | (0, 2, 2)  | 0.59188209 | 0.57604262 | 0.61710037
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.57913206 | 0.55474445 | 0.66400000
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.57796233 | 0.48688173 | 0.63846154
# EfficientNetB6_Block6          | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.56778571 | 0.59330734 | 0.61960784
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.56287285 | 0.54849130 | 0.66400000
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.56965727 | 0.55020432 | 0.63117871
# EfficientNetB6_Block6          | per frame (sum) | BalancedDistributionSVG/500/18/0.30      | (1, 0, 0)  | 0.56942332 | 0.60446626 | 0.62307692
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.58872383 | 0.59827709 | 0.65873016
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.55035677 | 0.46973748 | 0.63358779
# EfficientNetB6_Block6          | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.55725816 | 0.61052845 | 0.61710037
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.57340040 | 0.59072367 | 0.66666667
# EfficientNetB6_Block6          | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.57843023 | 0.58741431 | 0.62641509
# EfficientNetB6_Block6          | per frame (sum) | BalancedDistributionSVG/500/18/0.30      | (2, 0, 0)  | 0.56790268 | 0.62605544 | 0.62068966
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/0.20                      | None       | 0.90213678 | 0.27954741 | 0.39866555
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.93712990 | 0.39315326 | 0.49361702
# MobileNetV2_Block16            | per patch       | SVG                                      | None       | 0.88510760 | 0.21757480 | 0.35031403
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.93354472 | 0.40738521 | 0.51132075
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/0.50                      | None       | 0.90152057 | 0.27654801 | 0.39744613
# MobileNetV2_Block16            | per patch       | BalancedDistributionSVG/500/17/0.30      | None       | 0.88927164 | 0.24690764 | 0.37868852
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.93802984 | 0.39720337 | 0.47977941
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.96923827 | 0.57132969 | 0.61489362
# MobileNetV2_Block16            | per patch       | SVG                                      | (1, 1, 1)  | 0.91646707 | 0.26170235 | 0.41498559
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.96180385 | 0.54582990 | 0.59894180
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.93737454 | 0.38732482 | 0.47905282
# MobileNetV2_Block16            | per patch       | BalancedDistributionSVG/500/17/0.30      | (1, 1, 1)  | 0.92828083 | 0.31367519 | 0.44717445
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.93081288 | 0.39845871 | 0.45161290
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.95257571 | 0.46581314 | 0.53800000
# MobileNetV2_Block16            | per patch       | SVG                                      | (2, 2, 2)  | 0.90145898 | 0.25476158 | 0.37138264
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.94880802 | 0.47097714 | 0.51063830
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.93001778 | 0.39439644 | 0.44941808
# MobileNetV2_Block16            | per patch       | BalancedDistributionSVG/500/17/0.30      | (2, 2, 2)  | 0.91636744 | 0.31186918 | 0.40544218
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.94208595 | 0.42732900 | 0.47875648
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.96880229 | 0.60090983 | 0.62605042
# MobileNetV2_Block16            | per patch       | SVG                                      | (0, 1, 1)  | 0.92182228 | 0.28146514 | 0.41571429
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.96060277 | 0.56548035 | 0.62513200
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.94145903 | 0.41955848 | 0.47814451
# MobileNetV2_Block16            | per patch       | BalancedDistributionSVG/500/17/0.30      | (0, 1, 1)  | 0.93006411 | 0.33604611 | 0.45240175
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.94484564 | 0.46766667 | 0.50181818
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.96109097 | 0.55755174 | 0.61054994
# MobileNetV2_Block16            | per patch       | SVG                                      | (0, 2, 2)  | 0.92046506 | 0.29280432 | 0.42294521
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.95629599 | 0.53480234 | 0.56928034
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.94416810 | 0.46319413 | 0.50090416
# MobileNetV2_Block16            | per patch       | BalancedDistributionSVG/500/17/0.30      | (0, 2, 2)  | 0.92779450 | 0.35375537 | 0.45461979
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.89648431 | 0.26594197 | 0.37637795
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.93618175 | 0.37597777 | 0.48244824
# MobileNetV2_Block16            | per patch       | SVG                                      | (1, 0, 0)  | 0.87877894 | 0.20805968 | 0.33458930
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.93119798 | 0.38907423 | 0.49170732
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.89586194 | 0.26326358 | 0.37628458
# MobileNetV2_Block16            | per patch       | BalancedDistributionSVG/500/17/0.30      | (1, 0, 0)  | 0.88560384 | 0.23811809 | 0.36377953
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.87378740 | 0.23813829 | 0.33798450
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.91317901 | 0.32723557 | 0.41188960
# MobileNetV2_Block16            | per patch       | SVG                                      | (2, 0, 0)  | 0.85571540 | 0.19205439 | 0.30663929
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.90657611 | 0.33245580 | 0.42633229
# MobileNetV2_Block16            | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.87311308 | 0.23597654 | 0.33564280
# MobileNetV2_Block16            | per patch       | BalancedDistributionSVG/500/17/0.30      | (2, 0, 0)  | 0.86184001 | 0.21628970 | 0.33102493
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.72207276 | 0.66142437 | 0.69090909
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.73552462 | 0.61841138 | 0.69662921
# MobileNetV2_Block16            | per frame (max) | SVG                                      | None       | 0.63867119 | 0.56120275 | 0.66292135
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.74067142 | 0.63119135 | 0.69318182
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.71903147 | 0.65762240 | 0.68711656
# MobileNetV2_Block16            | per frame (max) | BalancedDistributionSVG/500/17/0.30      | None       | 0.65914142 | 0.58341453 | 0.65921788
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.75377237 | 0.73219035 | 0.72941176
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.77646508 | 0.71478737 | 0.70786517
# MobileNetV2_Block16            | per frame (max) | SVG                                      | (1, 1, 1)  | 0.66148087 | 0.55963355 | 0.68108108
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.77716692 | 0.69989572 | 0.70588235
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.74815768 | 0.72246093 | 0.72514620
# MobileNetV2_Block16            | per frame (max) | BalancedDistributionSVG/500/17/0.30      | (1, 1, 1)  | 0.68546029 | 0.59303026 | 0.69318182
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.78863025 | 0.76603862 | 0.71186441
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.80535735 | 0.81835065 | 0.72392638
# MobileNetV2_Block16            | per frame (max) | SVG                                      | (2, 2, 2)  | 0.67680430 | 0.55453932 | 0.66355140
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.79237338 | 0.77377780 | 0.72289157
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.78348345 | 0.75530424 | 0.71186441
# MobileNetV2_Block16            | per frame (max) | BalancedDistributionSVG/500/17/0.30      | (2, 2, 2)  | 0.70733419 | 0.60204376 | 0.67777778
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.72569891 | 0.70233978 | 0.68717949
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.75412329 | 0.69424126 | 0.70050761
# MobileNetV2_Block16            | per frame (max) | SVG                                      | (0, 1, 1)  | 0.63867119 | 0.53835542 | 0.64615385
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.77178617 | 0.70537211 | 0.71578947
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.72312551 | 0.69796145 | 0.68717949
# MobileNetV2_Block16            | per frame (max) | BalancedDistributionSVG/500/17/0.30      | (0, 1, 1)  | 0.66206574 | 0.57513367 | 0.65608466
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.75435723 | 0.71872543 | 0.72093023
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.79927477 | 0.81151112 | 0.72514620
# MobileNetV2_Block16            | per frame (max) | SVG                                      | (0, 2, 2)  | 0.66826529 | 0.56102336 | 0.65989848
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.80114633 | 0.76620634 | 0.73372781
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.75131594 | 0.71484633 | 0.72093023
# MobileNetV2_Block16            | per frame (max) | BalancedDistributionSVG/500/17/0.30      | (0, 2, 2)  | 0.68932039 | 0.60175449 | 0.67391304
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.75377237 | 0.67235266 | 0.71676301
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.75751550 | 0.63362540 | 0.73255814
# MobileNetV2_Block16            | per frame (max) | SVG                                      | (1, 0, 0)  | 0.66651070 | 0.57581581 | 0.65968586
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.73985261 | 0.61516400 | 0.70175439
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.75073108 | 0.66735226 | 0.71264368
# MobileNetV2_Block16            | per frame (max) | BalancedDistributionSVG/500/17/0.30      | (1, 0, 0)  | 0.68405661 | 0.59574158 | 0.67021277
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.78593988 | 0.70250811 | 0.75151515
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.78477015 | 0.66019641 | 0.76744186
# MobileNetV2_Block16            | per frame (max) | SVG                                      | (2, 0, 0)  | 0.68405661 | 0.58682719 | 0.66666667
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.74885952 | 0.61475687 | 0.71428571
# MobileNetV2_Block16            | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.78219675 | 0.68903542 | 0.74698795
# MobileNetV2_Block16            | per frame (max) | BalancedDistributionSVG/500/17/0.30      | (2, 0, 0)  | 0.70008188 | 0.60485909 | 0.68571429
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.82442391 | 0.82627775 | 0.74025974
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.79366008 | 0.79666078 | 0.72000000
# MobileNetV2_Block16            | per frame (sum) | SVG                                      | None       | 0.70780208 | 0.66592071 | 0.65178571
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.79763715 | 0.79516144 | 0.73255814
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.81904316 | 0.82063919 | 0.74698795
# MobileNetV2_Block16            | per frame (sum) | BalancedDistributionSVG/500/17/0.30      | None       | 0.72488010 | 0.69904584 | 0.68062827
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.84290560 | 0.84215516 | 0.76300578
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.79892385 | 0.80288766 | 0.72392638
# MobileNetV2_Block16            | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.71446953 | 0.66997841 | 0.67724868
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.79658440 | 0.79603484 | 0.73142857
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.83635513 | 0.83595082 | 0.76300578
# MobileNetV2_Block16            | per frame (sum) | BalancedDistributionSVG/500/17/0.30      | (1, 1, 1)  | 0.73505673 | 0.69826757 | 0.69109948
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.84571295 | 0.84435378 | 0.77419355
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.79132062 | 0.78054351 | 0.70935961
# MobileNetV2_Block16            | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.69306352 | 0.64570500 | 0.67906977
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.78196280 | 0.76615826 | 0.71764706
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.84126798 | 0.84014500 | 0.76433121
# MobileNetV2_Block16            | per frame (sum) | BalancedDistributionSVG/500/17/0.30      | (2, 2, 2)  | 0.71809568 | 0.67230096 | 0.70588235
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.82442391 | 0.82627775 | 0.74025974
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.79366008 | 0.79666078 | 0.72000000
# MobileNetV2_Block16            | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.70780208 | 0.66592071 | 0.65178571
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.79763715 | 0.79516144 | 0.73255814
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.81904316 | 0.82063919 | 0.74698795
# MobileNetV2_Block16            | per frame (sum) | BalancedDistributionSVG/500/17/0.30      | (0, 1, 1)  | 0.72488010 | 0.69904584 | 0.68062827
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.82442391 | 0.82627775 | 0.74025974
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.79366008 | 0.79666078 | 0.72000000
# MobileNetV2_Block16            | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.70780208 | 0.66592071 | 0.65178571
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.79763715 | 0.79516144 | 0.73255814
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.81904316 | 0.82063919 | 0.74698795
# MobileNetV2_Block16            | per frame (sum) | BalancedDistributionSVG/500/17/0.30      | (0, 2, 2)  | 0.72488010 | 0.69904584 | 0.68062827
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.84290560 | 0.84215516 | 0.76300578
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.79892385 | 0.80288766 | 0.72392638
# MobileNetV2_Block16            | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.71446953 | 0.66997841 | 0.67724868
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.79658440 | 0.79603484 | 0.73142857
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.83635513 | 0.83595082 | 0.76300578
# MobileNetV2_Block16            | per frame (sum) | BalancedDistributionSVG/500/17/0.30      | (1, 0, 0)  | 0.73505673 | 0.69826757 | 0.69109948
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.84571295 | 0.84435378 | 0.77419355
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.79132062 | 0.78054351 | 0.70935961
# MobileNetV2_Block16            | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.69306352 | 0.64570500 | 0.67906977
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.78196280 | 0.76615826 | 0.71764706
# MobileNetV2_Block16            | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.84126798 | 0.84014500 | 0.76433121
# MobileNetV2_Block16            | per frame (sum) | BalancedDistributionSVG/500/17/0.30      | (2, 0, 0)  | 0.71809568 | 0.67230096 | 0.70588235
# ResNet50V2                     | per patch       | SpatialBin/SVG/0.20                      | None       | 0.93894063 | 0.41071754 | 0.49546828
# ResNet50V2                     | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.92291048 | 0.34078609 | 0.44878049
# ResNet50V2                     | per patch       | BalancedDistributionSVG/500/63/0.30      | None       | 0.90150825 | 0.39231255 | 0.46831956
# ResNet50V2                     | per patch       | SpatialBin/SVG/0.50                      | None       | 0.93931100 | 0.41341221 | 0.50387597
# ResNet50V2                     | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.92035687 | 0.35390601 | 0.46153846
# ResNet50V2                     | per patch       | SVG                                      | None       | 0.91288898 | 0.42438647 | 0.46630237
# ResNet50V2                     | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.94125443 | 0.36269405 | 0.47217538
# ResNet50V2                     | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.92741663 | 0.32842181 | 0.47467167
# ResNet50V2                     | per patch       | BalancedDistributionSVG/500/63/0.30      | (1, 1, 1)  | 0.92461062 | 0.30892651 | 0.44850949
# ResNet50V2                     | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.93999416 | 0.33498853 | 0.46531303
# ResNet50V2                     | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.92864477 | 0.33074347 | 0.45963756
# ResNet50V2                     | per patch       | SVG                                      | (1, 1, 1)  | 0.92736495 | 0.35334374 | 0.44284687
# ResNet50V2                     | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.90323718 | 0.30904166 | 0.37422553
# ResNet50V2                     | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.88185972 | 0.22985470 | 0.37555228
# ResNet50V2                     | per patch       | BalancedDistributionSVG/500/63/0.30      | (2, 2, 2)  | 0.88202602 | 0.21234024 | 0.34190078
# ResNet50V2                     | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.89678452 | 0.23990164 | 0.35485818
# ResNet50V2                     | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.89109670 | 0.25648623 | 0.38466899
# ResNet50V2                     | per patch       | SVG                                      | (2, 2, 2)  | 0.88979251 | 0.29265673 | 0.41233141
# ResNet50V2                     | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.93584352 | 0.37677375 | 0.49288061
# ResNet50V2                     | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.92525575 | 0.33449074 | 0.45555556
# ResNet50V2                     | per patch       | BalancedDistributionSVG/500/63/0.30      | (0, 1, 1)  | 0.92184317 | 0.36326331 | 0.49139579
# ResNet50V2                     | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.93482989 | 0.36311022 | 0.48746239
# ResNet50V2                     | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.92628410 | 0.33335398 | 0.45992366
# ResNet50V2                     | per patch       | SVG                                      | (0, 1, 1)  | 0.92524182 | 0.37541820 | 0.47078747
# ResNet50V2                     | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.90978597 | 0.30690624 | 0.42007435
# ResNet50V2                     | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.89673203 | 0.26523018 | 0.41706924
# ResNet50V2                     | per patch       | BalancedDistributionSVG/500/63/0.30      | (0, 2, 2)  | 0.90208001 | 0.28588355 | 0.43164230
# ResNet50V2                     | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.90814543 | 0.28165198 | 0.41269841
# ResNet50V2                     | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.89692940 | 0.26278063 | 0.39080460
# ResNet50V2                     | per patch       | SVG                                      | (0, 2, 2)  | 0.89867412 | 0.28905028 | 0.39932318
# ResNet50V2                     | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.94286391 | 0.38492496 | 0.48243115
# ResNet50V2                     | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.92875349 | 0.32663946 | 0.44244898
# ResNet50V2                     | per patch       | BalancedDistributionSVG/500/63/0.30      | (1, 0, 0)  | 0.91311661 | 0.34795037 | 0.45878848
# ResNet50V2                     | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.94316653 | 0.38444453 | 0.48991935
# ResNet50V2                     | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.92683283 | 0.33393934 | 0.44626866
# ResNet50V2                     | per patch       | SVG                                      | (1, 0, 0)  | 0.92326438 | 0.37734421 | 0.45311049
# ResNet50V2                     | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.92708831 | 0.32842512 | 0.44380952
# ResNet50V2                     | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.91223126 | 0.27535871 | 0.39531250
# ResNet50V2                     | per patch       | BalancedDistributionSVG/500/63/0.30      | (2, 0, 0)  | 0.89480868 | 0.27444014 | 0.39930253
# ResNet50V2                     | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.92718740 | 0.31993966 | 0.44500420
# ResNet50V2                     | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.91322052 | 0.28313759 | 0.40207075
# ResNet50V2                     | per patch       | SVG                                      | (2, 0, 0)  | 0.90407004 | 0.30259047 | 0.40796020
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.69692362 | 0.61953106 | 0.67873303
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.74792373 | 0.66510988 | 0.73404255
# ResNet50V2                     | per frame (max) | BalancedDistributionSVG/500/63/0.30      | None       | 0.67083869 | 0.59880346 | 0.63111111
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.71213007 | 0.63962085 | 0.67532468
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.74570125 | 0.66364001 | 0.70652174
# ResNet50V2                     | per frame (max) | SVG                                      | None       | 0.64943268 | 0.62362107 | 0.65454545
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.73868289 | 0.68639606 | 0.69387755
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.72651772 | 0.61675141 | 0.69945355
# ResNet50V2                     | per frame (max) | BalancedDistributionSVG/500/63/0.30      | (1, 1, 1)  | 0.65224003 | 0.53021733 | 0.68717949
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.74967832 | 0.67164837 | 0.69841270
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.73306820 | 0.61485957 | 0.71005917
# ResNet50V2                     | per frame (max) | SVG                                      | (1, 1, 1)  | 0.64463680 | 0.62312546 | 0.64462810
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.70335712 | 0.65994365 | 0.66037736
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.70417593 | 0.60142640 | 0.68161435
# ResNet50V2                     | per frame (max) | BalancedDistributionSVG/500/63/0.30      | (2, 2, 2)  | 0.61738215 | 0.50596944 | 0.67213115
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.71411861 | 0.62665731 | 0.67619048
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.71201310 | 0.61062076 | 0.67906977
# ResNet50V2                     | per frame (max) | SVG                                      | (2, 2, 2)  | 0.65528132 | 0.64273200 | 0.64341085
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.74230904 | 0.67116490 | 0.70212766
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.76043982 | 0.67889957 | 0.72727273
# ResNet50V2                     | per frame (max) | BalancedDistributionSVG/500/63/0.30      | (0, 1, 1)  | 0.68756580 | 0.60511811 | 0.65024631
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.75646274 | 0.68038604 | 0.69565217
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.76441689 | 0.68127382 | 0.71219512
# ResNet50V2                     | per frame (max) | SVG                                      | (0, 1, 1)  | 0.66101298 | 0.62938625 | 0.64150943
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.74710492 | 0.68770436 | 0.68421053
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.75400632 | 0.68568749 | 0.72251309
# ResNet50V2                     | per frame (max) | BalancedDistributionSVG/500/63/0.30      | (0, 2, 2)  | 0.66148087 | 0.59420809 | 0.64220183
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.75915312 | 0.69275267 | 0.69273743
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.76254533 | 0.69478848 | 0.71794872
# ResNet50V2                     | per frame (max) | SVG                                      | (0, 2, 2)  | 0.64299918 | 0.62864376 | 0.62978723
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.67212539 | 0.61569372 | 0.64317181
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.71879752 | 0.60597245 | 0.71794872
# ResNet50V2                     | per frame (max) | BalancedDistributionSVG/500/63/0.30      | (1, 0, 0)  | 0.62510235 | 0.51307208 | 0.65833333
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.68195111 | 0.61267391 | 0.64347826
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.71751082 | 0.58773300 | 0.72527473
# ResNet50V2                     | per frame (max) | SVG                                      | (1, 0, 0)  | 0.66253363 | 0.63647122 | 0.64680851
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.62697392 | 0.61265792 | 0.63114754
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.71423558 | 0.58194607 | 0.71770335
# ResNet50V2                     | per frame (max) | BalancedDistributionSVG/500/63/0.30      | (2, 0, 0)  | 0.59141420 | 0.48385514 | 0.68067227
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.64685928 | 0.58324691 | 0.63865546
# ResNet50V2                     | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.69201076 | 0.54764989 | 0.72727273
# ResNet50V2                     | per frame (max) | SVG                                      | (2, 0, 0)  | 0.68312083 | 0.65124823 | 0.63755459
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.76067376 | 0.70854336 | 0.69892473
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.76570359 | 0.71949228 | 0.70351759
# ResNet50V2                     | per frame (sum) | BalancedDistributionSVG/500/63/0.30      | None       | 0.69095801 | 0.63727857 | 0.65048544
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.77166920 | 0.71464972 | 0.71134021
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.76757515 | 0.71011404 | 0.70297030
# ResNet50V2                     | per frame (sum) | SVG                                      | None       | 0.68885250 | 0.68016746 | 0.66028708
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.75576091 | 0.73124491 | 0.68899522
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.74733887 | 0.66532306 | 0.70270270
# ResNet50V2                     | per frame (sum) | BalancedDistributionSVG/500/63/0.30      | (1, 1, 1)  | 0.67809100 | 0.59879846 | 0.65671642
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.76418295 | 0.70143922 | 0.70476190
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.75108200 | 0.66887234 | 0.71084337
# ResNet50V2                     | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.68393964 | 0.68331887 | 0.66359447
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.72803837 | 0.69759275 | 0.68108108
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.73377003 | 0.65121081 | 0.69035533
# ResNet50V2                     | per frame (sum) | BalancedDistributionSVG/500/63/0.30      | (2, 2, 2)  | 0.66335244 | 0.57264480 | 0.68421053
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.73587554 | 0.67475580 | 0.70466321
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.73856591 | 0.66280633 | 0.67873303
# ResNet50V2                     | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.69692362 | 0.68361039 | 0.66949153
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.76067376 | 0.70854336 | 0.69892473
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.76570359 | 0.71949228 | 0.70351759
# ResNet50V2                     | per frame (sum) | BalancedDistributionSVG/500/63/0.30      | (0, 1, 1)  | 0.69095801 | 0.63727857 | 0.65048544
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.77166920 | 0.71464972 | 0.71134021
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.76757515 | 0.71011404 | 0.70297030
# ResNet50V2                     | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.68885250 | 0.68016746 | 0.66028708
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.76067376 | 0.70854336 | 0.69892473
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.76570359 | 0.71949228 | 0.70351759
# ResNet50V2                     | per frame (sum) | BalancedDistributionSVG/500/63/0.30      | (0, 2, 2)  | 0.69095801 | 0.63727857 | 0.65048544
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.77166920 | 0.71464972 | 0.71134021
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.76757515 | 0.71011404 | 0.70297030
# ResNet50V2                     | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.68885250 | 0.68016746 | 0.66028708
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.75576091 | 0.73124491 | 0.68899522
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.74733887 | 0.66532306 | 0.70270270
# ResNet50V2                     | per frame (sum) | BalancedDistributionSVG/500/63/0.30      | (1, 0, 0)  | 0.67809100 | 0.59879846 | 0.65671642
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.76418295 | 0.70143922 | 0.70476190
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.75108200 | 0.66887234 | 0.71084337
# ResNet50V2                     | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.68393964 | 0.68331887 | 0.66359447
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.72803837 | 0.69759275 | 0.68108108
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.73377003 | 0.65121081 | 0.69035533
# ResNet50V2                     | per frame (sum) | BalancedDistributionSVG/500/63/0.30      | (2, 0, 0)  | 0.66335244 | 0.57264480 | 0.68421053
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.73587554 | 0.67475580 | 0.70466321
# ResNet50V2                     | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.73856591 | 0.66280633 | 0.67873303
# ResNet50V2                     | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.69692362 | 0.68361039 | 0.66949153
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/0.20                      | None       | 0.86471056 | 0.17352491 | 0.26234732
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.86693773 | 0.21887022 | 0.28955533
# ResNet50V2_Stack4              | per patch       | SVG                                      | None       | 0.86059446 | 0.15945814 | 0.25825572
# ResNet50V2_Stack4              | per patch       | BalancedDistributionSVG/500/31/0.30      | None       | 0.86418299 | 0.17607135 | 0.28004535
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/0.50                      | None       | 0.86478581 | 0.17329442 | 0.26295690
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.86048092 | 0.21596979 | 0.29663330
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.89152009 | 0.18164917 | 0.32340010
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.91446981 | 0.26784224 | 0.36808957
# ResNet50V2_Stack4              | per patch       | SVG                                      | (1, 1, 1)  | 0.88478463 | 0.17328281 | 0.30989583
# ResNet50V2_Stack4              | per patch       | BalancedDistributionSVG/500/31/0.30      | (1, 1, 1)  | 0.90459385 | 0.21906757 | 0.33235867
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.89129915 | 0.18089848 | 0.32258065
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.89261540 | 0.22846091 | 0.34322581
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.88789755 | 0.20814730 | 0.32963989
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.89679898 | 0.22418226 | 0.36266982
# ResNet50V2_Stack4              | per patch       | SVG                                      | (2, 2, 2)  | 0.87870770 | 0.20939376 | 0.31100478
# ResNet50V2_Stack4              | per patch       | BalancedDistributionSVG/500/31/0.30      | (2, 2, 2)  | 0.89522458 | 0.22454821 | 0.34604520
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.88758155 | 0.20867006 | 0.32862454
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.87374134 | 0.17388117 | 0.30559611
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.90267051 | 0.22880908 | 0.34028114
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.91876399 | 0.29307473 | 0.38846480
# ResNet50V2_Stack4              | per patch       | SVG                                      | (0, 1, 1)  | 0.89460864 | 0.20590319 | 0.32311015
# ResNet50V2_Stack4              | per patch       | BalancedDistributionSVG/500/31/0.30      | (0, 1, 1)  | 0.90710877 | 0.25697996 | 0.35108481
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.90243136 | 0.22793271 | 0.33901607
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.90099193 | 0.25726549 | 0.37848606
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.91522876 | 0.28097038 | 0.38554217
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.91600270 | 0.27843364 | 0.38185654
# ResNet50V2_Stack4              | per patch       | SVG                                      | (0, 2, 2)  | 0.90619584 | 0.26078501 | 0.36666667
# ResNet50V2_Stack4              | per patch       | BalancedDistributionSVG/500/31/0.30      | (0, 2, 2)  | 0.91307350 | 0.29678433 | 0.39017544
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.91499925 | 0.28213623 | 0.38772455
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.89663107 | 0.22964487 | 0.36000000
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.86126343 | 0.15600569 | 0.25330688
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.87193395 | 0.20416267 | 0.28368794
# ResNet50V2_Stack4              | per patch       | SVG                                      | (1, 0, 0)  | 0.85920993 | 0.14872190 | 0.25673250
# ResNet50V2_Stack4              | per patch       | BalancedDistributionSVG/500/31/0.30      | (1, 0, 0)  | 0.87407957 | 0.16948450 | 0.28867102
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.86134511 | 0.15570214 | 0.25374625
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.85898445 | 0.19445739 | 0.28808446
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.84570582 | 0.13819199 | 0.24273504
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.85827076 | 0.17426636 | 0.26741996
# ResNet50V2_Stack4              | per patch       | SVG                                      | (2, 0, 0)  | 0.84582686 | 0.13664079 | 0.23758099
# ResNet50V2_Stack4              | per patch       | BalancedDistributionSVG/500/31/0.30      | (2, 0, 0)  | 0.86338896 | 0.15578257 | 0.27809308
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.84591416 | 0.13816080 | 0.24211597
# ResNet50V2_Stack4              | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.84511371 | 0.16664779 | 0.26657163
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.64510469 | 0.57928263 | 0.63076923
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.66487308 | 0.54339876 | 0.67661692
# ResNet50V2_Stack4              | per frame (max) | SVG                                      | None       | 0.60276056 | 0.54347493 | 0.62406015
# ResNet50V2_Stack4              | per frame (max) | BalancedDistributionSVG/500/31/0.30      | None       | 0.61059773 | 0.55837553 | 0.61940299
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.64241432 | 0.57626293 | 0.63076923
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.68043046 | 0.56502872 | 0.66666667
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.63913908 | 0.53541792 | 0.65326633
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.64756112 | 0.51658130 | 0.66367713
# ResNet50V2_Stack4              | per frame (max) | SVG                                      | (1, 1, 1)  | 0.56720084 | 0.48594820 | 0.62641509
# ResNet50V2_Stack4              | per frame (max) | BalancedDistributionSVG/500/31/0.30      | (1, 1, 1)  | 0.57843023 | 0.56785030 | 0.62406015
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.63621476 | 0.53340599 | 0.64646465
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.66604281 | 0.53294300 | 0.66985646
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.70593052 | 0.64255175 | 0.67428571
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.63715054 | 0.51122438 | 0.65789474
# ResNet50V2_Stack4              | per frame (max) | SVG                                      | (2, 2, 2)  | 0.64978360 | 0.62733407 | 0.63492063
# ResNet50V2_Stack4              | per frame (max) | BalancedDistributionSVG/500/31/0.30      | (2, 2, 2)  | 0.63586384 | 0.62396584 | 0.63101604
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.70429290 | 0.64042724 | 0.66666667
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.63352439 | 0.49696039 | 0.68398268
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.65656802 | 0.57839139 | 0.64921466
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.68253597 | 0.56041699 | 0.67906977
# ResNet50V2_Stack4              | per frame (max) | SVG                                      | (0, 1, 1)  | 0.58965961 | 0.52165130 | 0.61940299
# ResNet50V2_Stack4              | per frame (max) | BalancedDistributionSVG/500/31/0.30      | (0, 1, 1)  | 0.60299450 | 0.58591378 | 0.62222222
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.65504737 | 0.57566246 | 0.64921466
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.68054743 | 0.56199370 | 0.65454545
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.68464148 | 0.63418771 | 0.64390244
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.69587086 | 0.58007543 | 0.69189189
# ResNet50V2_Stack4              | per frame (max) | SVG                                      | (0, 2, 2)  | 0.63562990 | 0.63784452 | 0.63358779
# ResNet50V2_Stack4              | per frame (max) | BalancedDistributionSVG/500/31/0.30      | (0, 2, 2)  | 0.63399228 | 0.65017778 | 0.62256809
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.68347175 | 0.63598785 | 0.64114833
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.69867821 | 0.57297891 | 0.69696970
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.54520997 | 0.50154305 | 0.64591440
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.59632706 | 0.48543179 | 0.63598326
# ResNet50V2_Stack4              | per frame (max) | SVG                                      | (1, 0, 0)  | 0.51432916 | 0.49183083 | 0.64257028
# ResNet50V2_Stack4              | per frame (max) | BalancedDistributionSVG/500/31/0.30      | (1, 0, 0)  | 0.54064803 | 0.51999241 | 0.62641509
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.54216867 | 0.49463695 | 0.64591440
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.60662066 | 0.50907347 | 0.62882096
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.51877413 | 0.50332232 | 0.64843750
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.49608141 | 0.42178608 | 0.61710037
# ResNet50V2_Stack4              | per frame (max) | SVG                                      | (2, 0, 0)  | 0.49888876 | 0.49895861 | 0.64591440
# ResNet50V2_Stack4              | per frame (max) | BalancedDistributionSVG/500/31/0.30      | (2, 0, 0)  | 0.52462276 | 0.54811824 | 0.63565891
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.51351035 | 0.49758967 | 0.64843750
# ResNet50V2_Stack4              | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.54696456 | 0.46532269 | 0.62406015
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.79471283 | 0.75738284 | 0.73684211
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.74008656 | 0.60760940 | 0.71844660
# ResNet50V2_Stack4              | per frame (sum) | SVG                                      | None       | 0.75552696 | 0.72691063 | 0.70408163
# ResNet50V2_Stack4              | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | None       | 0.76991461 | 0.75810314 | 0.70526316
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.79307521 | 0.75736299 | 0.72916667
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.70171950 | 0.57697114 | 0.71568627
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.79389402 | 0.76048767 | 0.73033708
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.71622412 | 0.58075064 | 0.70408163
# ResNet50V2_Stack4              | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.74921043 | 0.71808604 | 0.69743590
# ResNet50V2_Stack4              | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | (1, 1, 1)  | 0.76687332 | 0.75715435 | 0.69565217
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.79190549 | 0.75897569 | 0.72222222
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.69984794 | 0.56937784 | 0.69950739
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.76839396 | 0.74111973 | 0.71345029
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.68253597 | 0.55212404 | 0.67346939
# ResNet50V2_Stack4              | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.72488010 | 0.69647380 | 0.67326733
# ResNet50V2_Stack4              | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | (2, 2, 2)  | 0.74885952 | 0.73657287 | 0.67005076
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.76757515 | 0.73958322 | 0.71856287
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.67224237 | 0.54971919 | 0.66666667
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.79471283 | 0.75738284 | 0.73684211
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.74008656 | 0.60760940 | 0.71844660
# ResNet50V2_Stack4              | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.75552696 | 0.72691063 | 0.70408163
# ResNet50V2_Stack4              | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | (0, 1, 1)  | 0.76991461 | 0.75810314 | 0.70526316
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.79307521 | 0.75736299 | 0.72916667
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.70171950 | 0.57697114 | 0.71568627
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.79471283 | 0.75738284 | 0.73684211
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.74008656 | 0.60760940 | 0.71844660
# ResNet50V2_Stack4              | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.75552696 | 0.72691063 | 0.70408163
# ResNet50V2_Stack4              | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | (0, 2, 2)  | 0.76991461 | 0.75810314 | 0.70526316
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.79307521 | 0.75736299 | 0.72916667
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.70171950 | 0.57697114 | 0.71568627
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.79389402 | 0.76048767 | 0.73033708
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.71622412 | 0.58075064 | 0.70408163
# ResNet50V2_Stack4              | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.74921043 | 0.71808604 | 0.69743590
# ResNet50V2_Stack4              | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | (1, 0, 0)  | 0.76687332 | 0.75715435 | 0.69565217
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.79190549 | 0.75897569 | 0.72222222
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.69984794 | 0.56937784 | 0.69950739
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.76839396 | 0.74111973 | 0.71345029
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.68253597 | 0.55212404 | 0.67346939
# ResNet50V2_Stack4              | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.72488010 | 0.69647380 | 0.67326733
# ResNet50V2_Stack4              | per frame (sum) | BalancedDistributionSVG/500/31/0.30      | (2, 0, 0)  | 0.74885952 | 0.73657287 | 0.67005076
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.76757515 | 0.73958322 | 0.71856287
# ResNet50V2_Stack4              | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.67224237 | 0.54971919 | 0.66666667
# VGG16_Block4                   | per patch       | SpatialBin/SVG/0.20                      | None       | 0.93976845 | 0.43750499 | 0.45980707
# VGG16_Block4                   | per patch       | SpatialBin/SVG/simple_0.50               | None       | 0.94009936 | 0.44144531 | 0.48014786
# VGG16_Block4                   | per patch       | SVG                                      | None       | 0.90920492 | 0.22301287 | 0.35195438
# VGG16_Block4                   | per patch       | SpatialBin/SVG/simple_0.20               | None       | 0.93954237 | 0.44281001 | 0.48414272
# VGG16_Block4                   | per patch       | SpatialBin/SVG/0.50                      | None       | 0.93978535 | 0.42953320 | 0.45665673
# VGG16_Block4                   | per patch       | BalancedDistributionSVG/500/17/0.30      | None       | 0.89782603 | 0.21959790 | 0.33144178
# VGG16_Block4                   | per patch       | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.94175250 | 0.38444688 | 0.43782409
# VGG16_Block4                   | per patch       | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.94412085 | 0.37714693 | 0.45187694
# VGG16_Block4                   | per patch       | SVG                                      | (1, 1, 1)  | 0.90056550 | 0.19882189 | 0.34418414
# VGG16_Block4                   | per patch       | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.94172438 | 0.36509235 | 0.44554503
# VGG16_Block4                   | per patch       | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.94078449 | 0.37321086 | 0.43093599
# VGG16_Block4                   | per patch       | BalancedDistributionSVG/500/17/0.30      | (1, 1, 1)  | 0.90183221 | 0.20559150 | 0.34345766
# VGG16_Block4                   | per patch       | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.91911154 | 0.30557566 | 0.37381049
# VGG16_Block4                   | per patch       | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.92380390 | 0.28873673 | 0.38016279
# VGG16_Block4                   | per patch       | SVG                                      | (2, 2, 2)  | 0.87755219 | 0.17867221 | 0.29271527
# VGG16_Block4                   | per patch       | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.92079230 | 0.27496398 | 0.37491883
# VGG16_Block4                   | per patch       | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.91774260 | 0.29444291 | 0.36870575
# VGG16_Block4                   | per patch       | BalancedDistributionSVG/500/17/0.30      | (2, 2, 2)  | 0.87889891 | 0.18574052 | 0.29555278
# VGG16_Block4                   | per patch       | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.95453228 | 0.48491605 | 0.50777202
# VGG16_Block4                   | per patch       | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.95566189 | 0.48790100 | 0.51367635
# VGG16_Block4                   | per patch       | SVG                                      | (0, 1, 1)  | 0.91407062 | 0.22879951 | 0.36891452
# VGG16_Block4                   | per patch       | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.95425773 | 0.47525722 | 0.50641708
# VGG16_Block4                   | per patch       | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.95397660 | 0.47508462 | 0.50274302
# VGG16_Block4                   | per patch       | BalancedDistributionSVG/500/17/0.30      | (0, 1, 1)  | 0.91245070 | 0.23529504 | 0.36260031
# VGG16_Block4                   | per patch       | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.95763370 | 0.48419708 | 0.52379833
# VGG16_Block4                   | per patch       | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.95842907 | 0.47799835 | 0.51468976
# VGG16_Block4                   | per patch       | SVG                                      | (0, 2, 2)  | 0.91407337 | 0.23343875 | 0.36941192
# VGG16_Block4                   | per patch       | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.95549373 | 0.45977194 | 0.49950580
# VGG16_Block4                   | per patch       | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.95683892 | 0.47305369 | 0.51236033
# VGG16_Block4                   | per patch       | BalancedDistributionSVG/500/17/0.30      | (0, 2, 2)  | 0.91502951 | 0.24536805 | 0.36798379
# VGG16_Block4                   | per patch       | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.93284759 | 0.35761973 | 0.40600791
# VGG16_Block4                   | per patch       | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.93406755 | 0.35543268 | 0.42845850
# VGG16_Block4                   | per patch       | SVG                                      | (1, 0, 0)  | 0.89931210 | 0.19598488 | 0.33807840
# VGG16_Block4                   | per patch       | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.93263366 | 0.35132491 | 0.42919571
# VGG16_Block4                   | per patch       | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.93253090 | 0.34858520 | 0.40497851
# VGG16_Block4                   | per patch       | BalancedDistributionSVG/500/17/0.30      | (1, 0, 0)  | 0.89617842 | 0.19820109 | 0.32885844
# VGG16_Block4                   | per patch       | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.90941449 | 0.27960528 | 0.33777853
# VGG16_Block4                   | per patch       | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.91180832 | 0.27298538 | 0.35537403
# VGG16_Block4                   | per patch       | SVG                                      | (2, 0, 0)  | 0.87772027 | 0.17030275 | 0.29601599
# VGG16_Block4                   | per patch       | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.91030403 | 0.26901421 | 0.35577454
# VGG16_Block4                   | per patch       | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.90891663 | 0.27091800 | 0.33802356
# VGG16_Block4                   | per patch       | BalancedDistributionSVG/500/17/0.30      | (2, 0, 0)  | 0.87520670 | 0.17365378 | 0.29508660
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/0.20                      | None       | 0.60638671 | 0.58952428 | 0.64655172
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/simple_0.50               | None       | 0.64358404 | 0.63482118 | 0.64462810
# VGG16_Block4                   | per frame (max) | SVG                                      | None       | 0.60112294 | 0.58953541 | 0.62641509
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/simple_0.20               | None       | 0.64159551 | 0.59502790 | 0.64039409
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/0.50                      | None       | 0.62159317 | 0.58625210 | 0.64935065
# VGG16_Block4                   | per frame (max) | BalancedDistributionSVG/500/17/0.30      | None       | 0.59199906 | 0.57330611 | 0.62406015
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.56568020 | 0.57465946 | 0.64092664
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.64627442 | 0.62645921 | 0.65338645
# VGG16_Block4                   | per frame (max) | SVG                                      | (1, 1, 1)  | 0.60802433 | 0.56140998 | 0.63565891
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.61129957 | 0.57106314 | 0.64566929
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.58451281 | 0.57589129 | 0.64092664
# VGG16_Block4                   | per frame (max) | BalancedDistributionSVG/500/17/0.30      | (1, 1, 1)  | 0.59925137 | 0.54950201 | 0.63320463
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.57164581 | 0.57209084 | 0.63846154
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.63691660 | 0.62000216 | 0.65098039
# VGG16_Block4                   | per frame (max) | SVG                                      | (2, 2, 2)  | 0.63820330 | 0.58423031 | 0.67248908
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.61246929 | 0.55654277 | 0.64843750
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.59691192 | 0.58084584 | 0.63779528
# VGG16_Block4                   | per frame (max) | BalancedDistributionSVG/500/17/0.30      | (2, 2, 2)  | 0.63329044 | 0.58054030 | 0.67317073
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.61983858 | 0.60320761 | 0.64285714
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.65914142 | 0.64626522 | 0.64655172
# VGG16_Block4                   | per frame (max) | SVG                                      | (0, 1, 1)  | 0.59656100 | 0.57323928 | 0.63813230
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.64533864 | 0.60941615 | 0.63681592
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.64826295 | 0.60851036 | 0.65420561
# VGG16_Block4                   | per frame (max) | BalancedDistributionSVG/500/17/0.30      | (0, 1, 1)  | 0.58895777 | 0.56922845 | 0.63565891
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.63738449 | 0.62085395 | 0.63745020
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.66709557 | 0.65580439 | 0.65612648
# VGG16_Block4                   | per frame (max) | SVG                                      | (0, 2, 2)  | 0.59071236 | 0.57167375 | 0.64341085
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.64896479 | 0.61233711 | 0.63967611
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.64837993 | 0.61931906 | 0.64220183
# VGG16_Block4                   | per frame (max) | BalancedDistributionSVG/500/17/0.30      | (0, 2, 2)  | 0.58404492 | 0.56961126 | 0.64092664
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.53667096 | 0.56105409 | 0.63779528
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.63165282 | 0.61467326 | 0.64341085
# VGG16_Block4                   | per frame (max) | SVG                                      | (1, 0, 0)  | 0.59843257 | 0.54836853 | 0.63281250
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.60030413 | 0.54978091 | 0.63779528
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.56848754 | 0.56448325 | 0.63813230
# VGG16_Block4                   | per frame (max) | BalancedDistributionSVG/500/17/0.30      | (1, 0, 0)  | 0.58638437 | 0.52020477 | 0.63358779
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.52988654 | 0.56289522 | 0.63117871
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.60626974 | 0.58844888 | 0.63601533
# VGG16_Block4                   | per frame (max) | SVG                                      | (2, 0, 0)  | 0.63165282 | 0.50348198 | 0.66990291
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.57983390 | 0.52345109 | 0.63601533
# VGG16_Block4                   | per frame (max) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.53304480 | 0.55077674 | 0.63492063
# VGG16_Block4                   | per frame (max) | BalancedDistributionSVG/500/17/0.30      | (2, 0, 0)  | 0.61632940 | 0.49652510 | 0.66666667
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/0.20                      | None       | 0.70312317 | 0.67809623 | 0.67289720
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/simple_0.50               | None       | 0.71540531 | 0.71306772 | 0.69189189
# VGG16_Block4                   | per frame (sum) | SVG                                      | None       | 0.80173120 | 0.76399767 | 0.76404494
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/simple_0.20               | None       | 0.68733185 | 0.67884628 | 0.66666667
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/0.50                      | None       | 0.71388466 | 0.68105243 | 0.68695652
# VGG16_Block4                   | per frame (sum) | BalancedDistributionSVG/500/17/0.30      | None       | 0.79167154 | 0.75894322 | 0.74853801
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 1, 1)  | 0.66358638 | 0.64457282 | 0.65116279
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 1, 1)  | 0.68241900 | 0.66098713 | 0.65306122
# VGG16_Block4                   | per frame (sum) | SVG                                      | (1, 1, 1)  | 0.77892151 | 0.74036930 | 0.74725275
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 1, 1)  | 0.65493040 | 0.64225576 | 0.63849765
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 1, 1)  | 0.67235934 | 0.64323828 | 0.66037736
# VGG16_Block4                   | per frame (sum) | BalancedDistributionSVG/500/17/0.30      | (1, 1, 1)  | 0.76851094 | 0.73622723 | 0.73142857
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 2, 2)  | 0.63703357 | 0.62031256 | 0.64220183
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 2, 2)  | 0.66311849 | 0.63834828 | 0.63207547
# VGG16_Block4                   | per frame (sum) | SVG                                      | (2, 2, 2)  | 0.74932741 | 0.71028770 | 0.73626374
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 2, 2)  | 0.63750146 | 0.62712316 | 0.62172285
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 2, 2)  | 0.64089367 | 0.61675102 | 0.64485981
# VGG16_Block4                   | per frame (sum) | BalancedDistributionSVG/500/17/0.30      | (2, 2, 2)  | 0.74195812 | 0.70788658 | 0.71038251
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 1, 1)  | 0.70312317 | 0.67809623 | 0.67289720
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 1, 1)  | 0.71540531 | 0.71306772 | 0.69189189
# VGG16_Block4                   | per frame (sum) | SVG                                      | (0, 1, 1)  | 0.80173120 | 0.76399767 | 0.76404494
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 1, 1)  | 0.68733185 | 0.67884628 | 0.66666667
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 1, 1)  | 0.71388466 | 0.68105243 | 0.68695652
# VGG16_Block4                   | per frame (sum) | BalancedDistributionSVG/500/17/0.30      | (0, 1, 1)  | 0.79167154 | 0.75894322 | 0.74853801
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/0.20                      | (0, 2, 2)  | 0.70312317 | 0.67809623 | 0.67289720
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (0, 2, 2)  | 0.71540531 | 0.71306772 | 0.69189189
# VGG16_Block4                   | per frame (sum) | SVG                                      | (0, 2, 2)  | 0.80173120 | 0.76399767 | 0.76404494
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (0, 2, 2)  | 0.68733185 | 0.67884628 | 0.66666667
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/0.50                      | (0, 2, 2)  | 0.71388466 | 0.68105243 | 0.68695652
# VGG16_Block4                   | per frame (sum) | BalancedDistributionSVG/500/17/0.30      | (0, 2, 2)  | 0.79167154 | 0.75894322 | 0.74853801
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/0.20                      | (1, 0, 0)  | 0.66358638 | 0.64457282 | 0.65116279
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (1, 0, 0)  | 0.68241900 | 0.66098713 | 0.65306122
# VGG16_Block4                   | per frame (sum) | SVG                                      | (1, 0, 0)  | 0.77892151 | 0.74036930 | 0.74725275
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (1, 0, 0)  | 0.65493040 | 0.64225576 | 0.63849765
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/0.50                      | (1, 0, 0)  | 0.67235934 | 0.64323828 | 0.66037736
# VGG16_Block4                   | per frame (sum) | BalancedDistributionSVG/500/17/0.30      | (1, 0, 0)  | 0.76851094 | 0.73622723 | 0.73142857
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/0.20                      | (2, 0, 0)  | 0.63703357 | 0.62031256 | 0.64220183
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/simple_0.50               | (2, 0, 0)  | 0.66311849 | 0.63834828 | 0.63207547
# VGG16_Block4                   | per frame (sum) | SVG                                      | (2, 0, 0)  | 0.74932741 | 0.71028770 | 0.73626374
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/simple_0.20               | (2, 0, 0)  | 0.63750146 | 0.62712316 | 0.62172285
# VGG16_Block4                   | per frame (sum) | SpatialBin/SVG/0.50                      | (2, 0, 0)  | 0.64089367 | 0.61675102 | 0.64485981
# VGG16_Block4                   | per frame (sum) | BalancedDistributionSVG/500/17/0.30      | (2, 0, 0)  | 0.74195812 | 0.70788658 | 0.71038251