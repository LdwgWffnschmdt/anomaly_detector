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

    # files = filter(lambda f: f.endswith("MobileNetV2_Block16.h5"), files)

    if args.index is not None:
        files = files[args.index::args.total]

    with tqdm(total=len(files), file=sys.stderr) as pbar:
        metrics = list()
        for features_file in files:
            pbar.set_description(os.path.basename(features_file))
            # Check parameters
            if features_file == "" or not os.path.exists(features_file) or not os.path.isfile(features_file):
                logger.error("Specified feature file does not exist (%s)" % features_file)
                return
            
            try:
                # Load the file
                patches = PatchArray(features_file)

                # patches.calculate_patch_labels()
                # patches.calculate_tsne()

                # # Calculate and save the locations
                # if patches.contains_locations:
                #     logger.info("Locations already calculated")
                # else:
                #     patches.calculate_patch_locations()

                # patches.calculate_rasterization(0.2)
                # patches.calculate_rasterization(0.5)

                # # Calculate anomaly models

                # # threshold_learning = int(np.nanmax(patches.mahalanobis_distances["SpatialBin/SVG/0.50"]) * 0.7)

                # models = [
                #     # AnomalyModelSVG(),
                #     # AnomalyModelBalancedDistributionSVG(initial_normal_features=1000, threshold_learning=threshold_learning, pruning_parameter=0.5),
                #     AnomalyModelSpatialBinsBase(AnomalyModelSVG, cell_size=0.2),
                #     AnomalyModelSpatialBinsBase(AnomalyModelSVG, cell_size=0.5)
                #     # AnomalyModelSpatialBinsBase(lambda: AnomalyModelBalancedDistributionSVG(initial_normal_features=10, threshold_learning=threshold_learning, pruning_parameter=0.5), cell_size=0.5)
                # ]

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
                metrics.extend(patches.calculate_roc())

            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                logger.error("%s: %s" % (features_file, traceback.format_exc()))
            pbar.update()


        print("%-30s | %-7s | %-7s | %-7s" % ("NAME", "ROC_AUC", "AUC_PR", "Max. f1"))
        print("-" * 80)
        for m in metrics:
            print("%-30s |    %.2f |    %.2f |    %.2f" % m)

if __name__ == "__main__":
    calculate_locations()
    pass


# NAME                           | ROC_AUC | AUC_PR  | Max. f1
# --------------------------------------------------------------------------------
# C3D/SVG                        |    0.89 |    0.89 |    0.44
# C3D/SpatialBin/SVG/0.20        |    0.92 |    0.92 |    0.47
# C3D/SpatialBin/SVG/0.50        |    0.92 |    0.92 |    0.46
# C3D/SpatialBin/SVG/fake_0.20   |    0.86 |    0.86 |    0.33
# C3D/SpatialBin/SVG/fake_0.50   |    0.89 |    0.89 |    0.35
# EfficientNetB0_Block3/SVG      |    0.60 |    0.60 |    0.14
# EfficientNetB0_Block3/SpatialBin/SVG/0.20 |    0.66 |    0.66 |    0.14
# EfficientNetB0_Block3/SpatialBin/SVG/0.50 |    0.65 |    0.65 |    0.14
# EfficientNetB0_Block3/SpatialBin/SVG/fake_0.20 |    0.64 |    0.64 |    0.14
# EfficientNetB0_Block3/SpatialBin/SVG/fake_0.50 |    0.64 |    0.64 |    0.14
# EfficientNetB3/SVG             |    0.62 |    0.62 |    0.15
# EfficientNetB3/SpatialBin/SVG/0.20 |    0.65 |    0.65 |    0.17
# EfficientNetB3/SpatialBin/SVG/0.50 |    0.65 |    0.65 |    0.17
# EfficientNetB3/SpatialBin/SVG/fake_0.20 |    0.65 |    0.65 |    0.19
# EfficientNetB3/SpatialBin/SVG/fake_0.50 |    0.65 |    0.65 |    0.18
# EfficientNetB3_Block6/SVG      |    0.64 |    0.64 |    0.16
# EfficientNetB3_Block6/SpatialBin/SVG/0.20 |    0.66 |    0.66 |    0.17
# EfficientNetB3_Block6/SpatialBin/SVG/0.50 |    0.65 |    0.65 |    0.17
# EfficientNetB3_Block6/SpatialBin/SVG/fake_0.20 |    0.67 |    0.67 |    0.16
# EfficientNetB3_Block6/SpatialBin/SVG/fake_0.50 |    0.67 |    0.67 |    0.16
# EfficientNetB6_Block5/SVG      |    0.54 |    0.54 |    0.10
# EfficientNetB6_Block5/SpatialBin/SVG/0.20 |    0.54 |    0.54 |    0.10
# EfficientNetB6_Block5/SpatialBin/SVG/0.50 |    0.54 |    0.54 |    0.10
# EfficientNetB6_Block5/SpatialBin/SVG/fake_0.20 |    0.56 |    0.56 |    0.11
# EfficientNetB6_Block5/SpatialBin/SVG/fake_0.50 |    0.55 |    0.55 |    0.10
# MobileNetV2_Block14/SVG        |    0.83 |    0.83 |    0.24
# MobileNetV2_Block14/SpatialBin/SVG/0.20 |    0.84 |    0.84 |    0.25
# MobileNetV2_Block14/SpatialBin/SVG/0.50 |    0.84 |    0.84 |    0.25
# MobileNetV2_Block14/SpatialBin/SVG/fake_0.20 |    0.89 |    0.89 |    0.37
# MobileNetV2_Block14/SpatialBin/SVG/fake_0.50 |    0.90 |    0.90 |    0.36
# MobileNetV2_Block9/SVG         |    0.84 |    0.84 |    0.25
# MobileNetV2_Block9/SpatialBin/SVG/0.20 |    0.85 |    0.85 |    0.25
# MobileNetV2_Block9/SpatialBin/SVG/0.50 |    0.85 |    0.85 |    0.25
# MobileNetV2_Block9/SpatialBin/SVG/fake_0.20 |    0.87 |    0.87 |    0.32
# MobileNetV2_Block9/SpatialBin/SVG/fake_0.50 |    0.87 |    0.87 |    0.32
# ResNet50V2_Stack3_LargeImage/SVG |    0.89 |    0.89 |    0.31
# ResNet50V2_Stack3_LargeImage/SpatialBin/SVG/0.20 |    0.88 |    0.88 |    0.34
# ResNet50V2_Stack3_LargeImage/SpatialBin/SVG/0.50 |    0.88 |    0.88 |    0.35
# ResNet50V2_Stack3_LargeImage/SpatialBin/SVG/fake_0.20 |    0.88 |    0.88 |    0.38
# ResNet50V2_Stack3_LargeImage/SpatialBin/SVG/fake_0.50 |    0.88 |    0.88 |    0.38
# VGG16_Block3/SVG               |    0.90 |    0.90 |    0.34
# VGG16_Block3/SpatialBin/SVG/0.20 |    0.90 |    0.90 |    0.44
# VGG16_Block3/SpatialBin/SVG/0.50 |    0.90 |    0.90 |    0.44
# VGG16_Block3/SpatialBin/SVG/fake_0.20 |    0.90 |    0.90 |    0.43
# VGG16_Block3/SpatialBin/SVG/fake_0.50 |    0.90 |    0.90 |    0.46

# C3D_Block3/SVG                 |    0.85 |    0.85 |    0.30
# C3D_Block3/SpatialBin/SVG/0.20 |    0.90 |    0.90 |    0.45
# C3D_Block3/SpatialBin/SVG/0.50 |    0.89 |    0.89 |    0.45
# C3D_Block3/SpatialBin/SVG/fake_0.20 |    0.90 |    0.90 |    0.43
# C3D_Block3/SpatialBin/SVG/fake_0.50 |    0.90 |    0.90 |    0.45
# EfficientNetB0_Block4/SVG      |    0.51 |    0.51 |    0.10
# EfficientNetB0_Block4/SpatialBin/SVG/0.20 |    0.59 |    0.59 |    0.12
# EfficientNetB0_Block4/SpatialBin/SVG/0.50 |    0.59 |    0.59 |    0.12
# EfficientNetB0_Block4/SpatialBin/SVG/fake_0.20 |    0.62 |    0.62 |    0.13
# EfficientNetB0_Block4/SpatialBin/SVG/fake_0.50 |    0.61 |    0.61 |    0.12
# EfficientNetB3_Block3/SVG      |    0.63 |    0.63 |    0.14
# EfficientNetB3_Block3/SpatialBin/SVG/0.20 |    0.57 |    0.57 |    0.10
# EfficientNetB3_Block3/SpatialBin/SVG/0.50 |    0.57 |    0.57 |    0.10
# EfficientNetB3_Block3/SpatialBin/SVG/fake_0.20 |    0.57 |    0.57 |    0.11
# EfficientNetB3_Block3/SpatialBin/SVG/fake_0.50 |    0.55 |    0.55 |    0.10
# EfficientNetB6/SVG             |    0.59 |    0.59 |    0.19
# EfficientNetB6/SpatialBin/SVG/0.20 |    0.57 |    0.57 |    0.14
# EfficientNetB6/SpatialBin/SVG/0.50 |    0.57 |    0.57 |    0.14
# EfficientNetB6/SpatialBin/SVG/fake_0.20 |    0.62 |    0.62 |    0.18
# EfficientNetB6/SpatialBin/SVG/fake_0.50 |    0.62 |    0.62 |    0.17
# EfficientNetB6_Block6/SVG      |    0.61 |    0.61 |    0.19
# EfficientNetB6_Block6/SpatialBin/SVG/0.20 |    0.61 |    0.61 |    0.18
# EfficientNetB6_Block6/SpatialBin/SVG/0.50 |    0.60 |    0.60 |    0.18
# EfficientNetB6_Block6/SpatialBin/SVG/fake_0.20 |    0.65 |    0.65 |    0.18
# EfficientNetB6_Block6/SpatialBin/SVG/fake_0.50 |    0.64 |    0.64 |    0.17
# MobileNetV2_Block16/SVG        |    0.89 |    0.89 |    0.35
# MobileNetV2_Block16/SpatialBin/SVG/0.20 |    0.91 |    0.91 |    0.41
# MobileNetV2_Block16/SpatialBin/SVG/0.50 |    0.90 |    0.90 |    0.41
# MobileNetV2_Block16/SpatialBin/SVG/fake_0.20 |    0.92 |    0.92 |    0.52
# MobileNetV2_Block16/SpatialBin/SVG/fake_0.50 |    0.94 |    0.94 |    0.53
# ResNet50V2/SVG                 |    0.91 |    0.91 |    0.47
# ResNet50V2/SpatialBin/SVG/0.20 |    0.93 |    0.93 |    0.48
# ResNet50V2/SpatialBin/SVG/0.50 |    0.93 |    0.93 |    0.47
# ResNet50V2/SpatialBin/SVG/fake_0.20 |    0.85 |    0.85 |    0.30
# ResNet50V2/SpatialBin/SVG/fake_0.50 |    0.87 |    0.87 |    0.32
# ResNet50V2_Stack4/SVG          |    0.86 |    0.86 |    0.26
# ResNet50V2_Stack4/SpatialBin/SVG/0.20 |    0.86 |    0.86 |    0.26
# ResNet50V2_Stack4/SpatialBin/SVG/0.50 |    0.86 |    0.86 |    0.26
# ResNet50V2_Stack4/SpatialBin/SVG/fake_0.20 |    0.85 |    0.85 |    0.31
# ResNet50V2_Stack4/SpatialBin/SVG/fake_0.50 |    0.86 |    0.86 |    0.32
# VGG16_Block4/SVG               |    0.91 |    0.91 |    0.35
# VGG16_Block4/SpatialBin/SVG/0.20 |    0.93 |    0.93 |    0.46
# VGG16_Block4/SpatialBin/SVG/0.50 |    0.93 |    0.93 |    0.47
# VGG16_Block4/SpatialBin/SVG/fake_0.20 |    0.91 |    0.91 |    0.42
# VGG16_Block4/SpatialBin/SVG/fake_0.50 |    0.93 |    0.93 |    0.45

# C3D_Block4/SVG                 |    0.88 |    0.88 |    0.36
# C3D_Block4/SpatialBin/SVG/0.20 |    0.91 |    0.91 |    0.44
# C3D_Block4/SpatialBin/SVG/0.50 |    0.91 |    0.91 |    0.46
# C3D_Block4/SpatialBin/SVG/fake_0.20 |    0.88 |    0.88 |    0.35
# C3D_Block4/SpatialBin/SVG/fake_0.50 |    0.90 |    0.90 |    0.39
# EfficientNetB0_Block5/SVG      |    0.54 |    0.54 |    0.11
# EfficientNetB0_Block5/SpatialBin/SVG/0.20 |    0.55 |    0.55 |    0.11
# EfficientNetB0_Block5/SpatialBin/SVG/0.50 |    0.55 |    0.55 |    0.11
# EfficientNetB0_Block5/SpatialBin/SVG/fake_0.20 |    0.62 |    0.62 |    0.12
# EfficientNetB0_Block5/SpatialBin/SVG/fake_0.50 |    0.61 |    0.61 |    0.12
# EfficientNetB3_Block4/SVG      |    0.56 |    0.56 |    0.15
# EfficientNetB3_Block4/SpatialBin/SVG/0.20 |    0.58 |    0.58 |    0.12
# EfficientNetB3_Block4/SpatialBin/SVG/0.50 |    0.58 |    0.58 |    0.12
# EfficientNetB3_Block4/SpatialBin/SVG/fake_0.20 |    0.59 |    0.59 |    0.11
# EfficientNetB3_Block4/SpatialBin/SVG/fake_0.50 |    0.57 |    0.57 |    0.11
# EfficientNetB6_Block3/SVG      |    0.55 |    0.55 |    0.13
# EfficientNetB6_Block3/SpatialBin/SVG/0.20 |    0.56 |    0.56 |    0.11
# EfficientNetB6_Block3/SpatialBin/SVG/0.50 |    0.56 |    0.56 |    0.11
# EfficientNetB6_Block3/SpatialBin/SVG/fake_0.20 |    0.55 |    0.55 |    0.10
# EfficientNetB6_Block3/SpatialBin/SVG/fake_0.50 |    0.52 |    0.52 |    0.10
# MobileNetV2/SVG                |    0.88 |    0.88 |    0.33
# MobileNetV2/SpatialBin/SVG/0.20 |    0.94 |    0.94 |    0.50
# MobileNetV2/SpatialBin/SVG/0.50 |    0.94 |    0.94 |    0.49
# MobileNetV2/SpatialBin/SVG/fake_0.20 |    0.91 |    0.91 |    0.43
# MobileNetV2/SpatialBin/SVG/fake_0.50 |    0.92 |    0.92 |    0.43
# MobileNetV2_Block3/SVG         |    0.86 |    0.86 |    0.28
# MobileNetV2_Block3/SpatialBin/SVG/0.20 |    0.84 |    0.84 |    0.32
# MobileNetV2_Block3/SpatialBin/SVG/0.50 |    0.85 |    0.85 |    0.33
# MobileNetV2_Block3/SpatialBin/SVG/fake_0.20 |    0.85 |    0.85 |    0.34
# MobileNetV2_Block3/SpatialBin/SVG/fake_0.50 |    0.85 |    0.85 |    0.35
# ResNet50V2_LargeImage/SVG      |    0.91 |    0.91 |    0.43
# ResNet50V2_LargeImage/SpatialBin/SVG/0.20 |    0.95 |    0.95 |    0.54
# ResNet50V2_LargeImage/SpatialBin/SVG/0.50 |    0.94 |    0.94 |    0.53
# ResNet50V2_LargeImage/SpatialBin/SVG/fake_0.20 |    0.87 |    0.87 |    0.32
# ResNet50V2_LargeImage/SpatialBin/SVG/fake_0.50 |    0.90 |    0.90 |    0.37
# ResNet50V2_Stack4_LargeImage/SVG |    0.88 |    0.88 |    0.29
# ResNet50V2_Stack4_LargeImage/SpatialBin/SVG/0.20 |    0.86 |    0.86 |    0.28
# ResNet50V2_Stack4_LargeImage/SpatialBin/SVG/0.50 |    0.87 |    0.87 |    0.29
# ResNet50V2_Stack4_LargeImage/SpatialBin/SVG/fake_0.20 |    0.86 |    0.86 |    0.33
# ResNet50V2_Stack4_LargeImage/SpatialBin/SVG/fake_0.50 |    0.86 |    0.86 |    0.33

# EfficientNetB0/SVG             |    0.66 |    0.66 |    0.27
# EfficientNetB0/SpatialBin/SVG/0.20 |    0.67 |    0.67 |    0.28
# EfficientNetB0/SpatialBin/SVG/0.50 |    0.67 |    0.67 |    0.28
# EfficientNetB0/SpatialBin/SVG/fake_0.20 |    0.74 |    0.74 |    0.30
# EfficientNetB0/SpatialBin/SVG/fake_0.50 |    0.73 |    0.73 |    0.31
# EfficientNetB0_Block6/SVG      |    0.65 |    0.65 |    0.26
# EfficientNetB0_Block6/SpatialBin/SVG/0.20 |    0.66 |    0.66 |    0.26
# EfficientNetB0_Block6/SpatialBin/SVG/0.50 |    0.66 |    0.66 |    0.26
# EfficientNetB0_Block6/SpatialBin/SVG/fake_0.20 |    0.78 |    0.78 |    0.30
# EfficientNetB0_Block6/SpatialBin/SVG/fake_0.50 |    0.78 |    0.78 |    0.31
# EfficientNetB3_Block5/SVG      |    0.60 |    0.60 |    0.14
# EfficientNetB3_Block5/SpatialBin/SVG/0.20 |    0.60 |    0.60 |    0.14
# EfficientNetB3_Block5/SpatialBin/SVG/0.50 |    0.60 |    0.60 |    0.14
# EfficientNetB3_Block5/SpatialBin/SVG/fake_0.20 |    0.61 |    0.61 |    0.12
# EfficientNetB3_Block5/SpatialBin/SVG/fake_0.50 |    0.59 |    0.59 |    0.11
# EfficientNetB6_Block4/SVG                             |    0.56 |    0.56 |    0.11
# EfficientNetB6_Block4/SpatialBin/SVG/0.20             |    0.57 |    0.57 |    0.12
# EfficientNetB6_Block4/SpatialBin/SVG/0.50             |    0.58 |    0.58 |    0.12
# EfficientNetB6_Block4/SpatialBin/SVG/fake_0.20        |    0.57 |    0.57 |    0.11
# EfficientNetB6_Block4/SpatialBin/SVG/fake_0.50        |    0.55 |    0.55 |    0.10
# MobileNetV2_Block12/SVG                               |    0.86 |    0.86 |    0.27
# MobileNetV2_Block12/SpatialBin/SVG/0.20               |    0.87 |    0.87 |    0.27
# MobileNetV2_Block12/SpatialBin/SVG/0.50               |    0.87 |    0.87 |    0.27
# MobileNetV2_Block12/SpatialBin/SVG/fake_0.20          |    0.87 |    0.87 |    0.35
# MobileNetV2_Block12/SpatialBin/SVG/fake_0.50          |    0.87 |    0.87 |    0.35
# MobileNetV2_Block6/SVG                                |    0.87 |    0.87 |    0.30
# MobileNetV2_Block6/SpatialBin/SVG/0.20                |    0.87 |    0.87 |    0.29
# MobileNetV2_Block6/SpatialBin/SVG/0.50                |    0.88 |    0.88 |    0.30
# MobileNetV2_Block6/SpatialBin/SVG/fake_0.20           |    0.89 |    0.89 |    0.37
# MobileNetV2_Block6/SpatialBin/SVG/fake_0.50           |    0.89 |    0.89 |    0.37
# ResNet50V2_Stack3/SVG                                 |    0.90 |    0.90 |    0.33
# ResNet50V2_Stack3/SpatialBin/SVG/0.20                 |    0.90 |    0.90 |    0.34
# ResNet50V2_Stack3/SpatialBin/SVG/0.50                 |    0.91 |    0.91 |    0.35
# ResNet50V2_Stack3/SpatialBin/SVG/fake_0.20            |    0.89 |    0.89 |    0.38
# ResNet50V2_Stack3/SpatialBin/SVG/fake_0.50            |    0.90 |    0.90 |    0.39
# VGG16/SVG                                             |    0.94 |    0.94 |    0.48
# VGG16/SpatialBin/SVG/0.20                             |    0.96 |    0.96 |    0.57
# VGG16/SpatialBin/SVG/0.50                             |    0.96 |    0.96 |    0.57
# VGG16/SpatialBin/SVG/fake_0.20                        |    0.90 |    0.90 |    0.36
# VGG16/SpatialBin/SVG/fake_0.50                        |    0.92 |    0.92 |    0.40
