#! /usr/bin/env python
# -*- coding: utf-8 -*-

import consts
import argparse

parser = argparse.ArgumentParser(description="Extract features from images.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--list", dest="list", action="store_true",
                    help="List all extractors and exit")

parser.add_argument("--files", metavar="F", dest="files", type=str, nargs='*', default=consts.EXTRACT_FILES,
                    help="File(s) to use (*.jpg)")

parser.add_argument("--extractor", metavar="EXT", dest="extractor", nargs='*', type=str,
                    help="Extractor name. Leave empty for all extractors (default: \"\")")

args = parser.parse_args()

import os
import sys
import time
from tqdm import tqdm
from common import utils, logger, PatchArray, Visualize
import traceback

import numpy as np

def extract_features():
    if not args.list and len(args.files) == 0:
        logger.error("No input file specified.")
        return

    import tensorflow as tf
    import inspect
    import feature_extractor as feature_extractor

    # Add before any TF calls (https://github.com/tensorflow/tensorflow/issues/29931#issuecomment-504217770)
    # Initialize the keras global outside of any tf.functions
    temp = tf.zeros([4, 32, 32, 3])
    tf.keras.applications.vgg16.preprocess_input(temp)

    # Get all the available feature extractor names
    extractor_names = list([e[0] for e in inspect.getmembers(feature_extractor, inspect.isclass) if e[0] != "FeatureExtractorBase"])

    module = __import__("feature_extractor")

    if args.list:
        print("%-30s | %-15s | %-4s | %-8s | %-5s" % ("NAME", "OUTPUT SHAPE", "RF", "IMG SIZE", "RF / IMG"))
        print("-" * 80)
        for e in list(map(lambda e: getattr(module, e), extractor_names)):
            factor = e.RECEPTIVE_FIELD["size"][0] / float(e.IMG_SIZE)
            print("%-30s | %-15s | %-4s | %-8s | %.3f %s" % (e.__name__.replace("FeatureExtractor", ""), e.OUTPUT_SHAPE, e.RECEPTIVE_FIELD["size"][0], e.IMG_SIZE, factor, "!" if factor >= 2 else ""))
        return

    if args.extractor is None:
        args.extractor = extractor_names

    # args.extractor = filter(lambda f: "EfficientNet" in f, args.extractor)

    if isinstance(args.files, basestring):
        args.files = [args.files]

    patches = PatchArray(args.files)


    ## WZL:
    patches = patches.training_and_validation
    # For the benchmark subset:
    # patches = patches.training_and_validation[0:10]

    ## FieldSAFE:
    # p = patches[:, 0, 0]
    # f = p.round_numbers == 1
    # patches = patches[f]

    # vis = Visualize(patches)
    # vis.show()

    dataset = patches.to_dataset()
    dataset_3D = patches.to_temporal_dataset(16)
    total = patches.shape[0]

    # Add progress bar if multiple extractors
    if len(args.extractor) > 1:
        args.extractor = tqdm(args.extractor, desc="Extractors", file=sys.stderr)

    for extractor_name in args.extractor:
        try:
            bs = getattr(module, extractor_name).TEMPORAL_BATCH_SIZE
            # shape = getattr(module, extractor_name).OUTPUT_SHAPE
            # if np.prod(shape) > 300000:
            #     logger.warning("Skipping %s (output too big)" % extractor_name)
            #     continue

            logger.info("Instantiating %s" % extractor_name)
            extractor = getattr(module, extractor_name)()
            # Get an instance
            if bs > 1:
                extractor.extract_dataset(dataset_3D, total)
            else:
                extractor.extract_dataset(dataset, total)
        except KeyboardInterrupt:
            logger.info("Terminated by CTRL-C")
            return
        except:
            logger.error("%s: %s" % (extractor_name, traceback.format_exc()))

if __name__ == "__main__":
    extract_features()

# NAME                           | OUTPUT SHAPE    | RF   | IMG SIZE | RF / IMG
# --------------------------------------------------------------------------------
# C3D_Block3                     | (28, 28, 256)   | 23   | 112      | 0.205 
# C3D_Block4                     | (14, 14, 512)   | 55   | 112      | 0.491 
# C3D_Block5                     | (7, 7, 512)     | 119  | 112      | 1.062 
# EfficientNetB0_Level6          | (14, 14, 112)   | 339  | 224      | 1.513 
# EfficientNetB0_Level7          | (7, 7, 192)     | 787  | 224      | 3.513 !
# EfficientNetB0_Level8          | (7, 7, 320)     | 819  | 224      | 3.656 !
# EfficientNetB0_Level9          | (7, 7, 1280)    | 851  | 224      | 3.799 !
# EfficientNetB3_Level6          | (19, 19, 136)   | 575  | 300      | 1.917 
# EfficientNetB3_Level7          | (10, 10, 232)   | 1200 | 300      | 4.000 !
# EfficientNetB3_Level8          | (10, 10, 384)   | 1200 | 300      | 4.000 !
# EfficientNetB3_Level9          | (10, 10, 1536)  | 1200 | 300      | 4.000 !
# EfficientNetB6_Level6          | (33, 33, 200)   | 987  | 528      | 1.869 
# EfficientNetB6_Level7          | (17, 17, 344)   | 1056 | 528      | 2.000 !
# EfficientNetB6_Level8          | (17, 17, 576)   | 1056 | 528      | 2.000 !
# EfficientNetB6_Level9          | (17, 17, 2304)  | 1056 | 528      | 2.000 !
# MobileNetV2_Block03            | (28, 28, 32)    | 27   | 224      | 0.121 
# MobileNetV2_Block06            | (14, 14, 64)    | 75   | 224      | 0.335 
# MobileNetV2_Block09            | (14, 14, 64)    | 171  | 224      | 0.763 
# MobileNetV2_Block12            | (14, 14, 96)    | 267  | 224      | 1.192 
# MobileNetV2_Block14            | (7, 7, 160)     | 363  | 224      | 1.621 
# MobileNetV2_Block16            | (7, 7, 320)     | 491  | 224      | 2.192 !
# MobileNetV2_Last               | (7, 7, 1280)    | 491  | 224      | 2.192 !
# ResNet50V2_Block3              | (14, 14, 512)   | 95   | 224      | 0.424 
# ResNet50V2_Block4              | (7, 7, 1024)    | 287  | 224      | 1.281 
# ResNet50V2_Block5              | (7, 7, 2048)    | 479  | 224      | 2.138 !
# ResNet50V2_LargeImage_Block3   | (29, 29, 512)   | 95   | 449      | 0.212 
# ResNet50V2_LargeImage_Block4   | (15, 15, 1024)  | 287  | 449      | 0.639 
# ResNet50V2_LargeImage_Block5   | (15, 15, 2048)  | 479  | 449      | 1.067 
# VGG16_Block3                   | (56, 56, 512)   | 37   | 224      | 0.165 
# VGG16_Block4                   | (28, 28, 512)   | 85   | 224      | 0.379
# VGG16_Block5                   | (14, 14, 512)   | 181  | 224      | 0.808 