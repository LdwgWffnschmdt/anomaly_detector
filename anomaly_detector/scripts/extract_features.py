#! /usr/bin/env python
# -*- coding: utf-8 -*-

import consts
import argparse

parser = argparse.ArgumentParser(description="Extract features from tfrecords.",
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

    if isinstance(args.files, basestring):
        args.files = [args.files]

    patches = PatchArray(args.files)

    p = patches[:, 0, 0]

    ## WZL:
    # f = np.zeros(p.shape, dtype=np.bool)
    # f[:] = np.logical_and(p.directions == 1,                                   # CCW and
    #                         np.logical_or(p.labels == 2,                         #   Anomaly or
    #                                     np.logical_and(p.round_numbers >= 7,   #     Round between 2 and 5
    #                                                     p.round_numbers <= 9)))

    # # Let's make contiguous blocks of at least 10, so
    # # we can do some meaningful temporal smoothing afterwards
    # for i, b in enumerate(f):
    #     if b and i - 10 >= 0:
    #         f[i - 10:i] = True

    patches = patches.training_and_validation



    ## FieldSAFE:
    # f = p.round_numbers == 1
    # patches = patches[f]

    # vis = Visualize(patches)
    # vis.show()

    # patches = getattr(patches, args.filter, None)
    # assert patches is not None, "The filter was not valid."
    # if args.filter_argument is not None:
    #     patches = patches(args.filter_argument)
    #     assert patches is not None, "The filter argument was not valid."
    dataset = patches.to_dataset()
    dataset_3D = patches.to_temporal_dataset(16)
    total = patches.shape[0]

    # Add progress bar if multiple extractors
    if len(args.extractor) > 1:
        args.extractor = tqdm(args.extractor, desc="Extractors", file=sys.stderr)

    for extractor_name in args.extractor:
        try:
            bs = getattr(module, extractor_name).TEMPORAL_BATCH_SIZE
            shape = getattr(module, extractor_name).OUTPUT_SHAPE
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

# | NAME                           | OUTPUT SHAPE    | RF   | IMG SIZE | RF / IMG |
# |--------------------------------|-----------------|------|----------|----------|
# | C3D                            | (7, 7, 512)     | 119  | 112      | 1.062    |
# | C3D_Block3                     | (28, 28, 256)   | 23   | 112      | 0.205    |
# | C3D_Block4                     | (14, 14, 512)   | 55   | 112      | 0.491    |
# | EfficientNetB0                 | (7, 7, 1280)    | 851  | 224      | 3.799 !  |
# | EfficientNetB0_Block3          | (28, 28, 40)    | 67   | 224      | 0.299    |
# | EfficientNetB0_Block4          | (14, 14, 80)    | 147  | 224      | 0.656    |
# | EfficientNetB0_Block5          | (14, 14, 112)   | 339  | 224      | 1.513    |
# | EfficientNetB0_Block6          | (7, 7, 192)     | 787  | 224      | 3.513 !  |
# | EfficientNetB3                 | (10, 10, 1536)  | 1200 | 300      | 4.000 !  |
# | EfficientNetB3_Block3          | (38, 38, 48)    | 111  | 300      | 0.370    |
# | EfficientNetB3_Block4          | (19, 19, 96)    | 255  | 300      | 0.850    |
# | EfficientNetB3_Block5          | (19, 19, 136)   | 575  | 300      | 1.917    |
# | EfficientNetB3_Block6          | (10, 10, 232)   | 1200 | 300      | 4.000 !  |
# | EfficientNetB6                 | (17, 17, 2304)  | 1056 | 528      | 2.000 !  |
# | EfficientNetB6_Block3          | (66, 66, 72)    | 235  | 528      | 0.445    |
# | EfficientNetB6_Block4          | (33, 33, 144)   | 475  | 528      | 0.900    |
# | EfficientNetB6_Block5          | (33, 33, 200)   | 987  | 528      | 1.869    |
# | EfficientNetB6_Block6          | (17, 17, 344)   | 1056 | 528      | 2.000 !  |
# | MobileNetV2                    | (7, 7, 1280)    | 491  | 224      | 2.192 !  |
# | MobileNetV2_Block12            | (14, 14, 96)    | 267  | 224      | 1.192    |
# | MobileNetV2_Block14            | (7, 7, 160)     | 363  | 224      | 1.621    |
# | MobileNetV2_Block16            | (7, 7, 320)     | 491  | 224      | 2.192 !  |
# | MobileNetV2_Block3             | (28, 28, 32)    | 27   | 224      | 0.121    |
# | MobileNetV2_Block6             | (14, 14, 64)    | 75   | 224      | 0.335    |
# | MobileNetV2_Block9             | (14, 14, 64)    | 171  | 224      | 0.763    |
# | ResNet50V2                     | (7, 7, 2048)    | 479  | 224      | 2.138 !  |
# | ResNet50V2_LargeImage          | (15, 15, 2048)  | 479  | 449      | 1.067    |
# | ResNet50V2_Stack3              | (14, 14, 512)   | 95   | 224      | 0.424    |
# | ResNet50V2_Stack3_LargeImage   | (29, 29, 512)   | 95   | 449      | 0.212    |
# | ResNet50V2_Stack4              | (7, 7, 1024)    | 287  | 224      | 1.281    |
# | ResNet50V2_Stack4_LargeImage   | (15, 15, 1024)  | 287  | 449      | 0.639    |
# | VGG16                          | (14, 14, 512)   | 181  | 224      | 0.808    |
# | VGG16_Block3                   | (56, 56, 512)   | 37   | 224      | 0.165    |
# | VGG16_Block4                   | (28, 28, 512)   | 85   | 224      | 0.379    |