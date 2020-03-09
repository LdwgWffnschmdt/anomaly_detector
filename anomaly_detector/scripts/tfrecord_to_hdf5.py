#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser(description="Extract images from tfrecords and store them in an HDF5 file.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("files", metavar="F", type=str, nargs='*',
                    help="The feature file(s). Supports \"path/to/*.tfrecord\"")

parser.add_argument("--output", metavar="OUT", dest="output_file", type=str,
                    help="Output HDF5 file (default: \"{files}/Images/images.h5\")")

parser.add_argument("--compression", metavar="C", dest="compression", type=str,
                    default="lzf",
                    help="Output file compression (default: lzf)")

parser.add_argument("--compression_opts", metavar="CO", dest="compression_opts", type=str,
                    default=None,
                    help="Output file compression options (default: None)")

args = parser.parse_args()

import os
import time
import logging
import traceback
from glob import glob

import cv2
import h5py
import numpy as np

import common.utils as utils

def tfrecord_to_hdf5():
    files = args.files

    # Expand wildcards
    files_expanded = []
    for s in files:
        files_expanded += glob(s)
    files = list(set(files_expanded)) # Remove duplicates

    # Check parameters
    if not files or len(files) < 1 or files[0] == "":
        logging.error("No input file specified.")
        return
    
    if args.output_file is None or args.output_file == "":
        output_dir = os.path.join(os.path.abspath(os.path.dirname(files[0])), "Images")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        args.output_file = os.path.join(output_dir, "Images.h5")
        logging.info("Output file set to %s" % args.output_file)

    logging.info("Loading dataset")
    parsed_dataset = utils.load_tfrecords(files)

    # Get number of examples in dataset
    total = sum(1 for record in parsed_dataset)

    # IO stuff
    h5Writer = h5py.File(args.output_file, "a")
    
    images_dataset = h5Writer.create_dataset("images",
                                                shape=(total,),
                                                dtype=h5py.string_dtype(),
                                                compression=args.compression,
                                                compression_opts=args.compression_opts)

    # Add metadata to the output file
    images_dataset.attrs["Files"]                     = files
    images_dataset.attrs["Compression"]               = args.compression
    if args.compression_opts is not None:
        images_dataset.attrs["Compression options"]   = args.compression_opts

    computer_info = utils.getComputerInfo()
    for key, value in computer_info.items():
        images_dataset.attrs[key] = value
    images_dataset.attrs["Start"] = time.time()

    try:
        # Add features to list
        for i, image, example in enumerate(tqdm(parsed_dataset, desc="Extracting images", total=total)):
            images_dataset[i] = cv2.imencode(".jpg", image.numpy())[1].tostring()
    except:
        exc = traceback.format_exc()
        logging.error(exc)
        h5Writer.attrs["Exception"] = exc
        return False
    finally:
        end = time.time()
        h5Writer.attrs["End"] = end
        h5Writer.attrs["Duration"] = end - start
        h5Writer.attrs["Duration (formatted)"] = utils.format_duration(end - start)
        h5Writer.attrs["Number of frames extracted"] = counter
        h5Writer.attrs["Number of total frames"] = total
        h5Writer.close()


if __name__ == "__main__":
    tfrecord_to_hdf5()