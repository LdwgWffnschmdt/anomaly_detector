#! /usr/bin/env python
# -*- coding: utf-8 -*-

import consts
import argparse

parser = argparse.ArgumentParser(description="Remove duplicate images and their metadata files based on md5 checksum and size.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("images", metavar="F", type=str, default=consts.IMAGES_PATH,
                    help="Path to images (default: %s)" % consts.IMAGES_PATH)

args = parser.parse_args()

import os
import sys
import md5
from glob import glob
from tqdm import tqdm

from common import logger

def getmd5(filename):
    m = md5.new()
    m.update(file(filename, 'rb').read(-1))
    return m.hexdigest()

def remove_duplicates():
    # Check parameters
    if args.images == "" or not os.path.exists(args.images) or not os.path.isdir(args.images):
        logger.error("Specified path does not exist (%s)" % args.images)
        return

    # Get all images
    files = sorted(glob(os.path.join(args.images, "*.jpg")))

    # Checksums
    last_checksum = None
    duplicates = list()

    with tqdm(desc="Get MD5 checksums", total=len(files), file=sys.stderr) as pbar:
        for f in files:
            time = int(os.path.splitext(os.path.basename(f))[0])
            check = getmd5(f)
            if last_checksum == check:
                duplicates.append(time)
            else:
                last_checksum = check
            pbar.set_postfix({"Duplicates": len(duplicates)})
            pbar.update()

    if len(duplicates) > 0:
        output_dir = os.path.join(args.images, "Duplicates")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for d in tqdm(duplicates, desc="Moving duplicates", file=sys.stderr):
            os.rename(os.path.join(args.images, "%i.jpg" % d), os.path.join(output_dir, "%i.jpg" % d))
            os.rename(os.path.join(args.images, "%i.yml" % d), os.path.join(output_dir, "%i.yml" % d))

        

if __name__ == "__main__":
    remove_duplicates()
    pass