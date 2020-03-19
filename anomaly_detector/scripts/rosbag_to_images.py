#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser(description="Convert bag files to images with metadata files.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("bag_files", metavar="F", type=str, nargs='+',
                    help="The bag file(s) to convert. Supports \"path/to/*.bag\"")

parser.add_argument("--output_dir", metavar="OUT", dest="output_dir", type=str,
                    help="Output directory (default: {bag_file}/Images)")

parser.add_argument("--image_topic", metavar="IM", dest="image_topic", type=str,
                    default="/camera/color/image_raw",
                    help="Image topic (default: \"/camera/color/image_raw\")")

parser.add_argument("--image_crop", metavar=("X", "Y", "W", "H"), type=int, nargs=4,
                    help="Crop images using. Cropping is applied before scaling (default: Complete image)")

parser.add_argument("--image_scale", metavar="SCALE", type=float,
                    default=1.0,
                    help="Scale images by this factor (default: 1.0)")

parser.add_argument("--tf_map", metavar="TF_M", dest="tf_map", type=str,
                    default="map",
                    help="TF reference frame (default: map)")

parser.add_argument("--tf_base_link", metavar="TF_B", dest="tf_base_link", type=str,
                    default="realsense_link",
                    help="TF camera frame (default: base_link)")

parser.add_argument("--label", metavar="L", dest="label", type=int,
                    default=0,
                    help="-2: labeling mode (show image and wait for input) [space]: No anomaly\n"
                         "                                                  [tab]  : Contains anomaly\n"
                         "-1: continuous labeling mode (show image for 10ms, keep label until change)\n"
                         " 0: Unknown (default)\n"
                         " 1: No anomaly\n"
                         " 2: Contains an anomaly")

args = parser.parse_args()

import os
import sys
import time
from glob import glob
import yaml

import rospy
import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError
import tf as ros_tf
import tf2_ros
import tf2_py as tf2
import numpy as np
from tqdm import tqdm

from common import Visualize, utils, logger

def rosbag_to_images():
    ################
    #  Parameters  #
    ################
    bag_files      = args.bag_files
    output_dir     = args.output_dir
    image_topic    = args.image_topic
    tf_map         = args.tf_map
    tf_base_link   = args.tf_base_link
    label          = args.label

    # Check parameters
    if not bag_files or len(bag_files) < 1 or bag_files[0] == "":
        raise ValueError("Please specify at least one filename (%s)" % bag_files)
    
    # Expand wildcards
    bag_files_expanded = []
    for s in bag_files:
        bag_files_expanded += glob(s)
    bag_files = list(set(bag_files_expanded)) # Remove duplicates

    if output_dir is None or output_dir == "" or not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        output_dir = os.path.join(os.path.abspath(os.path.dirname(bag_files[0])), "Images")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger.info("Output directory set to %s" % output_dir)

    if image_topic is None or image_topic == "":
        logger.error("No image topic given. Use parameter image_topic.")
        return

    if tf_map is None or tf_map == "" or tf_base_link == "":
        logger.error("Please specify tf frame names.")
        return

    if -2 > label or label > 2:
        logger.error("label has to be between -2 and 2.")
        return

    # Add progress bar if multiple files
    if len(bag_files > 1):
        bag_files = tqdm(bag_files, desc="Bag files", file=sys.stderr)

    for bag_file in bag_files:
        # Check parameters
        if bag_file == "" or not os.path.exists(bag_file) or not os.path.isfile(bag_file):
            logger.error("Specified bag does not exist (%s)" % bag_file)
            continue

        logger.info("Extracting %s" % bag_file)

        bag_file_name = os.path.splitext(os.path.basename(bag_file))[0]

        def get_label(image, last_label, auto_duration=10):
            if label < 0: # Labeling mode (show image and wait for input)
                image_cp = image.copy()

                if label == -1 and not last_label is None:
                    Visualize.image_write_label(image_cp, last_label)

                cv2.imshow("Label image | [1]: No anomaly, [2]: Contains anomaly, [0]: Unknown", image_cp)
                key = cv2.waitKey(0 if label == -2 or last_label == None else auto_duration)
                
                if key == 27:   # [esc] => Quit
                    return None
                elif key == 48: # [0]   => Unknown
                    return 0
                elif key == 49: # [1]   => No anomaly
                    return 1
                elif key == 50: # [2]   => Contains anomaly
                    return 2
                elif key == -1 and label == -1 and not last_label is None:
                    return last_label
                else:
                    return get_label(image, None)
            else:
                return label

        # Used to convert image message to opencv image
        bridge = CvBridge()
        
        ################
        #     MAIN     #
        ################
        with rosbag.Bag(bag_file, "r") as bag:
            ### Get /tf transforms
            expected_tf_count = bag.get_message_count(["/tf", "/tf_static"])
            
            tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(bag.get_end_time() - bag.get_start_time()), debug=False)
            
            for topic, msg, t in tqdm(bag.read_messages(topics=["/tf", "/tf_static"]),
                                        desc="Extracting transforms",
                                        total=expected_tf_count,
                                        file=sys.stderr):
                for msg_tf in msg.transforms:
                    if topic == "/tf_static":
                        tf_buffer.set_transform_static(msg_tf, "default_authority")
                    else:
                        tf_buffer.set_transform(msg_tf, "default_authority")

            ### Get images
            expected_im_count = bag.get_message_count(image_topic)

            skipped_count = 0       

            image_label = None

            with tqdm(desc="Writing images",
                        total=expected_im_count,
                        file=sys.stderr) as pbar:
                for topic, msg, t in bag.read_messages(topics=image_topic):
                    try:
                        # Get translation and orientation
                        msg_tf = tf_buffer.lookup_transform(tf_map, tf_base_link, t)#, rospy.Duration.from_sec(0.001))
                        translation = msg_tf.transform.translation
                        euler = ros_tf.transformations.euler_from_quaternion([msg_tf.transform.rotation.x, msg_tf.transform.rotation.y, msg_tf.transform.rotation.z, msg_tf.transform.rotation.w])

                        # Get the image
                        if msg._type == "sensor_msgs/CompressedImage":
                            image_arr = np.fromstring(msg.data, np.uint8)
                            cv_image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
                        elif msg._type == "sensor_msgs/Image":
                            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                        else:
                            raise ValueError("Image topic type must be either \"sensor_msgs/Image\" or \"sensor_msgs/CompressedImage\".")
                        
                        # Crop the image
                        if args.image_crop is not None:
                            cv_image = cv_image[args.image_crop[1]:args.image_crop[1] + args.image_crop[3], # y:y+h
                                                args.image_crop[0]:args.image_crop[0] + args.image_crop[2]] # x:x+w

                        # Scale the image
                        if args.image_scale != 1.0:
                            cv_image = cv2.resize(cv_image, (int(cv_image.shape[1] * args.image_scale),
                                                             int(cv_image.shape[0] * args.image_scale)), cv2.INTER_AREA)

                        # Get the label     0: Unknown, 1: No anomaly, 2: Contains an anomaly
                        image_label = get_label(cv_image, image_label)
                        if image_label == None: # [esc]
                            logger.warning("Interrupted!")
                            return
                        
                        output_file = os.path.join(output_dir, str(t.to_nsec()))

                        cv2.imwrite(output_file + ".jpg", cv_image)
                        
                        feature_dict = {
                            "location/translation/x"   : translation.x,
                            "location/translation/y"   : translation.y,
                            "location/translation/z"   : translation.z,
                            "location/rotation/x"      : euler[0],
                            "location/rotation/y"      : euler[1],
                            "location/rotation/z"      : euler[2],
                            "time"                     : t.to_nsec(),
                            "label"                    : image_label, # 0: Unknown, 1: No anomaly, 2: Contains an anomaly
                            "rosbag"                   : bag_file
                        }

                        with open(output_file + ".yml", "w") as yaml_file:
                            yaml.dump(feature_dict, yaml_file, default_flow_style=False)

                    except KeyboardInterrupt:
                        logger.info("Cancelled")
                        return
                    except tf2.ExtrapolationException:
                        skipped_count += 1
                        
                    # Print progress
                    pbar.set_postfix("%i skipped" % skipped_count)
                    pbar.update()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    rosbag_to_images()