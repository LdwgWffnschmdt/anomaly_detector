# anomaly_detector
WORK IN PROGRESS!
Detect anomalies in images using deep features

## Dependencies
```bash
python -m virtualenv .env
pip install -r requirements.txt^
```

# rosbag_to_tfrecord

A small script to convert bag files to [TensorFlow TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord). Will as of now only include an image topic with position and rotation from /tf.

## Usage
```bash
rosrun rosbag_to_tfrecord rosbag_to_tfrecord _bag_file:=/path/to/file.bag
```

### Parameters
```sh
bag_file      : ""
output_dir    : Defaults to {bag_file}/TFRecord
image_topic   : "/camera/color/image_raw"
images_per_bin: 10000
tf_map        : "map"
tf_base_link  : "base_link"

# -2: labeling mode (show image and wait for input) [space]: No anomaly
#                                                   [tab]  : Contains anomaly
# -1: continuous labeling mode (show image for 10ms, keep label until change)
#  0: Unknown
#  1: No anomaly
#  2: Contains an anomaly
label         : 0
```