# anomaly_detector
WORK IN PROGRESS!
Detect anomalies in images using deep features

## Dependencies
```bash
pip install --user --upgrade tensorflow
pip install --user -q pyyaml h5py  # Required to save extracted features in HDF5 format
```

## rosbag_to_tfrecord

A small script to convert bag files to [TensorFlow TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord). Will as of now only include an image topic with position and rotation from /tf.

### Install dependencies
```pip install opencv-python```

### Usage
```bash
rosrun rosbag_to_tfrecord rosbag_to_tfrecord _bag_file:=/path/to/file.bag
```

### Parameters
```
bag_file      : ""
output_dir    : Defaults to {bag_file}-TFRecord
image_topic   : "/camera/color/image_raw"
dataset_name  : "train"
images_per_bin: 10000
tf_map        : "map"
tf_base_link  : "base_link"
```