# anomaly_detector
`WORK IN PROGRESS!`
Detect anomalies in images using deep features

## Install
```bash

python -m virtualenv .env           # Create virtualenv
source .env/bin/                    # Activate it
pip install -r requirements.txt     # Install the python dependencies

pip install -e .    # Install current directory as editable pip package
```
Note that if you want to be able to use the `rosbag_to_...` scripts to extract images and metadata from bag files you need to have at least
- a bare bones [ROS (Kinetic)](http://wiki.ros.org/kinetic/Installation/Ubuntu) and
- the [cv_bridge](http://wiki.ros.org/cv_bridge) package installed (`sudo apt-get install ros-kinetic-cv-bridge`).

## Constants
Create a file `./anomaly_detector/consts.py` with the following constants for quick debug excecutions:
```python
IMAGES_PATH   = "/path/to/Images/"
EXTRACT_FILES = "/path/to/Images/*.jpg"
FEATURES_PATH = "/path/to/Features/"
FEATURES_FILE = FEATURES_PATH + "C3D.h5"
FEATURES_FILES = FEATURES_PATH + "*.h5"

# Defaults for feature extraction
DEFAULT_BATCH_SIZE = 128
```

# Data preparation

### `Bag file` ➡ `TFRecord file(s)` ➡ `Feature extractor` ➡ `Anomaly model`

## rosbag_to_tfrecord

A small script to convert bag files to [TensorFlow TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord). Will as of now only include an image topic with position and rotation from `/tf`.

## Usage
```bash
python rosbag_to_tfrecord.py /path/to/file.bag
```

### Help
```sh
python rosbag_to_tfrecord.py --help
```