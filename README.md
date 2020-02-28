# anomaly_detector
`WORK IN PROGRESS!`
Detect anomalies in images using deep features

## Install
```bash
python -m virtualenv .env
source .env/bin/
pip install -r requirements.txt

pip install -e .    # Install the current directory as pip package but keep it editable (-e)
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