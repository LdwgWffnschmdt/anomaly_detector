# rosbag_to_tfrecord

A small script to convert bag files to [TensorFlow TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord). Will as of now only include an image topic with position and rotation from /tf.

Utilizes [tf_bag](https://github.com/IFL-CAMP/tf_bag) and ```pip install opencv-python```