from featureExtractorBase import FeatureExtractorBase

class FeatureExtractorEfficientNetB6(FeatureExtractorBase):
    """Feature extractor based on EfficientNetB6 (trained on ImageNet).
    Generates 1x1x2304 feature vectors per image
    """

    def __init__(self):
        FeatureExtractorBase.__init__(self)

        self.IMG_SIZE = 528 # All images will be resized to 528x528

        # Create the base model from the pre-trained model
        self.model = self.load_model("https://tfhub.dev/google/efficientnet/b6/feature-vector/1")
    
    def extract_batch(self, batch):
        return self.model(batch)

# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorEfficientNetB6()
    extractor.plot_model(extractor.model, 600)
    extractor.extract_files("/home/ludwig/ros/src/ROS-kate_bag/bags/FieldSAFE/TFRecord/2016-10-25-11-41-21_example.tfrecord")
    # extractor.extract_files("/home/ludwig/ros/src/ROS-kate_bag/bags/real/TFRecord/*.tfrecord")