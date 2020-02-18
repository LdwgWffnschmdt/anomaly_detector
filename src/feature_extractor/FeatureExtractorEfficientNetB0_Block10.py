from featureExtractorBase import FeatureExtractorBase

class FeatureExtractorEfficientNetB0_Block10(FeatureExtractorBase):
    """Feature extractor based on EfficientNetB0 (trained on ImageNet).
    Output layer: block_10
    Generates 14x14x112 feature vectors per image
    """

    def __init__(self):
        FeatureExtractorBase.__init__(self)

        self.IMG_SIZE = 224 # All images will be resized to 224x224

        # Create the base model from the pre-trained model
        self.model = self.load_model("https://tfhub.dev/google/efficientnet/b0/feature-vector/1",
                                     output_key="block_10")
                       
    def extract_batch(self, batch):
        return self.model(batch)

# Only for tests
if __name__ == "__main__":
    extractor = FeatureExtractorEfficientNetB0_Block10()
    extractor.plot_model(extractor.model)
    extractor.extract_files("/home/ludwig/ros/src/ROS-kate_bag/bags/real/TFRecord/*.tfrecord")
                 
    #   features                        : (1, 7, 7, 320)
    #   block_11                        : (1, 7, 7, 192)
    #   block_12                        : (1, 7, 7, 192)
    #   block_15                        : (1, 7, 7, 320)
    #   block_14                        : (1, 7, 7, 192)
    #   block_12/expansion_output       : (1, 7, 7, 1152)
    #   reduction_5                     : (1, 7, 7, 320)
    #   block_13                        : (1, 7, 7, 192)
    #   reduction_5/expansion_output    : (1, 7, 7, 1152)
    #   block_11/expansion_output       : (1, 7, 7, 672)
    #   block_13/expansion_output       : (1, 7, 7, 1152)
    #   block_15/expansion_output       : (1, 7, 7, 1152)
    #   block_14/expansion_output       : (1, 7, 7, 1152)
    #   block_10                        : (1, 14, 14, 112)
    #   block_6/expansion_output        : (1, 14, 14, 480)
    #   block_5/expansion_output        : (1, 14, 14, 240)
    #   block_7/expansion_output        : (1, 14, 14, 480)
    #   reduction_4/expansion_output    : (1, 14, 14, 672)
    #   reduction_4                     : (1, 14, 14, 112)
    #   block_8/expansion_output        : (1, 14, 14, 480)
    #   block_9                         : (1, 14, 14, 112)
    #   block_9/expansion_output        : (1, 14, 14, 672)
    #   block_10/expansion_output       : (1, 14, 14, 672)
    #   block_8                         : (1, 14, 14, 112)
    #   block_5                         : (1, 14, 14, 80)
    #   block_7                         : (1, 14, 14, 80)
    #   block_6                         : (1, 14, 14, 80)
    #   block_3/expansion_output        : (1, 28, 28, 144)
    #   reduction_3                     : (1, 28, 28, 40)
    #   block_4/expansion_output        : (1, 28, 28, 240)
    #   reduction_3/expansion_output    : (1, 28, 28, 240)
    #   block_4                         : (1, 28, 28, 40)
    #   block_3                         : (1, 28, 28, 40)
    #   block_2/expansion_output        : (1, 56, 56, 144)
    #   block_1/expansion_output        : (1, 56, 56, 96)
    #   reduction_2                     : (1, 56, 56, 24)
    #   reduction_2/expansion_output    : (1, 56, 56, 144)
    #   block_1                         : (1, 56, 56, 24)
    #   block_2                         : (1, 56, 56, 24)
    #   reduction_1                     : (1, 112, 112, 16)
    #   stem                            : (1, 112, 112, 32)
    #   block_0/expansion_output        : (1, 112, 112, 32)
    #   reduction_1/expansion_output    : (1, 112, 112, 32)
    #   block_0                         : (1, 112, 112, 16)
    #   default                         : (1, 1280)
    #   pooled_features                 : (1, 1280)
