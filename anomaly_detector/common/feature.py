import numpy as np

class Feature(np.ndarray):
    """A Feature is the output of a Feature Extractor (values) with metadata as attributes"""

    def __new__(cls, input_array, metadata, x, y, w, h):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        
        obj.metadata = metadata
        obj.x = x
        obj.y = y
        obj.w = w
        obj.h = h

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        
        self.metadata = getattr(obj, "metadata", None)

        # Patch position
        self.x = getattr(obj, "x", None)
        self.y = getattr(obj, "y", None)

        # Patches per image
        self.w = getattr(obj, "w", None)
        self.h = getattr(obj, "h", None)

        # Will eventually be array [x, y]
        # (call FeatureArray.calculate_locations)
        self.location = None
    
    time              = property(lambda self: self.metadata["time"])
    label             = property(lambda self: self.metadata["label"])
    rosbag            = property(lambda self: self.metadata["rosbag"])
    tfrecord          = property(lambda self: self.metadata["tfrecord"])
    feature_extractor = property(lambda self: self.metadata["feature_extractor"])

    camera_position   = property(lambda self: np.array([self.metadata["location/translation/x"],
                                                        self.metadata["location/translation/y"]]))
    camera_rotation   = property(lambda self: self.metadata["location/rotation/z"])

if __name__ == "__main__":
    meta = {"time": 232, "test": 24}
    f = Feature(np.array([2,3,4]), meta, 0, 0, 10, 10)
    print f.w
    print f.metadata
    print f.time