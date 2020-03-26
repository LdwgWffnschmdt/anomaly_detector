import numpy as np

class FeatureArray(np.ndarray):
    """Array with metadata."""

    def __new__(cls, **kwargs):
        features = np.ones((20000, 8, 8, 32))

        labels = np.ones((20000), dtype=np.int8)
        labels[10:20] = 0
        
        directions = np.ones((20000), dtype=np.int8)
        directions[15:25] = 0

        metadata = np.rec.array([labels, directions], dtype=[("label", "int8"), ("direction", "int8")])

        obj = np.asarray(features).view(cls)
        obj.metadata = metadata
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.metadata = getattr(obj, "metadata", None)

    label0 = property(lambda self: self[self.metadata.label == 0])

    def __getitem__(self, key):
        result = np.ndarray.__getitem__(self, key)
        if isinstance(result, FeatureArray):
            if isinstance(key, tuple):
                # print "KEY", key
                result.metadata = self.metadata[key[0]]
            else:
                result.metadata = self.metadata[key]
        return result

if __name__ == "__main__":
    features = FeatureArray()

    # print features.shape
    # print features.metadata.shape

    print features[0,...,1:2].label0.shape
    print features[0,...,1:2].label0.metadata.label

    # print features[15:30].shape
    # print features[15:30].metadata.direction

    # print features[features.metadata.label == 0].shape
    # print features[features.metadata.label == 0].metadata.label.shape