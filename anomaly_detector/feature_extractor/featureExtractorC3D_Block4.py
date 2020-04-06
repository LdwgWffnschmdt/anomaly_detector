from feature_extractor import FeatureExtractorC3D

class FeatureExtractorC3D_Block4(FeatureExtractorC3D):
    """Feature extractor based on C3D (trained on sports1M).
    Output layer: conv4b + MaxPooling3D to reduce frames
    Generates 14x14x512 feature vectors per temporal image batch
    """
    __layer__ = "conv4b"
    BATCH_SIZE = 32

# Only for tests
if __name__ == "__main__":
    from common import PatchArray
    import numpy as np
    extractor = FeatureExtractorC3D_Block4()
    # extractor.plot_model(extractor.model)
    patches = PatchArray()

    p = patches[:, 0, 0]

    f = np.zeros(p.shape, dtype=np.bool)
    f[:] = np.logical_and(p.directions == 1,                                   # CCW and
                            np.logical_or(p.labels == 2,                         #   Anomaly or
                                        np.logical_and(p.round_numbers >= 7,   #     Round between 2 and 5
                                                        p.round_numbers <= 9)))

    # Let's make contiguous blocks of at least 10, so
    # we can do some meaningful temporal smoothing afterwards
    for i, b in enumerate(f):
        if b and i - 10 >= 0:
            f[i - 10:i] = True

    patches = patches[f]

    extractor.extract_dataset(patches.to_temporal_dataset(), patches.shape[0])