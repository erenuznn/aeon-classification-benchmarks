import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket


class UnilateralFeatureExtractor:
    def __init__(self, num_kernels=10000):
        self.minirocket = MiniRocket(n_kernels=num_kernels, random_state=42)

    def _encode_time(self, timestamps):
        hours = timestamps.dt.hour + (timestamps.dt.minute / 60.0)
        time_sin = np.sin(2 * np.pi * hours / 24.0).values.reshape(-1, 1)
        time_cos = np.cos(2 * np.pi * hours / 24.0).values.reshape(-1, 1)
        return np.hstack((time_sin, time_cos))

    def fit_transform(self, X_raw, timestamps):
        print("Executing MiniRocket fit_transform and temporal encoding.")
        mr_features = self.minirocket.fit_transform(X_raw)
        time_features = self._encode_time(timestamps)
        return np.hstack((mr_features, time_features))

    def transform(self, X_raw, timestamps):
        print("Executing MiniRocket transform and temporal encoding.")
        mr_features = self.minirocket.transform(X_raw)
        time_features = self._encode_time(timestamps)
        return np.hstack((mr_features, time_features))