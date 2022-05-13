import numpy as np


class Frame:
    _samples: np.ndarray
    _timestamp: float
    _duration_s: float

    def __init__(self, samples, timestamp, duration):
        self._samples = samples
        self._timestamp = timestamp
        self._duration_s = duration

    def get_samples(self):
        return self._samples

    def get_samples_bytes(self):
        return np.ndarray.tobytes(self.get_samples())
