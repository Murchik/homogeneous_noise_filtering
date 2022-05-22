import numpy as np


class Frame:
    _samples: 'np.ndarray[np.int16]'
    _timestamp: float
    _duration_s: float

    def __init__(self, samples, timestamp, duration):
        self._samples = samples
        self._timestamp = timestamp
        self._duration_s = duration

    def get_samples(self) -> int:
        return self._samples

    def get_samples_bytes(self) -> bytes:
        return np.ndarray.tobytes(self.get_samples())


def _get_signal_from_frames(frames: list[Frame], sample_rate, frame_duration_ms) -> 'np.ndarray[np.int16]':
    signal = []
    for frame in frames:
        for sample in frame.get_samples():
            signal.append(sample)
    signal = np.array(signal, dtype=np.int16)
    return signal
