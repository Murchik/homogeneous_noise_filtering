import numpy as np


class Segment:
    _frames: list

    def __init__(self, frames):
        self._frames = frames

    def get_frames(self):
        return self._frames

    def get_frames_bytes(self):
        return b''.join([frame.get_samples_bytes() for frame in self.get_frames() if frame])

    def get_signal(self):
        signal = []
        for frame in self.get_frames():
            for sample in frame.get_samples():
                signal.append(sample)
        return signal

    def get_signal_bytes(self):
        signal = np.array(self.get_signal(), dtype=np.int16)
        return np.ndarray.tobytes(signal)
