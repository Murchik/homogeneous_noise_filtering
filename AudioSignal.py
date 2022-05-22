import numpy as np
from scipy.io import wavfile

from Frame import Frame


class AudioSignal:
    _sample_rate: int
    _signal: 'np.ndarray[np.int16]'

    def __init__(self, path):
        self._sample_rate, self._signal = wavfile.read(path)

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def get_signal(self) -> 'np.ndarray[np.int16]':
        return self._signal

    def get_signal_bytes(self) -> bytes:
        return np.ndarray.tobytes(self.get_signal())

    def get_signal_frames(self, frame_duration_ms=20) -> list[Frame]:
        signal = self.get_signal()
        sample_rate = self.get_sample_rate()

        n = int(sample_rate * (frame_duration_ms / 1000.0))

        offset = 0
        frames = []
        frame_duration_s = n / sample_rate
        while offset + n < len(signal):
            frames.append(Frame(signal[offset:offset + n],
                                offset, frame_duration_s))
            offset += n
        return frames

    def get_frames_from_interval(self, frames_time_start, frames_time_end, frame_duration_ms=20) -> list[Frame]:
        signal = self.get_signal()
        sample_rate = self.get_sample_rate()

        n = int(sample_rate * (frame_duration_ms / 1000.0))

        offset = 0
        frames = []
        timestamp = 0.0
        frame_duration_s = n / sample_rate
        while offset + n < len(signal):
            if(timestamp > frames_time_start and timestamp < frames_time_end):
                frames.append(Frame(signal[offset:offset + n],
                                    offset, frame_duration_s))
            timestamp += frame_duration_s
            offset += n
        return frames
