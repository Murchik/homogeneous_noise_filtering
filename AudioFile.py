import wave
import contextlib
import numpy as np


class Frame(object):
    samples: np.ndarray
    num: int
    timestamp: float
    duration_s: float

    def __init__(self, samples, num, timestamp, duration):
        self.samples = samples
        self.num = num
        self.timestamp = timestamp
        self.duration_s = duration

    def get_bytes(self):
        return np.ndarray.tobytes(self.samples)


class AudioFile:
    num_channels: int
    bitrate: int
    sample_rate: int
    signal: np.ndarray  # dtype=np.int16

    def wavread(self, path):
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            # Чтение числа каналов (поддерживается только моно звук)
            self.num_channels = wf.getnchannels()
            assert self.num_channels == 1

            # Чтение битрейта (поддерживается только 16-bit integer PCM)
            self.bitrate = wf.getsampwidth()
            assert self.bitrate == 2

            # Чтение частоты дискретизации (поддерживается только 8kHz, 16kHz, 32kHz и 48kHz)
            self.sample_rate = wf.getframerate()
            assert self.sample_rate in (8000, 16000, 32000, 48000)

            # Чтение аудио сигнала
            audio_bytes = wf.readframes(wf.getnframes())

            # Преобразование сигнала из контейнера bytes в numpy.ndarray где dtype=np.int16
            data = np.frombuffer(audio_bytes, dtype=np.int16)
            self.signal = np.array(data)

            return self.signal, self.sample_rate

    def wavwrite(self, path):
        with contextlib.closing(wave.open(path, 'wb')) as wf:
            wf.setnchannels(self.num_channels)
            wf.setsampwidth(self.bitrate)
            wf.setframerate(self.sample_rate)

            # Преобразование numpy.ndarray(dtype=np.int16) в bytes
            signal_bytes = np.ndarray.tobytes(self.signal)
            wf.writeframes(signal_bytes)

    def getframes(self, frame_duration_ms=30):
        # n - кол-во семплов в одном фрейме
        # (при длительности фрейма 30ms и sr=32kHz: n = 1920)
        n = int(self.sample_rate * (frame_duration_ms / 1000.0))

        offset = 0
        frames = []
        timestamp = 0.0
        frame_duration_s = n / self.sample_rate
        i = 0
        while offset + n < len(self.signal):
            frames.append(Frame(self.signal[offset:offset + n],
                                i, timestamp, frame_duration_s))
            timestamp += frame_duration_s
            offset += n
            i += 1
        return frames
