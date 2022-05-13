import numpy as np
import contextlib
import wave

from Frame import Frame


class Audio:
    _num_channels: int
    _bitrate: int
    _sample_rate: int
    _signal: np.ndarray

    def get_sample_rate(self):
        return self._sample_rate

    def get_signal(self):
        return self._signal

    def get_signal_bytes(self):
        return np.ndarray.tobytes(self.get_signal())

    def get_signal_frames(self, frame_duration_ms=30) -> list:
        signal = self.get_signal()
        sample_rate = self.get_sample_rate()

        # n - кол-во семплов в одном фрейме
        # (при длительности фрейма 30ms и sr=32kHz: n = 960)
        n = int(sample_rate * (frame_duration_ms / 1000.0))

        offset = 0
        frames = []
        timestamp = 0.0
        frame_duration_s = n / sample_rate
        while offset + n < len(signal):
            frames.append(Frame(signal[offset:offset + n],
                                timestamp, frame_duration_s))
            timestamp += frame_duration_s
            offset += n
        return frames

    def wavread(self, path):
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            # Чтение числа каналов (поддерживается только моно звук)
            self._num_channels = wf.getnchannels()
            assert self._num_channels == 1

            # Чтение битрейта (поддерживается только 16-bit integer PCM)
            self._bitrate = wf.getsampwidth()
            assert self._bitrate == 2

            # Чтение частоты дискретизации (поддерживается только 8kHz, 16kHz, 32kHz и 48kHz)
            self._sample_rate = wf.getframerate()
            assert self._sample_rate in (8000, 16000, 32000, 48000)

            # Чтение аудио сигнала
            audio_bytes = wf.readframes(wf.getnframes())

            # Преобразование сигнала из контейнера bytes в numpy.ndarray где dtype=np.int16
            data = np.frombuffer(audio_bytes, dtype=np.int16)
            self._signal = np.array(data)

            return self._signal, self._sample_rate

    def wavwrite(self, path):
        with contextlib.closing(wave.open(path, 'wb')) as wf:
            wf.setnchannels(self._num_channels)
            wf.setsampwidth(self._bitrate)
            wf.setframerate(self._sample_rate)

            # Преобразование numpy.ndarray(dtype=np.int16) в bytes
            signal_bytes = np.ndarray.tobytes(self._signal)
            wf.writeframes(signal_bytes)
