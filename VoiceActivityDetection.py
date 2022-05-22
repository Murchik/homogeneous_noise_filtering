import numpy as np
import scipy.io.wavfile as wf

from Frame import Frame, _get_signal_from_frames


class VoiceActivityDetector():
    def __init__(self, sample_window_ms=20, sample_overlap_ms=10):
        self.sample_window_ms = sample_window_ms
        self.sample_overlap_ms = sample_overlap_ms
        self.sample_window = self.sample_window_ms / 1000
        self.sample_overlap = self.sample_overlap_ms / 1000
        self.speech_window = 0.5
        self.speech_energy_threshold = 0.2  # % of energy in voice frequency range
        self.speech_start_frequency = 300
        self.speech_end_frequency = 3000

    def _read_wavefile(self, wave_file):
        self.rate, self.data = wf.read(wave_file)
        self.channels = len(self.data.shape)
        self.filename = wave_file
        return self

    def _convert_to_mono(self):
        if self.channels == 2:
            self.data = np.mean(self.data, axis=1, dtype=self.data.dtype)
            self.channels = 1
        return self

    def _calculate_frequencies(self, audio_data):
        signal_frequency = np.fft.fftfreq(len(audio_data), 1.0/self.rate)
        signal_frequency = signal_frequency[1:]
        return signal_frequency

    def _calculate_amplitude(self, audio_data):
        data_ampl = np.abs(np.fft.fft(audio_data))
        data_ampl = data_ampl[1:]
        return data_ampl

    def _calculate_energy(self, data):
        data_amplitude = self._calculate_amplitude(data)
        data_energy = data_amplitude ** 2
        return data_energy

    def _connect_energy_with_frequencies(self, data_freq, data_energy):
        energy_freq = {}
        for (i, freq) in enumerate(data_freq):
            if abs(freq) not in energy_freq:
                energy_freq[abs(freq)] = data_energy[i] * 2
        return energy_freq

    def _calculate_normalized_energy(self, data):
        signal_frequency = self._calculate_frequencies(data)
        data_energy = self._calculate_energy(data)
        energy_frequency = self._connect_energy_with_frequencies(
            signal_frequency, data_energy)
        return energy_frequency

    def _sum_energy_in_band(self, energy_frequencies, voice_start_frequency, voice_end_frequency):
        sum_energy = 0
        for f in energy_frequencies.keys():
            if voice_start_frequency < f < voice_end_frequency:
                sum_energy += energy_frequencies[f]
        return sum_energy

    def _median_filter(self, x, k):
        assert k % 2 == 1, "Длина медианного фильтра должна быть нечетной."
        assert x.ndim == 1, "Входные данные должны быть одномерными."
        k2 = (k - 1) // 2
        y = np.zeros((len(x), k), dtype=x.dtype)
        y[:, k2] = x
        for i in range(k2):
            j = k2 - i
            y[j:, i] = x[:-j]
            y[:j, i] = x[0]
            y[:-j, -(i+1)] = x[j:]
            y[-j:, -(i+1)] = x[-1]
        return np.median(y, axis=1)

    def _smooth_speech_detection(self, detected_windows):
        median_window = int(self.speech_window/self.sample_window)
        if median_window % 2 == 0:
            median_window = median_window-1
        median_energy = self._median_filter(
            detected_windows[:, 1], median_window)
        return median_energy

    def get_timestamps(self, detected_windows):
        speech_time = []
        is_speech = 0
        for window in detected_windows:
            if (window[1] == 1.0 and is_speech == 0):
                is_speech = 1
                speech_label = {}
                speech_time_start = window[0] / self.rate
                speech_label['speech_begin'] = speech_time_start
            if (window[1] == 0.0 and is_speech == 1):
                is_speech = 0
                speech_time_end = window[0] / self.rate
                speech_label['speech_end'] = speech_time_end
                speech_time.append(speech_label)
        return speech_time

    def detect_speech(self, frames: list[Frame], sample_rate):
        self.rate = sample_rate
        detected_windows = np.array([])
        sample_window = int(self.rate * self.sample_window)
        sample_overlap = int(self.rate * self.sample_overlap)
        data = _get_signal_from_frames(
            frames, self.rate, self.sample_window_ms)
        sample_start = 0
        start_band = self.speech_start_frequency
        end_band = self.speech_end_frequency
        while (sample_start < (len(data) - sample_window)):
            sample_end = sample_start + sample_window
            if sample_end >= len(data):
                sample_end = len(data) - 1
            data_window = data[sample_start:sample_end]
            energy_frequency = self._calculate_normalized_energy(data_window)
            sum_voice_energy = self._sum_energy_in_band(
                energy_frequency, start_band, end_band)
            sum_full_energy = sum(energy_frequency.values())
            speech_ratio = sum_voice_energy / sum_full_energy
            speech_ratio = speech_ratio > self.speech_energy_threshold
            detected_windows = np.append(
                detected_windows, [sample_start, speech_ratio])
            sample_start += sample_overlap
        detected_windows = detected_windows.reshape(
            int(len(detected_windows) / 2), 2)
        detected_windows[:, 1] = self._smooth_speech_detection(
            detected_windows)
        return detected_windows
