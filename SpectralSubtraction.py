import numpy as np
import scipy.signal
import librosa

import plot


class SpectralSubtractor():
    def __init__(self) -> None:
        self.n_grad_frequency = 2
        self.n_grad_time = 4
        self.num_ft = 2048
        self.window_len = 2048
        self.step_len = 512
        self.standard_threshold = 1.5
        self.prop_decrease = 1.0
        self.verbose = False
        self.visual = False

    def filter_noise(self, voice_signal: 'np.ndarray[np.int16]', noise_signal: 'np.ndarray[np.int16]', visual=False) -> 'np.ndarray[np.int16]':
        voice_signal = voice_signal / 32768
        noise_signal = noise_signal / 32768

        noise_spectre = _short_time_fourier_transform(
            noise_signal, self.num_ft, self.step_len, self.window_len)
        noise_spectre_in_decebels = _magnitude_to_decibels(
            np.abs(noise_spectre))

        mean_frequency_noise = np.mean(noise_spectre_in_decebels, axis=1)
        std_frequency_noise = np.std(noise_spectre_in_decebels, axis=1)
        noise_threshold = mean_frequency_noise + \
            std_frequency_noise * self.standard_threshold

        signal_spectre = _short_time_fourier_transform(
            voice_signal, self.num_ft, self.step_len, self.window_len)
        signal_spectre_decebels = _magnitude_to_decibels(
            np.abs(signal_spectre))

        mask_gain_decebels = np.min(
            _magnitude_to_decibels(np.abs(signal_spectre)))

        db_threshold = np.repeat(
            np.reshape(noise_threshold, [1, len(mean_frequency_noise)]),
            np.shape(signal_spectre_decebels)[1],
            axis=0,
        ).T

        signal_mask = signal_spectre_decebels < db_threshold

        smoothing_filter = np.outer(
            np.concatenate(
                [
                    np.linspace(0, 1, self.n_grad_frequency +
                                1, endpoint=False),
                    np.linspace(1, 0, self.n_grad_frequency + 2),
                ]
            )[1:-1],
            np.concatenate(
                [
                    np.linspace(0, 1, self.n_grad_time + 1, endpoint=False),
                    np.linspace(1, 0, self.n_grad_time + 2),
                ]
            )[1:-1],
        )
        smoothing_filter = smoothing_filter / np.sum(smoothing_filter)

        signal_mask = scipy.signal.fftconvolve(
            signal_mask, smoothing_filter, mode="same")
        signal_mask = signal_mask * self.prop_decrease
        signal_ft_decebels_masked = (signal_spectre_decebels * (1 - signal_mask)
                                     + np.ones(np.shape(mask_gain_decebels)) * mask_gain_decebels * signal_mask)
        signal_imaginary_masked = np.imag(signal_spectre) * (1 - signal_mask)
        signal_ft_amplitude = (_magnitude_to_amplitude(
            signal_ft_decebels_masked) * np.sign(signal_spectre)) + (1j * signal_imaginary_masked)

        recovered_signal = _inverse_short_time_fourier_transform(
            signal_ft_amplitude, self.step_len, self.window_len)
        recovered_spectrogram = _magnitude_to_decibels(np.abs(_short_time_fourier_transform(
            recovered_signal, self.num_ft, self.step_len, self.window_len)))

        if visual:
            plot.spectrogram(noise_spectre_in_decebels,
                             title="Спектрограмма шума")
        if visual:
            plot.spectrogram(signal_spectre_decebels,
                             title="Спектрограмма исходного сигнала")
        if visual:
            plot.spectrogram(signal_mask, title="Маска")
        if visual:
            plot.spectrogram(recovered_spectrogram,
                             title="Спектрограмма обработанного сигнала")

        recovered_signal = recovered_signal * 32768
        recovered_signal = np.array(recovered_signal, dtype=np.int16)
        return recovered_signal


def _short_time_fourier_transform(y, n_fft, hop_length, window_len):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=window_len)


def _inverse_short_time_fourier_transform(y, hop_length, window_len):
    return librosa.istft(y, hop_length=hop_length, win_length=window_len)


def _magnitude_to_decibels(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _magnitude_to_amplitude(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)
