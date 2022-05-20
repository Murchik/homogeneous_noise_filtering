import time
import librosa
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

from datetime import timedelta as td
from scipy.io import wavfile

import plot


def _short_time_fourier_transform(y, n_fft, hop_length, window_len):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=window_len)


def _inverse_short_time_fourier_transform(y, hop_length, window_len):
    return librosa.istft(y, hop_length=hop_length, win_length=window_len)


def _magnitude_to_decibels(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _magnitude_to_amplitude(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)


def plot_statistics_and_filter(
    mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
):
    fig, ax = plt.subplots(ncols=2, figsize=(20, 4))
    plt_mean, = ax[0].plot(mean_freq_noise, label="Mean power of noise")
    plt_std, = ax[0].plot(std_freq_noise, label="Std. power of noise")
    plt_std, = ax[0].plot(noise_thresh, label="Noise threshold (by frequency)")
    ax[0].set_title("")
    ax[0].legend()
    cax = ax[1].matshow(smoothing_filter, origin="lower")
    fig.colorbar(cax)
    ax[1].set_title("")
    plt.show()

def filter_noise(
    voice_signal_path,
    noise_sample_path,
    n_grad_frequency=2,
    n_grad_time=4,
    num_ft=2048,
    window_len=2048,
    step_len=512,
    standard_threshold=1.5,
    prop_decrease=1.0,
    verbose=False,
    visual=False,
):
    voice_sr, voice_signal = wavfile.read(voice_signal_path)
    voice_signal = voice_signal / 32768

    noise_sr, noise_signal = wavfile.read(noise_sample_path)
    noise_signal = noise_signal / 32768

    if verbose:
        start = time.time()

    noise_spectre = _short_time_fourier_transform(noise_signal, num_ft, step_len, window_len)
    noise_spectre_in_decebels = _magnitude_to_decibels(np.abs(noise_spectre))

    mean_frequency_noise = np.mean(noise_spectre_in_decebels, axis=1)
    std_frequency_noise = np.std(noise_spectre_in_decebels, axis=1)
    noise_threshold = mean_frequency_noise + std_frequency_noise * standard_threshold

    if verbose:
        print("STFT on noise:", td(seconds=time.time() - start))
        start = time.time()

    if verbose:
        start = time.time()

    signal_spectre = _short_time_fourier_transform(voice_signal, num_ft, step_len, window_len)
    signal_spectre_decebels = _magnitude_to_decibels(np.abs(signal_spectre))

    if verbose:
        print("STFT on signal:", td(seconds=time.time() - start))
        start = time.time()

    mask_gain_decebels = np.min(_magnitude_to_decibels(np.abs(signal_spectre)))

    db_threshold = np.repeat(
        np.reshape(noise_threshold, [1, len(mean_frequency_noise)]),
        np.shape(signal_spectre_decebels)[1],
        axis=0,
    ).T

    signal_mask = signal_spectre_decebels < db_threshold
    if verbose:
        print("Masking:", td(seconds=time.time() - start))
        start = time.time()

    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_frequency + 1, endpoint=False),
                np.linspace(1, 0, n_grad_frequency + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)

    signal_mask = scipy.signal.fftconvolve(
        signal_mask, smoothing_filter, mode="same")
    signal_mask = signal_mask * prop_decrease
    if verbose:
        print("Mask convolution:", td(seconds=time.time() - start))
        start = time.time()

    signal_ft_decebels_masked = (signal_spectre_decebels * (1 - signal_mask) 
                                + np.ones(np.shape(mask_gain_decebels)) * mask_gain_decebels * signal_mask)
    signal_imaginary_masked = np.imag(signal_spectre) * (1 - signal_mask)
    signal_ft_amplitude = (_magnitude_to_amplitude(signal_ft_decebels_masked) * np.sign(signal_spectre)) + (1j * signal_imaginary_masked)
    if verbose:
        print("Mask application:", td(seconds=time.time() - start))
        start = time.time()

    recovered_signal = _inverse_short_time_fourier_transform(signal_ft_amplitude, step_len, window_len)
    recovered_spectrogram = _magnitude_to_decibels(np.abs(_short_time_fourier_transform(recovered_signal, num_ft, step_len, window_len)))
    if verbose:
        print("Signal recovery:", td(seconds=time.time() - start))
    if visual:
        plot.spectrogram(noise_spectre_in_decebels, title="Спектрограмма шума")
    # if visual:
    #     plot_statistics_and_filter(
    #         mean_frequency_noise, std_frequency_noise, noise_threshold, smoothing_filter
    #     )
    if visual:
        plot.spectrogram(signal_spectre_decebels, title="Спектрограмма исходного сигнала")
    if visual:
        plot.spectrogram(signal_mask, title="Маска")
    # if visual:
    #     plot.spectrogram(signal_ft_decebels_masked, title="Masked signal")
    if visual:
        plot.spectrogram(recovered_spectrogram, title="Спектрограмма обработанного сигнала")
    return recovered_signal
