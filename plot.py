import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import librosa
import librosa.display


def signal(signal, sample_rate, title="signal"):
    time = np.linspace(0, len(signal) / sample_rate, len(signal))

    plt.figure(figsize=(18, 5))
    plt.plot(time, signal)
    plt.xlabel("time")
    plt.title(title)
    plt.show()


def magnitude_spectrum(signal, sample_rate, title="magnitude spectrum", f_ratio=0.5):
    ft = fft.fft(signal)
    magnitude = np.abs(ft)

    frequency = np.linspace(0, sample_rate, len(magnitude))
    num_frequency_bins = int(len(frequency) * f_ratio)

    plt.figure(figsize=(18, 5))
    plt.plot(frequency[:num_frequency_bins], magnitude[:num_frequency_bins])
    plt.xlabel("Hz")
    plt.title(title)
    plt.show()


def spectrogram(signal, sample_rate, y_axis="log", frame_size=2**11, hop_size=2**9):
    signal_stft = librosa.stft(signal, n_fft=frame_size, hop_length=hop_size)

    y_scale = np.abs(signal_stft) ** 2
    y_log_scale = librosa.power_to_db(y_scale)

    plt.figure(figsize=(18, 5))
    librosa.display.specshow(y_log_scale,
                             sr=sample_rate,
                             hop_length=hop_size,
                             x_axis="time",
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.show()
