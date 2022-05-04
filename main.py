import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import plot


def add_signals(signal_1, signal_2):
    result = np.array([0] * len(signal_1), dtype=type(signal_1[0]))
    for i in range(0, len(signal_1)):
        if i >= len(signal_2):
            break
        result[i] = signal_1[i] + signal_2[i]
    return result


def subtract_signals(signal_1, signal_2):
    result = np.array([0] * len(signal_1), dtype=type(signal_1[0]))
    for i in range(0, len(signal_1)):
        if i >= len(signal_2):
            break
        result[i] = signal_1[i] - signal_2[i]
    return result


# Исходный речевой сигнал
test_signal, test_sr = librosa.load("audio/test.wav")

# Гул частотой 440hz
noise_signal, noise_sr = librosa.load("audio/sin.wav")

# test_sr == noise_sr == 44100hz

# Добавление гула к речевому сигналу
new_signal = add_signals(test_signal, noise_signal)
new_signal_sr = test_sr
wavfile.write('out.wav', new_signal_sr, new_signal)

# Вычитание гула из речевого сигнала
signal, sr = librosa.load("out.wav")
signal = subtract_signals(signal, noise_signal)
wavfile.write('out2.wav', sr, signal)


# frame_size = 2**13
# hop_size = 2**5

# ft_previous = sp_fft.fft(signal[:frame_size])

# diff = np.array([0] * len(signal), dtype=np.abs(ft_previous[0]))

# for i in range(frame_size, len(signal), hop_size):
#     ft_current = sp_fft.fft(signal[i:i+frame_size])
#     for j in range(0, len(ft_current)):
#         if (i + j >= len(signal)):
#             break
#         diff[i + j] += np.abs(ft_previous[j] - ft_current[j])
#     ft_previous = ft_current

# val, noise_freq = max((val, idx) for (idx, val) in enumerate(diff))
# print(noise_freq)
