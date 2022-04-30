import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft, fftfreq

sample_rate, samples = wavfile.read("audio/sin.wav")
sample_num = len(samples)

samples_fft = fft(samples)

magnitude = np.absolute(samples_fft)
frequency = np.linspace(0, sample_rate, sample_num)
frequency_max = len(magnitude) // 2

plt.figure(figsize=(18, 8))
plt.plot(frequency[:frequency_max],
         magnitude[:frequency_max])  # magnitude spectrum
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()
