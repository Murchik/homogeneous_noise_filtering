import librosa
import librosa.display
import scipy as sp
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np

audio_path = "audio/piano_c.wav"
ipd.Audio(audio_path)

# load audio file
signal, sr = librosa.load(audio_path)

ft = sp.fft.fft(signal)
magnitude = np.absolute(ft)
frequency = np.linspace(0, sr, len(magnitude))

plt.figure(figsize=(18, 8))
plt.plot(frequency[:len(magnitude)//2],
         magnitude[:len(magnitude)//2])  # magnitude spectrum
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()
