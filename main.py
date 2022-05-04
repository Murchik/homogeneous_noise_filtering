import librosa
import plot

# Load signal
signal, sample_rate = librosa.load("audio/Jason Walsh - Renegade_01.wav")

# Plot signal
# plot.signal(signal, sample_rate)
# plot.magnitude_spectrum(signal, sample_rate)
plot.spectrogram(signal, sample_rate)
