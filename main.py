import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft, fftfreq

# Чтение синусоиды из файла
sample_rate, signal = wavfile.read("audio/Jason Walsh - Renegade_01.wav")
sample_num = len(signal)

# Быстрое преобразование фурье
signal_fft = fft(signal)

# DO WORK DO

# Обратное преобразование фурье
signal_ifft = ifft(signal_fft)

# Запись результата в файл
outputMatrix = np.array(signal_ifft, dtype=np.int16)
wavfile.write("audio/out.wav", sample_rate, outputMatrix)

# Вывод на график
signal_magnitude = np.absolute(signal_fft)
frequency = np.linspace(0, sample_rate, sample_num)
frequency_max = len(signal_magnitude) // 2
plt.figure(figsize=(18, 8))
plt.plot(frequency[:frequency_max], signal_magnitude[:frequency_max])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()
