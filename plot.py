import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import librosa
import librosa.display


figure_size = (12, 4)


def signal(signal, sample_rate, title="signal"):
    # Вычисление длительности сигнала:
    # длительность = кол-во_семплов / кол-во_семплов_в_секунду
    duration = len(signal) / sample_rate

    time = np.linspace(0, duration, len(signal))

    # Вывод результата
    plt.figure(figsize=(18, 5))
    plt.plot(time, signal)
    plt.ylabel("Амплитуда")
    plt.xlabel("Время (Секунды)")
    plt.title(title)
    plt.show()


def magnitude_spectrum(signal, sample_rate,
                       title="magnitude spectrum",
                       f_ratio=0.5):
    # Получение распределение сигналов по частотам (спектра)
    # с помощью быстрого преобразования фурье (БПФ)
    ft = fft.fft(signal)

    # Переход от комплексного результата БПФ к абсолютному значению сигнала
    magnitude = np.abs(ft)

    frequency = np.linspace(0, sample_rate, len(magnitude))

    num_frequency_bins = int(len(frequency) * f_ratio)

    # Вывод результата
    plt.figure(figsize=figure_size)
    plt.plot(frequency[:num_frequency_bins], magnitude[:num_frequency_bins])
    plt.ylabel("Величина")
    plt.xlabel("Частота (Гц)")
    plt.title(title)
    plt.show()


def spectrogram(signal, title):
    figure, x_axis = plt.subplots(figsize=figure_size)
    color_x_axis = x_axis.matshow(
        signal,
        origin="lower",
        aspect="auto",
        cmap=plt.cm.seismic,
        vmin=-1 * np.max(np.abs(signal)),
        vmax=np.max(np.abs(signal)),
    )
    plt.colorbar(color_x_axis)
    plt.xlabel(title)
    plt.tight_layout()
    plt.show()


def spectrogram_from_signal(signal, sample_rate,
                            y_axis="log",
                            frame_size=2048, hop_length=512):
    # Получение матрицы распределения спектров во времени (спектрограммы):
    # по сигналу перемещается окно размером frame_size с шагом hop_size,
    # к каждому полученному таким окном фрейму применяется БПФ
    signal = np.ndarray.astype(signal, dtype=np.float16)
    signal_stft = librosa.stft(signal, n_fft=frame_size, hop_length=hop_length)

    # Переход от комплексного результата к абсолютному значению сигнала
    y_scale = np.abs(signal_stft) ** 2

    # Перевод полученных значений к логарифмическому представлению
    y_log_scale = librosa.power_to_db(y_scale)

    # Вывод результата
    plt.figure(figsize=figure_size)
    librosa.display.specshow(y_log_scale,
                             sr=sample_rate,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.ylabel("Частота (Гц)")
    plt.xlabel("Время (Секунды)")
    plt.tight_layout()
    plt.show()
