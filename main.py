import sys
from scipy.io import wavfile
from PyQt6.QtWidgets import QMainWindow, QMessageBox, QApplication, QFileDialog

from AudioSignal import AudioSignal
from Segment import Segment, wavwrite_segment
from VoiceActivityDetection import VoiceActivityDetector
from SpectralGating import filter_noise
from window import Ui_Dialog


class MainWindow(QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.BrowseInput.clicked.connect(self.browseInput)
        self.BrowseNoise.clicked.connect(self.browseNoiseFile)
        self.BrowseOutput.clicked.connect(self.browseOutFile)
        self.RemoveNoise.clicked.connect(self.removeNoise)
    

    def browseInput(self):
        message = 'Select Audio with voice for processing'
        directory = 'audio'
        files = '*.wav'
        file_name = QFileDialog.getOpenFileName(self, message, directory, files)
        self.InputVoiceFile.setText(file_name[0])

    def browseNoiseFile(self):
        message = 'Select Noise sample destination'
        directory = 'out'
        files = '*.wav'
        file_name = QFileDialog.getOpenFileName(self, message, directory, files)
        self.NoiseSampleFile.setText(file_name[0])

    def browseOutFile(self):
        message = 'Select output audio file destination'
        directory = 'out'
        files = '*.wav'
        file_name = QFileDialog.getOpenFileName(self, message, directory, files)
        self.OutputCleanVoice.setText(file_name[0])

    def removeNoise(self):
        inputFileName = self.InputVoiceFile.text()
        if len(inputFileName) == 0 or inputFileName == "Выберете файл для обработки (*.wav)":
            dialog = QMessageBox(self)
            error_message = "Не указан файл с речевым сигналом!"
            dialog.setText(error_message)
            dialog.show()
            return

        noiseFileName = self.NoiseSampleFile.text()
        if len(noiseFileName) == 0:
            noiseFileName = "out/out_noise_sample.wav"

        outputFileName = self.BrowseOutput.text()
        if len(outputFileName) == 0 or outputFileName == "Выбрать":
            outputFileName = "out/out_clean_voice.wav"
                
        frame_duration_ms = 20

        # Чтение файла и разбитие на фреймы заданной длительности
        ass = AudioSignal()
        signal, sr = ass.wavread(inputFileName)
        frames = ass.get_signal_frames(frame_duration_ms)

        # Нахождение участков с речью
        vad = VoiceActivityDetector(inputFileName, frame_duration_ms)
        detected_windows = vad.detect_speech()
        speech_timestamps = vad.get_timestamps(detected_windows)

        # Выбор фреймов исходного сигнала в которых отсутствует речь
        noise_frames = []
        noise_start = 0
        for timestamp in speech_timestamps:
            speech_start = timestamp["speech_begin"]
            speech_end = timestamp["speech_end"]
            for frame in ass.get_frames_from_interval(noise_start, speech_start):
                noise_frames.append(frame)
            noise_start = speech_end

        # Запись выделенного образца шума в файл
        wavwrite_segment(Segment(noise_frames), noiseFileName)

        # Вычетание шума из исходного сигнала на основе выделенного образца шума
        clean_signal = filter_noise(inputFileName, noiseFileName, visual=self.GraphsOut.isChecked())

        # Запись очищенного сигнала в файл
        wavfile.write(outputFileName, sr, clean_signal)

        dialog = QMessageBox(self)
        final_message = "Фильтрация звука прошла успешно. Результат записан в " + outputFileName
        dialog.setText(final_message)
        dialog.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
