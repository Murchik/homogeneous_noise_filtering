import sys
from scipy.io import wavfile
from PyQt6.QtWidgets import QMainWindow, QMessageBox, QApplication, QFileDialog

from AudioSignal import AudioSignal
from NoiseRemover import NoiseRemover
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
        file_name = QFileDialog.getOpenFileName(
            self, message, directory, files)
        self.InputVoiceFile.setText(file_name[0])

    def browseNoiseFile(self):
        message = 'Select Noise sample destination'
        directory = 'out'
        files = '*.wav'
        file_name = QFileDialog.getSaveFileName(
            self, message, directory, files)
        self.NoiseSampleFile.setText(file_name[0])

    def browseOutFile(self):
        message = 'Select output audio file destination'
        directory = 'out'
        files = '*.wav'
        file_name = QFileDialog.getSaveFileName(
            self, message, directory, files)
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

        ass = AudioSignal(inputFileName)
        sr = ass.get_sample_rate()

        nr = NoiseRemover()

        clean_signal, noise_sample = nr.removeNoise(
            ass, visual=self.GraphsOut.isChecked())

        wavfile.write(noiseFileName, sr, noise_sample)

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
