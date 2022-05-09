import sys
import wave
import contextlib

import plot

from AudioFile import AudioFile
from VoiceDetection import VoiceDetector
from VoiceDetection import VoiceSegment as vs


def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


af = AudioFile()

# Прочитать трек из файла
audio, sr = af.wavread("audio/test_long_32kHz.wav")

# Поделить трек на фреймы
frame_duration = 30
frames = af.getframes(frame_duration)

# Выбрать такие группы фреймов (сегменты), в которых отсутствует речь
# TODO

# Найти сегменты без речи
# TODO

# Выделить из сегментов без речи общие черты (шум)
# TODO

# Вычесть из всего трека шум
# TODO

# Записать трек в файл
af.wavwrite("out.wav")

# Нахождение сегментов с речью
vd = VoiceDetector(sr, frame_duration, 300, 3)
segments = vd.get_voice_segments_gen(frames)

# Запись сегментов с речью в отдельные файлы
for i, segment in enumerate(segments):
    plot.signal(vs.get_signal(segment), sr)
    path = 'out_chunk-%002d.wav' % (i,)
    print(' Writing %s' % (path,))
    write_wave(path, vs.get_bytes(segment), sr)
