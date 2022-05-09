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
frame_duration = 10
frames = af.getframes(frame_duration)

# Выбрать группы фреймов (сегменты) с речью
vd = VoiceDetector(sr, frame_duration, 300, 3)
segments = vd.get_voice_segments_gen(frames)

# Выбрать сегменты, в которых отсутствует речь
noise_frames = []
num = 0
widening = 5
for i, segment in enumerate(segments):
    start = segment.start_frame_num - widening
    for frame in frames[num:start]:
        noise_frames.append(frame)
    num = segment.end_frame_num + widening

# Выделить из сегментов без речи общие черты (шум)
# TODO

# Вычесть из всего трека шум
# TODO

seg = vs()

# Запись речи в файл
seg._frames = vd.voice_frames
write_wave('out_voice.wav', vs.get_bytes(seg), sr)

# Запись шума в файл
seg._frames = noise_frames
write_wave('out_noise.wav', vs.get_bytes(seg), sr)
