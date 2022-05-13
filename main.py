import matplotlib.pyplot as plt
import numpy as np
import contextlib
import wave

from AudioSignal import Audio
from VoiceDetection import VoiceDetector


def wavwrite_segment(segment, path):
    segment_signal_bytes = segment.get_frames_bytes()
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(32000)
        wf.writeframes(segment_signal_bytes)


audio_sig = Audio()
_, sample_rate = audio_sig.wavread("audio/test_long_long_32kHz_loud.wav")

frame_duranion_ms = 30
frames = audio_sig.get_signal_frames(frame_duranion_ms)

vd = VoiceDetector(sample_rate, frame_duranion_ms)
voice_segments = vd.get_voice_segments(frames)
noise_segments = vd.get_silence_segments(frames)

i = 0
for segment in voice_segments:
    path = 'out_voice_chunk-%002d.wav' % (i,)
    print(' Writing %s' % (path,))
    wavwrite_segment(segment, path)
    i += 1


i = 0
for segment in noise_segments:
    path = 'out_noise_chunk-%002d.wav' % (i,)
    print(' Writing %s' % (path,))
    wavwrite_segment(segment, path)
    i += 1
