import sys
import contextlib
import wave
import collections
import webrtcvad
import numpy as np

import plot


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class AudioFile:
    num_channels: int
    bitrate: int
    sample_rate: int
    num_samples: int
    duration: float
    audio_data: list

    def wavread(self, path):
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            self.num_channels = wf.getnchannels()
            assert self.num_channels == 1
            self.bitrate = wf.getsampwidth()
            assert self.bitrate == 2
            self.sample_rate = wf.getframerate()
            assert self.sample_rate in (8000, 16000, 32000, 48000)
            self.num_samples = wf.getnframes()
            audio_bytes = wf.readframes(self.num_samples)
            # print(len(audio_bytes), type(audio_bytes), type(audio_bytes[0]))
            self.audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            self.duration = len(self.audio_data) / self.sample_rate
            # print(len(self.audio_data), self.duration)
            return self.audio_data, self.sample_rate

    def write_wave(self, path):
        with contextlib.closing(wave.open(path, 'wb')) as wf:
            wf.setnchannels(self.num_channels)
            wf.setsampwidth(self.bitrate)
            wf.setframerate(self.sample_rate)
            wf.writeframes(self.audio_data)

    def getframes(self, frame_duration_ms):
        audio_bytes = np.ndarray.tobytes(self.audio_data)
        n = int(self.sample_rate * (frame_duration_ms / 1000.0) * self.bitrate)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / self.sample_rate) / 2.0
        frames = []
        while offset + n < len(audio_bytes):
            frames.append(
                Frame(audio_bytes[offset:offset + n], timestamp, duration))
            timestamp += duration
            offset += n
        return frames


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


def main(args):
    af = AudioFile()
    audio, sr = af.wavread("audio/test_long_32kHz.wav")

    plot.signal(audio, sr)
    plot.magnitude_spectrum(audio, sr)
    # plot.spectrogram(audio, sr)

    frame_duration = 30
    frames = af.getframes(frame_duration)

    vad = webrtcvad.Vad(int(3))
    segments = vad_collector(sr, frame_duration, 300, vad, frames)
    for i, segment in enumerate(segments):
        path = 'chunk-%002d.wav' % (i,)
        print(' Writing %s' % (path,))
        write_wave(path, segment, sr)


if __name__ == '__main__':
    main(sys.argv[1:])
