from AudioSignal import AudioSignal
import numpy as np
from VoiceActivityDetection import VoiceActivityDetector
from SpectralSubtraction import SpectralSubtractor
from Frame import Frame, _get_signal_from_frames


class NoiseRemover:
    def __init__(self, frame_duration_ms=20):
        self.frame_duration_ms = frame_duration_ms
        self.vad = VoiceActivityDetector(frame_duration_ms)
        self.ss = SpectralSubtractor()

    def removeNoise(self, audio_signal: AudioSignal, visual=False) -> 'np.ndarray[np.int16]':
        frames: list[Frame] = audio_signal.get_signal_frames(
            self.frame_duration_ms)
        sr = audio_signal.get_sample_rate()

        detected_windows = self.vad.detect_speech(frames, sr)
        speech_timestamps = self.vad.get_timestamps(detected_windows)
        noise_frames = self._get_noise_frames(audio_signal, speech_timestamps)

        clean_signal = self.ss.filter_noise(_get_signal_from_frames(frames, sr, self.frame_duration_ms),
                                            _get_signal_from_frames(
                                                noise_frames, sr, self.frame_duration_ms),
                                            visual=visual)

        noise_signal = _get_signal_from_frames(
            noise_frames, sr, self.frame_duration_ms)
        return clean_signal, noise_signal

    def _get_noise_frames(self, audio_signal: AudioSignal, speech_timestamps) -> list[Frame]:
        noise_frames = []
        noise_start = 0
        for timestamp in speech_timestamps:
            speech_start = timestamp["speech_begin"]
            speech_end = timestamp["speech_end"]
            for frame in audio_signal.get_frames_from_interval(noise_start, speech_start):
                noise_frames.append(frame)
            noise_start = speech_end
        return noise_frames
