import collections

from webrtcvad import Vad

from Segment import Segment


class VoiceDetector:
    _sample_rate: int
    _frame_duration_ms: int
    _padding_duration_ms: int
    _mode: int
    __vad: Vad

    def __init__(self, sample_rate, frame_duration_ms, padding_duration_ms=300, mode=2):
        self._sample_rate = sample_rate
        self._frame_duration_ms = frame_duration_ms
        self._padding_duration_ms = padding_duration_ms
        self._mode = mode

        self.__vad = Vad(int(self._mode))

    def get_voice_segments(self, frames):
        num_padding_frames = int(
            self._padding_duration_ms / self._frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        voice_segments = []

        frames_with_voice = []
        for frame in frames:
            is_voice = self.__vad.is_speech(frame.get_samples_bytes(),
                                            self._sample_rate)
            ring_buffer.append((frame, is_voice))

            if not triggered:
                num_with_voice = len(
                    [frame for frame, is_voice in ring_buffer if is_voice])
                if num_with_voice > 0.3 * ring_buffer.maxlen:
                    triggered = True
                    for frame, is_voice in ring_buffer:
                        frames_with_voice.append(frame)
                    ring_buffer.clear()
            else:
                frames_with_voice.append(frame)
                num_without_voice = len(
                    [frame for frame, is_voice in ring_buffer if not is_voice])
                if num_without_voice > 0.3 * ring_buffer.maxlen:
                    triggered = False
                    voice_segments.append(Segment(frames_with_voice))
                    ring_buffer.clear()
                    frames_with_voice = []

        if frames_with_voice:
            voice_segments.append(Segment(frames_with_voice))
        return voice_segments

    def get_silence_segments(self, frames):
        num_padding_frames = int(
            self._padding_duration_ms / self._frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        silence_segments = []

        frames_with_silence = []
        for frame in frames:
            is_voice = self.__vad.is_speech(frame.get_samples_bytes(),
                                            self._sample_rate)
            ring_buffer.append((frame, is_voice))

            if not triggered:
                num_without_voice = len(
                    [frame for frame, is_voice in ring_buffer if not is_voice])
                if num_without_voice > 0.3 * ring_buffer.maxlen:
                    triggered = True
                    for frame, _ in ring_buffer:
                        frames_with_silence.append(frame)
                    ring_buffer.clear()
            else:
                frames_with_silence.append(frame)
                num_with_voice = len(
                    [frame for frame, is_voice in ring_buffer if is_voice])
                if num_with_voice > 0.3 * ring_buffer.maxlen:
                    triggered = False
                    silence_segments.append(Segment(frames_with_silence))
                    ring_buffer.clear()
                    frames_with_silence = []

        if frames_with_silence:
            silence_segments.append(Segment(frames_with_silence))
        return silence_segments
