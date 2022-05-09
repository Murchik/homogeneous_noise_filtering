import collections
import webrtcvad


class VoiceSegment(object):
    _frames: list
    start_frame_num: int
    end_frame_num: int

    def __init__(self):
        self._frames = []
        self.start_frame_num = 0
        self.end_frame_num = 0

    def set_frames(self, frames):
        self._frames = frames

    def get_frames(self):
        return self._frames

    def get_bytes(self):
        """ Объединение семплов всех фреймов в один контейнер bytes """
        return b''.join([frame.get_bytes() for frame in self._frames if frame])

    def get_signal(self):
        """ Объединение семплов всех фреймов в один лист семплов """
        signal = []
        for frame in self._frames:
            for sample in frame.samples:
                signal.append(sample)
        return signal


class VoiceDetector:
    sample_rate: int
    frame_duration_ms: int
    padding_duration_ms: int
    vad: webrtcvad.Vad
    voice_frames: list

    def __init__(self, sample_rate, frame_duration_ms, padding_duration_ms, mode):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.vad = webrtcvad.Vad(mode)

    def get_voice_segments_gen(self, frames):
        num_padding_frames = int(
            self.padding_duration_ms / self.frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)

        self.voice_frames = []
        triggered = False
        segment = VoiceSegment()

        for frame in frames:
            is_speech = self.vad.is_speech(frame.get_bytes(), self.sample_rate)
            # '1' if is_speech else '0'
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    segment.start_frame_num = ring_buffer[0][0].num
                    for f, s in ring_buffer:
                        self.voice_frames.append(f)
                    ring_buffer.clear()
            else:
                self.voice_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len(
                    [f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    segment.end_frame_num = frame.num
                    segment.set_frames(self.voice_frames)
                    triggered = False
                    yield segment
                    ring_buffer.clear()
                    # self.voice_frames = []
        if triggered:
            segment.end_frame_num = frame.num
        if self.voice_frames:
            segment.set_frames(self.voice_frames)
            yield segment
