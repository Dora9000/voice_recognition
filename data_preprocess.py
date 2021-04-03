import collections
import contextlib
import librosa
import numpy as np
import python_speech_features
import wave


def extract_features(y, sr, n_mfcc, window=0.032, hop=0.01):
    mfcc = python_speech_features.mfcc(signal=y, samplerate=sr, numcep=n_mfcc)# , winstep=hop,winlen=window)
    mfcc = mfcc.T
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    stacked = np.vstack((mfcc, mfcc_delta, mfcc_delta2))
    return stacked.T


def normalize(feature, eps=1e-14):
    """ Center a feature using the mean and std
    Params:
        feature (numpy.ndarray): Feature to normalize
    """
    feats_mean = np.mean(feature, axis=0)
    feats_std = np.std(feature, axis=0)
    return (feature - feats_mean) / (feats_std + eps)



def write_wave(path, audio, sample_rate):
    # code modified for compactness
    # original code https://github.com/wiseman/py-webrtcvad/blob/master/example.py
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])
