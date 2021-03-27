import collections
import contextlib
import librosa
import numpy as np
import os
import pickle
import python_speech_features
import wave
import webrtcvad

from model import map_adaptation
from data_preprocess import *
from settings import *

y = []

if LOAD_SIGNAL:
    y, sr = librosa.load(data_name, sr=SR)
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])


if REMOVE_PAUSE:
    vad = webrtcvad.Vad(3)
    audio = np.int16(y / np.max(np.abs(y)) * 32768)

    frames = frame_generator(10, audio, sr)
    frames = list(frames)
    segments = vad_collector(sr, 50, 200, vad, frames)

    if not os.path.exists(abs_path + '/data/chunks'):
        os.makedirs(abs_path + '/data/chunks')

    full_chunk = []
    for i, segment in enumerate(segments):
        # chunk_name = abs_path + '/data/chunks/chunk-%003d.wav' % (i,)
        # write_wave(chunk_name, segment[0: len(segment)-int(100*sr/1000)], sr)
        full_chunk.append(segment[0: len(segment) - int(100 * sr / 1000)])
    chunk_name = abs_path + '/data/chunks/full_chunk.wav'
    write_wave(chunk_name, b''.join([full_chunk[i] for i in range(len(full_chunk))]), sr)


if FEATURES_FROM_FILE:
    ubm_features = pickle.load(open(feature_name, 'rb'))
else:
    #y, sr = librosa.load(data_name, sr=None)
    f = extract_features(np.array(y), sr, n_mfcc=N_MFCC, hop=HOP_LENGTH, window=N_FFT)
    ubm_features = normalize(f)
    pickle.dump(ubm_features, open(feature_name, "wb"))


# get big data
# preprocess
# train ubm
# get one data
# preprocess
# try to predict