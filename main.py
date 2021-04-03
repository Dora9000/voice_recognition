import os
import shutil
import pickle
import webrtcvad
from sklearn.mixture import GaussianMixture
from model import map_adaptation
from data_preprocess import *
from settings import *
import copy
import joblib
from pydub import AudioSegment
import time

RESULTS_PATH = '../UIR3/data'


def get_data(file, result_file=RESULTS_PATH + '/chunks/full_chunk.wav', all_data=True):
    y, sr = librosa.load(file, sr=SR)
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    vad = webrtcvad.Vad(3)
    audio = np.int16(y / np.max(np.abs(y)) * 32768)

    frames = frame_generator(10, audio, sr)
    frames = list(frames)
    segments = vad_collector(sr, 50, 200, vad, frames)

    full_chunk = []
    if all_data:  # all data
        for i, segment in enumerate(segments):
            full_chunk.append(segment[0: len(segment) - int(100 * sr / 1000)])
        write_wave(result_file, b''.join([full_chunk[i] for i in range(len(full_chunk))]), sr)
    else:  # part of 35000
        for i, segment in enumerate(segments):
            full_chunk.append(segment[0: min(35000, len(segment) - int(100 * sr / 1000))])
        a = 0
        s = 0
        for i in range(len(full_chunk)):
            if a < len(full_chunk[i]):
                a = len(full_chunk[i])
                s = i
        write_wave(result_file, b''.join([full_chunk[s]]), sr)
    return result_file


def get_features(file, features_to_file, features_from_file=None, mfcc=None):
    if features_from_file is not None:
        ubm_features = pickle.load(open(features_from_file, 'rb'))
    else:
        y, sr = librosa.load(file, sr=None)
        f = extract_features(np.array(y), sr, n_mfcc=N_MFCC if mfcc is None else mfcc, hop=HOP_LENGTH, window=N_FFT)
        ubm_features = normalize(f)
        pickle.dump(ubm_features, open(features_to_file, "wb"))
    return ubm_features


def get_ubm(ubm_features, ubm_file=None, ubm_to_file=None):
    if ubm_file is not None:
        ubm = joblib.load(ubm_file)
    else:
        ubm = GaussianMixture(n_components=N_COMPONENTS, covariance_type=COVARINACE_TYPE)
        ubm.fit(ubm_features)
        joblib.dump(ubm, ubm_to_file)
    return ubm


def get_gmm(ubm, gmm_features, gmm_to_file, gmm_file=None):
    if gmm_file is not None:
        gmm = joblib.load(gmm_file)
    else:
        gmm = copy.deepcopy(ubm)
        gmm = map_adaptation(gmm, gmm_features, max_iterations=1, relevance_factor=16)
        joblib.dump(gmm, gmm_to_file)
    return gmm


def clear_dir(path):
    shutil.rmtree(path)
    os.mkdir(path, mode=0o777, dir_fd=None)


def train_ubm():
    DATA_PATH = '../LibriSpeech'
    features = []
    max_cnt = {}
    for root, dirs, files in os.walk(DATA_PATH):
        for fn in files:
            if fn[-4:] == '.wav':
                s = fn[:-4].split('-')
                s = s[0]
                if max_cnt.get(s, None) is not None and max_cnt.get(s, None) == 1:
                    continue
                max_cnt[s] = 1 if max_cnt.get(s, None) is None else max_cnt[s] + 1
                data = get_data(file=root + '/' + fn,
                                result_file=RESULTS_PATH + '/chunks/' + fn, all_data=True)
                feature = get_features(file=data,
                                       features_to_file=RESULTS_PATH + '/chunks/' + fn[-4:] + '.pkl')
                features.append(feature)
    features_ = None
    for i in range(len(features)):
        if i == 0:
            features_ = features[i]
        else:
            features_ = np.vstack((features_, np.array(features[i], dtype="float16")))
    return features_


TRAIN_UBM = False
if TRAIN_UBM:
    clear_dir(RESULTS_PATH + '/chunks')
    mfcc = [8, 13, 20]
    for N_MFCC in mfcc:
        features = train_ubm()
        n_comp = [8, 16, 32]
        cov = ['tied', 'diag', 'full']
        for N_COMPONENTS in n_comp:
            for COVARINACE_TYPE in cov:
                get_ubm(features,
                        ubm_to_file=RESULTS_PATH + '/ubm_models/ubm_{0}_{1}_{2}mfcc.pkl'.format(N_COMPONENTS,
                                                                                                COVARINACE_TYPE,
                                                                                                N_MFCC))


def get_recording(path, path_to_save, remember=True):
    infiles = []
    for root, dirs, files in os.walk(path):
        for fn in files:
            if fn[-4:] == '.wav':
                infiles.append(root + '/' + fn)
    sound1 = AudioSegment.from_wav(infiles[0])
    for i in range(1, len(infiles)):
        sound2 = AudioSegment.from_wav(infiles[i])
        sound1 = sound1 + sound2
    if remember:
        sound1 = sound1[0:DURATION_TO_REMEMBER * 1000]
    else:
        step = 4
        assert (DURATION_TO_REMEMBER * 1000 <
                DURATION_TO_REMEMBER * 1000 + min(DURATION_TO_RECOGNIZE * 1000, len(sound1) - 1))
        assert (step * DURATION_TO_REMEMBER * 1000 + min(DURATION_TO_RECOGNIZE * 1000, len(sound1) - 1) <
                len(sound1) - 1)
        sound1 = sound1[DURATION_TO_REMEMBER * 1000:
                        step * DURATION_TO_REMEMBER * 1000 + min(DURATION_TO_RECOGNIZE * 1000, len(sound1) - 1)]
    sound1.export(path_to_save, format="wav")
    return path_to_save


def parse_params(model_name):
    s = model_name[-6:-4]
    if s[0] == '_':
        s = s[1]
    return int(s)


def get_speaker(id, mfcc, remember):
    DATA_PATH = '../LibriSpeech/dev-clean/{0}'.format(id)
    data = get_data(file=get_recording(DATA_PATH,
                                       remember=remember,
                                       path_to_save=RESULTS_PATH + "/gmm/speaker.wav"),
                    result_file=RESULTS_PATH + '/gmm/chunks/' + str(id) + '.wav',
                    all_data=True)
    feature = get_features(file=data,
                           features_to_file=RESULTS_PATH + '/gmm/chunks/' + str(id) + '.pkl', mfcc=mfcc)
    return feature


def create_gmms(cur_ubm):
    ubm_name = RESULTS_PATH + '/ubm_models/' + cur_ubm
    ids = os.listdir('../LibriSpeech/dev-clean/')
    for id in ids:
        gmm_features = np.asarray(get_speaker(id, parse_params(cur_ubm[:-4]), remember=True))
        ubm = get_ubm(None, ubm_file=ubm_name)
        if not os.path.exists(RESULTS_PATH + '/gmm_models/{0}'.format(cur_ubm[:-4])):
            os.makedirs(RESULTS_PATH + '/gmm_models/{0}'.format(cur_ubm[:-4]))
        gmm = get_gmm(ubm, gmm_features,
                      gmm_to_file=RESULTS_PATH + '/gmm_models/{0}/gmm_{1}_speaker_{2}.pkl'.format(
                          cur_ubm[:-4],
                          cur_ubm[:-4],
                          id))
    return RESULTS_PATH + '/gmm_models/{0}'.format(cur_ubm[:-4])


def test_gmms(path, mfcc):
    ids = os.listdir('../LibriSpeech/dev-clean/')
    models = [(joblib.load(path + '/' + user), user) for user in os.listdir(path)]
    statistics = {}

    for _, c in models:
        statistics[str(c.split("_")[-1][:-4])] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    times = []
    for id in ids:
        start_time = time.time()
        gmm_features = np.asarray(get_speaker(id, mfcc, remember=False))
        answer_speaker = 0
        answer_score = -1e9
        for model, name in models:
            s = np.array(model.score(gmm_features))
            if answer_score < s:
                answer_score = s
                answer_speaker = name
        answer_speaker = str(answer_speaker.split("_")[-1][:-4])
        times.append(time.time() - start_time)
        if answer_speaker == str(id):
            statistics[answer_speaker]['TP'] += 1
            for c in statistics:
                if c == answer_speaker:
                    continue
                statistics[c]['TN'] += 1
        else:
            statistics[id]['FN'] += 1
            statistics[answer_speaker]['FP'] += 1
            for c in statistics:
                if c == answer_speaker or c == str(id):
                    continue
                statistics[c]['TN'] += 1
    print('---- time : ', sum(times) / len(times), min(times), max(times))
    answer = {'acc': [], 'prec': [], 'rec': []}
    for c in statistics:
        TP = statistics[c]['TP']
        TN = statistics[c]['TN']
        FP = statistics[c]['FP']
        FN = statistics[c]['FN']

        answer['acc'].append((TP + TN) / max(TP + TN + FP + FN, 1))
        answer['prec'].append(TP / max(TP + FP, 1))
        answer['rec'].append(TP / max(TP + FN, 1))
    acc = answer['acc']
    prec = answer['prec']
    rec = answer['rec']
    print('acc: ', sum(acc) / len(acc))
    print('prec: ', sum(prec) / len(prec))
    print('rec: ', sum(rec) / len(rec))


UBMS = os.listdir('../UIR3/data/ubm_models')
for CUR_MODEL in UBMS:
    gmm_path = RESULTS_PATH + '/gmm_models/{0}'.format(CUR_MODEL[:-4])
    print(CUR_MODEL)
    test_gmms(gmm_path, parse_params(CUR_MODEL[:-4]))
    print('=============')
