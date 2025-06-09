# utils/preprocessing.py
import librosa
import numpy as np

def extract_features(audio_path, sr=16000, n_mfcc=40):
    y, _ = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)
