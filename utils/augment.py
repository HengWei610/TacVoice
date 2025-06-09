# utils/augment.py
import numpy as np
import librosa

def add_noise(signal, noise_level=0.005):
    noise = np.random.randn(len(signal))
    return signal + noise_level * noise

def time_shift(signal, shift_max=0.2):
    shift = np.random.randint(int(len(signal) * shift_max))
    return np.roll(signal, shift)
