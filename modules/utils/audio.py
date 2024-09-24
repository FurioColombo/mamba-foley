import os.path

import librosa
import numpy as np
import torch

import torchaudio
from torchaudio.transforms import Resample
from scipy.signal import ellip, filtfilt, firwin, lfilter
from tqdm import tqdm


# --- Event Processing ---
def get_event_cond(x, event_type='rms'):
    assert event_type in ['rms', 'power', 'onset']
    if event_type == 'rms':
        return get_rms(x)
    if event_type == 'power':
        return get_power(x)
    if event_type == 'onset':
        return get_onset(x)

def get_rms(signal):
    rms = librosa.feature.rms(y=signal, frame_length=512, hop_length=128)
    rms = rms[0]
    rms = zero_phased_filter(rms)
    return torch.tensor(rms.copy(), dtype=torch.float32)


def get_power(signal):
    if torch.is_tensor(signal):
        signal_copy_grad = signal.clone().detach().requires_grad_(signal.requires_grad)
        return signal_copy_grad * signal_copy_grad
    else:
        return torch.tensor(signal * signal, dtype=torch.float32)


def get_onset(y, sr=22050):
    y = np.array(y)
    o_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median, fmax=8000, n_mels=256)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, normalize=True, delta=0.3, units='samples')
    onsets = np.zeros(y.shape)
    onsets[onset_frames] = 1.0
    return torch.tensor(onsets, dtype=torch.float32)


# =========== Resampling ===========

def resample_audio(audio, original_sr, target_sr, device='cpu'):
    resampler = Resample(original_sr, target_sr, resampling_method='sinc_interp_hann')
    resampler.to(device)
    return resampler(audio)


def resample_wav_files_in_folder(folder_path, target_sr, output_folder=None):
    assert os.path.isdir(folder_path)
    # Check if the output folder path is provided
    if output_folder is None:
        output_folder = folder_path
    # Check if the output folder exists, if not, create it
    elif not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the folder
    for file_name in tqdm(os.listdir(folder_path), desc=f'resampling all .wav in {os.path.relpath(folder_path)}'):
        file_path = os.path.join(folder_path, file_name)
        # Check if the file is a .wav file
        if file_name.endswith('.wav'):
            # Load the audio
            audio, sr = torchaudio.load(file_path)
            # Resample the audio
            resampled_audio = resample_audio(audio, sr, target_sr)
            # Save the resampled audio with the same filename
            output_file_path = os.path.join(output_folder, file_name)
            torchaudio.save(output_file_path, resampled_audio, sample_rate=target_sr)

def adjust_audio_length(audio, length):
    if audio.shape[1] >= length:
        return audio[0, :length]
    return torch.cat((audio[0, :], torch.zeros(length - audio.shape[1])), dim=-1)


# ================ Audio Filtering ================
def high_pass_filter(x, sr=22050):
    b = firwin(101, cutoff=20, fs=sr, pass_zero='highpass')
    x = lfilter(b, [1, 0], x)
    return x

def zero_phased_filter(x):
    b, a = ellip(4, 0.01, 120, 0.125)
    x = filtfilt(b, a, x, method="gust")
    return x
