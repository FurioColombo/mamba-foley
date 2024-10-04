from types import SimpleNamespace
import matplotlib.pyplot as plt
import psutil
import json

from torchaudio.transforms import MelSpectrogram
import numpy as np
import torch

from modules.utils import notifications
from modules.utils.file_system import ProjectPaths


# --- Normalizations and Mappings ---
def normalize(x):
    return x / torch.max(torch.abs(x)).item()


def normalize_to_range(tensor, min_val, max_val):
    # Find min and max values in the tensor
    min_tensor = torch.min(tensor)
    max_tensor = torch.max(tensor)

    # Scale values to the range [0, 1]
    normalized_tensor = (tensor - min_tensor) / (max_tensor - min_tensor)

    # Scale values to the desired range [min_val, max_val]
    scaled_tensor = normalized_tensor * (max_val - min_val) + min_val
    return scaled_tensor

def map_to_range(value, min_val, max_val, new_min, new_max):
    normalized_value = (value - min_val) / (max_val - min_val)
    scaled_value = normalized_value * (new_max - new_min) + new_min
    return scaled_value

def pooling(x, block_num=49):
    block_size = x.shape[-1] // block_num
    
    device = x.device
    pooling = torch.nn.MaxPool1d(block_size, stride=block_size)
    x = x.unsqueeze(1)
    pooled_x = pooling(x).to(device)
    
    return pooled_x


# --- Plot ---
def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


# =========== Plotting ===========
def plot_spec(waveform, sample_rate):
    # Transform to mel-spec
    transform = MelSpectrogram(sample_rate)
    mel_spec = transform(waveform)
    
    # Plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(mel_spec)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    # Turn into numpy format to upload to tensorboard
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def plot_env(waveform):
    # Plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    plt.plot(waveform)
    plt.tight_layout()
    
    # Turn into numpy format to upload to tensorboard
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


# =========== Common Sanity Checks ==========

def check_nan(t:torch.Tensor, error_msg:str):
    if torch.isnan(t).any():
        raise RuntimeError(error_msg)


def check_RAM_usage(config, callback=lambda: None):
    max_percentage = config.max_RAM_usage
    if max_percentage is None:
        config = load_json_config(ProjectPaths.config_file)
        max_percentage = config.max_RAM_usage

    assert 100 >= max_percentage >= 0
    ram_usage = psutil.virtual_memory().percent

    if ram_usage > max_percentage:
        callback()
        notification = f'TRAINING INTERRUPTED\nThreshold ram_usage exceeded:{ram_usage}%'
        notifications.notify_telegram(notification, config)
        raise MemoryError('Threshold ram_usage exceeded:', ram_usage, '%')

# ======= Config Utils =======
def dict_to_namespace(d):
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_namespace(value)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

def load_json_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return dict_to_namespace(config_dict)

# Define modular functions to extract specific sections
def get_training_config(config):
    return config.training

def get_model_config(config):
    return config.model

def str2bool(val: str or bool):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    if type(val) is bool:
        return val
    assert type(val) is str
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))