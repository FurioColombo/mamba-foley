import logging
import random
import csv
import os

import numpy as np
import torch
import torchaudio

from torch.utils.data import Dataset

def moving_avg(input, window_size):
    if type(input) != list: input = list(input)
    result = []
    for i in range(1, window_size+1):
        result.append(sum(input[:i])/i)

    moving_sum = sum(input[:window_size])
    result.append(moving_sum/window_size)
    for i in range(len(input) - window_size):
        moving_sum += (input[i+window_size] - input[i])
        result.append(moving_sum/window_size)
    return np.array(result)

def parse_filelist(filelist_path):
    # if filelist_path is txt file
    if filelist_path.endswith('.txt'):
        with open(filelist_path, 'r') as f:
            filelist = [line.strip() for line in f.readlines()]
        return filelist

    # if filelist_path is csv file
    if filelist_path.endswith('.csv'):
        with open(filelist_path, 'r') as f:
            reader = csv.reader(f)
            filelist = [row[0] for row in reader]
            f.close()
        return filelist

class AbstractAudioDataset(Dataset):
    def __init__(self, params, labels):
        super().__init__()
        # config file compatibility setup
        self.data_params = params.data if hasattr(params, 'data') else params
        self.condition_params = params.condition if hasattr(params, 'condition') else params

        self.audio_length = self.data_params.audio_length
        self.sample_rate = self.data_params.sample_rate
        self.labels = labels

        self.event_type = self.condition_params.event_type

    def __len__(self):
        raise NotImplementedError("AbstractAudioDataset is an abstract class.")

    def __getitem__(self, idx):
        raise NotImplementedError("AbstractAudioDataset is an abstract class.")

    def _load_audio(self, audio_filename):
        signal, _ = torchaudio.load(audio_filename)
        signal = signal[0, :self.audio_length]
        return signal

    def _extract_class_cond(self, audio_filename):
        cls_name = os.path.dirname(audio_filename).split('/')[-1]
        cls = torch.tensor(self.labels.index(cls_name))
        return cls

    def get_random_sample(self):
        idx = random.randint(0, len(self) - 1)
        return self.__getitem__(idx)

    def get_random_sample_from_class(self, class_idx):
        random_sample = self.get_random_sample()
        logging.debug(random_sample['class'].item(), ' - ', class_idx)
        while random_sample['class'].item() is not class_idx:
            random_sample = self.get_random_sample()
            logging.debug(random_sample['class'].item(), ' - ', class_idx)
        return random_sample
