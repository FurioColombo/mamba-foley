import os
import json
from pathlib import Path

import torch
import numpy as np
import torchaudio

import pydub
import soundfile as sf
from scipy.io.wavfile import write

from modules.model.model import UNet
from modules.model.sampler import SDESampling_batch
from modules.model.sde import VpSdeCos
from modules.utils.data_sources import dataset_from_path
from modules.utils.audio import adjust_audio_length, get_event_cond, high_pass_filter, resample_audio
from modules.utils.utilities import normalize, check_RAM_usage
from config.config import Config
from utils.file_system import ProjectPaths


def load_ema_weights(model, model_path):
    checkpoint = torch.load(model_path)
    dic_ema = {}
    for (key, tensor) in zip(checkpoint['model'].keys(), checkpoint['ema_weights']):
        dic_ema[key] = tensor
    model.load_state_dict(dic_ema)
    return model

def save_samples(samples, output_dir, sr, class_name:str, starting_idx:int=0, stereo:bool=False, target_audio=None, is_ground_truth:bool=False):
    assert len(samples.shape) == 2, f"ERROR: did not receive an array of samples: received tensor shape: {samples.shape}"

    for j in range(samples.shape[0]):
        sample = samples[j].cpu()
        sample = high_pass_filter(sample)
        filename = f"{class_name}_gt_{str(j + 1 + starting_idx).zfill(3)}.wav" \
            if is_ground_truth \
            else f"{class_name}_{str(j + 1 + starting_idx).zfill(3)}.wav"

        if stereo is True:
            assert target_audio is not None, "Target audio is required for stereo output."
            left_audio = target_audio.cpu().numpy()
            right_audio = sample.copy()
            assert len(left_audio) == len(right_audio), "Length of target and generated audio must be the same."

            sf.write('temp_left.wav', left_audio, 22050, 'PCM_24')
            sf.write('temp_right.wav', right_audio, 22050, 'PCM_24')

            left_audio = pydub.AudioSegment.from_wav('temp_left.wav')
            right_audio = pydub.AudioSegment.from_wav('temp_right.wav')

            if left_audio.sample_width > 4:
                left_audio = left_audio.set_sample_width(4)
            if right_audio.sample_width > 4:
                right_audio = right_audio.set_sample_width(4)

            # pan the sound
            left_audio_panned = left_audio.pan(-1.)
            right_audio_panned = right_audio.pan(+1.)

            mixed = left_audio_panned.overlay(right_audio_panned)
            mixed.export(f"{output_dir}/{class_name}_{str(j + 1).zfill(3)}_stereo.wav", format='wav')

            # remove temp files
            os.remove('temp_left.wav')
            os.remove('temp_right.wav')
        else:
            write(os.path.join(output_dir, filename), sr, sample)

def measure_el1_distance(sample, target, event_type):
    sample = normalize(sample).cpu()
    target = normalize(target).cpu()

    sample_event = get_event_cond(sample, event_type)
    target_event = get_event_cond(target, event_type)

    # sample_event = pooling(sample_event, block_num=49)
    # target_event = pooling(target_event, block_num=49)

    loss_fn = torch.nn.L1Loss()
    loss = loss_fn(sample_event, target_event)
    return loss.cpu().item()


class SampleGenerator:
    def __init__(self, config, labels:list, n_gen_samples_per_class, results_dir, model_param_path, device, conditioning_audio_path = None, stereo:bool = False, save_conditioning=True, cond_type: str='rms'):
        self.config = config
        self.labels = labels
        self.device = device
        self.n_gen_samples_per_class = int(n_gen_samples_per_class)
        self.stereo = stereo
        self.results_dir = results_dir
        self.save_conditioning = save_conditioning
        self.cond_type = cond_type

        # load params
        self.model_params = Config.get_config(model_param_path)
        # config file compatibility setup
        self.config_cond = self.model_params.condition if hasattr(self.model_params, 'condition') else self.model_params
        self.config_data = self.model_params.data if hasattr(self.model_params, 'data') else self.model_params
        self.config_model = self.model_params.model if  hasattr(self.model_params, 'model') else self.model_params

        self.sample_rate = self.config_data.sample_rate
        self.audio_length = self.sample_rate * 4
        self.event_conditioning = self.config_cond.event_type

        self.target_audio = None
        self.target_event = None
        self.update_conditioning(conditioning_audio_path)

        self.test_set = None

    def make_inference(self, class_names, gen_all_classes:bool, checkpoint_path: str or Path, samples_per_temporal_cond:int=1, cond_scale:int=3, same_class_conditioning=False, target_audio_path=False):
        self.update_conditioning(target_audio_path)

        class_names = class_names if type(class_names) is list else [class_names, ]

        # load model
        model = UNet(len(self.labels), self.model_params).to(self.device)
        self.model = load_ema_weights(model, os.path.abspath(checkpoint_path))
        sde = VpSdeCos()

        sampler = SDESampling_batch(model, sde, batch_size=self.n_gen_samples_per_class , device=self.device)
        # Generate N samples
        if gen_all_classes is True:
            class_indices = range(len(self.labels))
        else:
            class_indices = [i for i, label in enumerate(self.labels) if label in class_names]

        if same_class_conditioning:
            test_cond_dirs = self.config_data.test_cond_dirs
            self.test_set = dataset_from_path(self.config_data.test_dirs, self.model_params, self.labels, cond_dirs=test_cond_dirs)

        self._generate_samples(
            class_indices = class_indices,
            sampler = sampler,
            cond_scale = int(cond_scale),
            samples_per_temporal_cond=int(samples_per_temporal_cond),
            same_class_conditioning=same_class_conditioning
        )
        print('Done!')

    def _generate_samples(self, class_indices:list, sampler, cond_scale:int=3, samples_per_temporal_cond:int=1, same_class_conditioning:bool=False, target_audio_path=None):
        assert (same_class_conditioning and target_audio_path is not None) is False
        def _compute_conditioning(class_idx: int):
            # If an audio file is provided for conditioning
            if target_audio_path is not None:
                self.target_audio, sr = torchaudio.load(target_audio_path)
                if sr != self.sample_rate:
                    self.target_audio = resample_audio(self.target_audio, sr, self.sample_rate)
                self.target_audio = adjust_audio_length(self.target_audio, self.audio_length)
                self.target_event = get_event_cond(self.target_audio, self.cond_type)

            #  If conditioning needs to be automatically extracted randomly from eval dataset
            elif same_class_conditioning:
                target_audio_dict = self.test_set.dataset.get_random_sample_from_class(class_idx)
                self.target_audio = target_audio_dict['audio']
                self.target_event = target_audio_dict["event"].unsqueeze(0).to(self.device)

        def _generate_conditioned_samples(target_event, class_idx, num_samples):
            print(f"Generating {num_samples} samples of class \'{self.labels[class_idx]}\'...")
            noise = torch.randn(num_samples, self.audio_length, device=self.device)
            classes = torch.tensor([class_idx] * num_samples, device=self.device)
            sampler.batch_size = num_samples
            samples = sampler.predict(noise, 100, classes, target_event, cond_scale=cond_scale)
            return samples

        def _save_samples(samples, out_dir, class_name=None, starting_gen_idx: int=0, is_ground_truth: bool=False):
            # expand dimensions if a single sample is passed as argument
            if len(samples.shape) == 1:
                samples = samples[None, :]
            # save samples
            save_samples(
                samples=samples,
                output_dir=out_dir,
                sr=self.sample_rate,
                class_name=class_name,
                starting_idx=starting_gen_idx,
                stereo=self.stereo,
                target_audio=self.target_audio,
                is_ground_truth=is_ground_truth
            )

        for class_idx in class_indices:
            # utils
            class_name = self.labels[class_idx]

            # create folders and paths
            # create class/category directory
            class_dir = os.path.join(self.results_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # create corresponding ground truth directory
            ground_truth_dir = os.path.join(class_dir, 'ground_truths')
            os.makedirs(ground_truth_dir, exist_ok=True)

            # handle max number of samples generated in a batch (and also with same conditioning)
            max_samples_batch = int(samples_per_temporal_cond)
            computed_samples = 0

            while self.n_gen_samples_per_class - computed_samples > 0:

                # sanity checks
                check_RAM_usage(self.config)

                samples_to_generate = min(self.n_gen_samples_per_class - computed_samples, max_samples_batch)

                # generated samples
                _compute_conditioning(class_idx)

                # save ground truths
                if (target_audio_path is not None or self.target_audio is not None) and self.save_conditioning :
                    print(self.target_audio.shape)
                    _save_samples(
                        samples=self.target_audio,
                        out_dir=ground_truth_dir,
                        class_name=class_name,
                        starting_gen_idx=computed_samples,
                        is_ground_truth=True
                    )

                # create and save generated samples
                class_samples = _generate_conditioned_samples(self.target_event, class_idx, samples_to_generate)
                _save_samples(
                    samples=class_samples,
                    out_dir=class_dir,
                    class_name=class_name,
                    starting_gen_idx=computed_samples,
                    is_ground_truth=False
                )

                computed_samples += samples_to_generate
                print(f'computed {computed_samples}/{self.n_gen_samples_per_class} samples from {class_name} class\n')

    def update_conditioning(self, cond_audio_path):
        # Prepare target audio for conditioning (if exist)
        if cond_audio_path is not None and os.path.isfile(cond_audio_path):
            print('conditioning audio path:', cond_audio_path)
            target_audio, sr = torchaudio.load(cond_audio_path)
            if sr != self.sample_rate:
                target_audio = resample_audio(target_audio, sr, self.sample_rate)
            self.target_audio = adjust_audio_length(target_audio, self.audio_length)
            self.target_event = get_event_cond(target_audio, self.model_params.event_type)
            self.target_event = self.target_event.repeat(self.n_gen_samples_per_class, 1).to(self.device)

    def remove_conditioning(self):
        self.target_audio = None
        self.target_event = None

    def compute_out_dir(self, results_dir, checkpoint_path=None, category: str=None):
        # results directory
        results_dir = os.path.abspath(results_dir)
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
        directory = results_dir

        # epoch inference directory
        if checkpoint_path is not None:
            epoch_results_dir_name = checkpoint_path.split('.')[0].split('/')[-1]
            directory = os.path.join(results_dir, epoch_results_dir_name)
            if not os.path.isdir(directory):
                os.mkdir(directory)

        # category inference directory
        if category is not None:
            directory = os.path.join(directory, category)
            if not os.path.isdir(directory):
                os.mkdir(directory)
        return  directory
