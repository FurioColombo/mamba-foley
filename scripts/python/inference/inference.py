from pathlib import Path
from tqdm import tqdm
import subprocess
import logging
import os.path
import sys

import torch
import torchaudio as tAudio

root_folder_path = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], universal_newlines=True).strip())
modules_folder_path = os.path.join(root_folder_path, "modules")
sys.path.append(root_folder_path)
sys.path.append(modules_folder_path)
from modules.eval.sample_generator import SampleGenerator
from modules.utils.utilities import check_RAM_usage
from modules.utils.file_system import ProjectPaths
from config.config import Config
from modules.utils.notifications import notify_telegram

os.environ["CUDA_VISIBLE_DEVICES"] = Config.get_config(ProjectPaths.config_script_inference).CUDA_VISIBLE_DEVICES
LABELS = ['DogBark', 'Footstep', 'GunShot', 'Keyboard', 'MovingMotorVehicle', 'Rain', 'Sneeze_Cough']

def list_checkpoint_paths_in_dir(dir: str or Path):
    d = os.path.abspath(dir)
    files = [os.path.abspath(os.path.join(d, f)) for f in os.listdir(d) if
             os.path.isfile(os.path.join(d, f)) and f.split('.')[-1] == "pt"]
    return files


def prepare_machine():
    # Set model and sampler
    tAudio.set_audio_backend('sox_io')


def main(params):

    prepare_machine()
    device = torch.device('cuda')
    stereo = params.stereo if params is bool else params.stereo == 'True'

    os.makedirs(params.output_dir, exist_ok=True)

    # Set models
    if os.path.isfile(params.model_checkpoints_path):
        checkpoints_paths = [os.path.abspath(params.model_checkpoints_path)]
    elif os.path.isdir(params.model_checkpoints_path):
        checkpoints_paths = list_checkpoint_paths_in_dir(os.path.abspath(params.model_checkpoints_path))
    else:
        raise IsADirectoryError(f'checkpoint path {params.model_checkpoints_path} is invalid')

    # Set sampler
    gen = SampleGenerator(
        labels=LABELS,
        n_gen_samples_per_class=params.samples_per_class,
        model_param_path=params.model_config_path,
        results_dir=params.output_dir,
        stereo=stereo,
        device=device,
        cond_type='rms'
    )

    for path in tqdm(checkpoints_paths, desc=f"checkpoints inference"):
        check_RAM_usage(config.max_RAM_usage)
        gen.make_inference(
            class_names=config.class_names,
            gen_all_classes=config.gen_all_classes,
            checkpoint_path=path,
            samples_per_temporal_cond=config.samples_per_temporal_cond,
            cond_scale=config.cond_scale,
            same_class_conditioning=config.same_class_conditioning,
            target_audio_path=config.target_audio_path
        )

    notify_telegram(f'Finished script execution: {__file__}')


def validate_args(config):
    if config.gen_all_classes is False and config.class_names is None:
        assert config.class_names is not None, "if gen_all_classes is False, user should specify a class"
    if config.gen_all_classes is True and config.class_names is not None: # todo: debug this crap
        logging.warning('WARNING: generating all classes, specified ones will be ignored.')

    if config.target_audio_path is None and config.same_class_conditioning:
        logging.warning('WARNING: generating without explicit conditioning')

    # todo: control samples per temp cond < samples per class and division % ? 0


if __name__ == '__main__':
    config = Config.get_config(ProjectPaths.config_script_inference)
    validate_args(config)
    main(config)
