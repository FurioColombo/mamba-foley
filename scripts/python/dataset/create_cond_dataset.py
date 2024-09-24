import torch
import torchaudio

from pathlib import Path
from tqdm import tqdm
import logging
import subprocess
import sys
import os

root_folder_path = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], universal_newlines=True).strip())
modules_folder_path = os.path.join(root_folder_path, "modules")
sys.path.append(root_folder_path)
sys.path.append(modules_folder_path)
from modules.data.dataset import parse_filelist
from modules.utils.utilities import load_json_config
from modules.utils.file_system import ProjectPaths
from modules.utils.audio import get_event_cond


def create_cond_files_from_txt(paths_files, params):
    condition_params = params.condition
    data_params = params.data

    # for each file containing paths:
    for paths_file in paths_files:
        paths_file = os.path.join(ProjectPaths.base_dir, paths_file)
        print('paths_file:', os.path.abspath(paths_file))
        audio_paths = parse_filelist(paths_file)

        # open/create txt file for storing paths
        txt_file_path, _ = os.path.splitext(os.path.normpath(paths_file))
        cond_txt_file_path = txt_file_path + f"_{condition_params.event_type}" + ".txt"
        if os.path.isfile(cond_txt_file_path):
            os.remove(cond_txt_file_path)
            logging.log('Deleted file at', cond_txt_file_path)

        print(f'Creating conditioning files from {paths_file} into dev folder') # todo: logs
        with open(cond_txt_file_path, 'w') as f:
            print(f'Create new conditioning paths file .txt at {cond_txt_file_path}')

            for audio_path in tqdm(audio_paths):
                audio_path = Path(audio_path)
                signal, _ = torchaudio.load(audio_path)
                signal = signal[0, :data_params.audio_length]

                # extract event cond
                cond = get_event_cond(signal, condition_params.event_type)

                # check for cond files folder existence
                cond_folder_path = Path(str(Path(audio_path).parent) + f"_{condition_params.event_type}")
                if not os.path.isdir(cond_folder_path):
                    os.mkdir(cond_folder_path)
                    print('Created', cond_folder_path, 'directory')

                # save conditioning file .pt
                cond_file_name = audio_path.stem + '.pt'
                cond_path = os.path.join(cond_folder_path, cond_file_name)
                torch.save(cond, cond_path)
                f.write(f'{cond_path}\n')


config = load_json_config(ProjectPaths().config_file)
txt_files = config.data.train_dirs
create_cond_files_from_txt(txt_files, config)

txt_files = config.data.test_dirs
create_cond_files_from_txt(txt_files, config)
