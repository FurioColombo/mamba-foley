# TODO: integrate this in project architecture
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm
import subprocess

from types import SimpleNamespace
from datetime import datetime
import numpy as np


def recursive_namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        # Convert SimpleNamespace to a dictionary and recursively process values
        obj = vars(obj)

    if isinstance(obj, dict):
        # Recursively convert all keys and values in the dictionary
        return {key: recursive_namespace_to_dict(value) for key, value in obj.items()}

    elif isinstance(obj, list):
        # Recursively convert all items in the list
        return [recursive_namespace_to_dict(item) for item in obj]

    elif isinstance(obj, datetime):
        # Convert datetime objects to ISO 8601 string
        return obj.isoformat()

    elif isinstance(obj, set):
        # Convert sets to lists
        return list(obj)

    elif isinstance(obj, np.ndarray):
        # Convert numpy arrays to lists
        return obj.tolist()

    else:
        # Base case: return the object if it is not one of the above types
        return obj


def get_git_root():
    try:
        # Execute the git command to get the top-level directory of the repository
        git_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], stderr=subprocess.STDOUT).strip().decode('utf-8')
        return git_root
    except subprocess.CalledProcessError:
        # Handle the case where the current directory is not in a git repository
        raise RuntimeError("This directory is not part of a Git repository")


def copy_wav_files_from_folder_recursively(source_dir, dest_dir):
    """
    Copy all .wav files from a source directory and its subdirectories to a destination directory,
    including the name of the folder they were in.

    Args:
        source_dir (str): Path to the source directory.
        dest_dir (str): Path to the destination directory.
    """

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    root = os.path.relpath(source_dir)
    for root, dirs, files in tqdm(os.walk(source_dir), desc=f'duplicating files from {root}'):
        if os.path.abspath(root) == os.path.abspath(dest_dir):
            continue

        for file in files:
            if file.endswith(".wav"):
                source_path = os.path.join(root, file)
                # Extract the name of the folder containing the file
                folder_name = os.path.basename(root)
                # Modify the destination path to include the folder name
                dest_path = os.path.join(dest_dir, folder_name + '_' + file)
                shutil.copyfile(source_path, dest_path)
                # print(f"\nCopied {source_path}\nto {dest_path}")

def get_files_in_dir(path: str or Path, extension=None):
    p = os.path.abspath(path)
    assert os.path.isdir(p)
    file_paths = [os.path.join(p, file) for file in os.listdir(p) if os.path.isfile(os.path.join(p, file))]
    if extension is not None:
        assert type(extension) is str
        extension = extension.split('.')[-1]
        file_paths = [f for f in file_paths if f.split('.')[-1] == extension]
    return file_paths


class ProjectPaths:
    try:
        base_dir = get_git_root()
    except RuntimeError:
        base_dir = Path(__file__).parent.parent

    dataset_dir = os.path.join(base_dir, 'DCASE_2023_Challenge_Task_7_Dataset')
    dataset_dev_dir = os.path.join(dataset_dir, 'dev')
    dataset_eval_dir = os.path.join(dataset_dir, 'eval')
    models_dir = os.path.join(base_dir, 'models')
    munet_model_dir = os.path.join(base_dir, 'munet')
    logs_dir = os.path.join(base_dir, 'logs')
    pretrained_models = os.path.join(base_dir, 'pretrained')

    # config
    config_dir = os.path.join(base_dir, 'config')
    config_script_inference = os.path.join(config_dir, 'inference', 'inference_config.json')
    config_script_train = os.path.join(config_dir, 'training', 'train_config.json')
    config_script_model_info = os.path.join(config_dir, 'testing', 'model_info_config.json')

    config_file = os.path.join(config_dir, 'config.json')
    fad_encoded_emb_folder_path = os.path.join(base_dir, 'fad', 'encodec')

