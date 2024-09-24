import os
import sys
import subprocess

import torch
import torchaudio as T
from torchsummary import summary

root_folder_path = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], universal_newlines=True).strip())
modules_folder_path = os.path.join(root_folder_path, "modules")
sys.path.append(root_folder_path)
sys.path.append(modules_folder_path)
from modules.model.model import UNet
from modules.utils.file_system import ProjectPaths
from config.config import Config

LABELS = ['DogBark', 'Footstep', 'GunShot', 'Keyboard', 'MovingMotorVehicle', 'Rain', 'Sneeze_Cough']


def load_ema_weights(model, model_path):
    checkpoint = torch.load(model_path)
    dic_ema = {}
    for (key, tensor) in zip(checkpoint['model'].keys(), checkpoint['ema_weights']):
        dic_ema[key] = tensor
    model.load_state_dict(dic_ema)
    return model


def main(params):
    # Set model and sampler
    T.set_audio_backend('sox_io')
    device = torch.device('cuda')

    model_config = Config.get_config(params.model_config_path)
    model = UNet(len(LABELS), config=model_config).to(device)

    summary(model, depth=params.depth)
    print('========================= END =========================')


if __name__ == '__main__':
    config = Config.get_config(ProjectPaths.config_script_model_info)
    main(config)
