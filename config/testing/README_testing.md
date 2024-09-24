# [Model Info Config](../../config/testing/model_info_config.json)

## Overview
This config file defines the settings for the [model info script](../../scripts/python/testing/model_info.py), which is responsible for inspecting and loading details about the model and its configuration, such as depth, model path, and configuration file paths.

## Configurable Parameters
The script reads its configuration from the file `model_info_config.json`. Below is a list of the key parameters that can be configured:

### General Settings
- **CUDA_VISIBLE_DEVICES**: The GPU device(s) IDs to use for loading the model (e.g., `2`).
- **max_RAM_usage**: Maximum percentage of RAM to use during model inspection (e.g., 85%). If this limit is exceeded, an error will be raised.

### Model Settings
- **model_path**: Path to the pre-trained model file (e.g., `./pretrained/epoch-500_step-637037.pt`).
- **model_config_path**: Path to the model configuration JSON file (e.g., `./pretrained/params.json`).
- **depth**: Specifies the depth of the model's architecture to be displayed (e.g., `3`).

## Model Info Script Usage
```bash
python scripts/python/testing/model_info.py --config config/model_info_config.json
