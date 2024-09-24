# [Inference Config](../../config/inference/inference_config.json)

## Overview
This config file is responsible for the configuration of the corresponding [inference script](../../scripts/python/inference/inference.py). It defines parameters for generating audio samples using a pre-trained model, allowing control over temporal conditioning, class-specific generation, and output settings.

## Configurable Parameters
The script reads its configuration from the file `inference_config.json`. Below is a list of the key parameters that can be configured:

### General Settings
- **CUDA_VISIBLE_DEVICES**: The GPU device(s) IDs to use for inference (e.g., `2`).
- **max_RAM_usage**: Maximum percentage of RAM to use during inference (e.g., 85%). Inference will stop if the RAM usage exceeds this limit.

### Model Settings
- **model_checkpoints_path**: Path to the pre-trained model checkpoint (e.g., `./pretrained/epoch-500_step-637037.pt`).
- **model_config_path**: Path to the model configuration JSON file (e.g., `./pretrained/params.json`).

### Output Settings
- **output_dir**: Directory where the generated audio files will be saved (e.g., `results`).

### Generation Settings
- **gen_all_classes**: If `True`, generates audio for all available classes. If `False`, only the class specified in `class_names` will be used.
- **class_names**: Specifies the class for which the audio is generated (e.g., `DogBark`).
- **same_class_conditioning**: If `True`, uses a random audio from the evaluation folder for conditioning. If `False`, specify a target audio using `target_audio_path`.
- **target_audio_path**: Path to the target audio used for conditioning (optional, used only if `same_class_conditioning` is set to `False`).
- **cond_scale**: Determines how much the temporal conditioning is weighted on a scale from 0 to 3 (e.g., `3`).
- **stereo**: If `True`, generates stereo audio with the target on the left and generated audio on the right (e.g., `False`).

### Sampling Settings
- **samples_per_class**: Number of audio samples to generate per class (e.g., `1`).
- **samples_per_temporal_cond**: Number of samples to generate for each temporal condition (e.g., `1`).

## Inference Script Usage
```bash
python scripts/python/inference/inference.py --config config/inference_config.json
