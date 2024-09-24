
# [Training Config](../../config/inference/inference_config.json)

## Overview
This config file is responsible for the configuration of the corresponding [training script](../../scripts/python/training/train_model.py).
I allows to set many parameters regarding both the dataset, the model's architecture and the training specs. 

## Configurable Parameters
The script reads its configuration from the file `train_config.json`. Below is a list of the key parameters that can be configured:

### General Settings
- **CUDA_VISIBLE_DEVICES**: The GPU device(s) IDs to use for training. Supports multi-GPU configurations (e.g. `[0, 1]`).
  - **max_RAM_usage**: Maximum percentage of RAM to use during training (e.g., 85%). When the maximum quota is exceeded, a checkpoint is saved and the training is stopped.

### Data Settings
- **train_dirs**: Path(s) to the training data, provided as a text file containing audio paths (e.g., `./DCASE_2023_Challenge_Task_7_Dataset/train.txt`).
  - **test_dirs**: Path(s) to the testing/evaluation data, provided as a text file containing audio paths (e.g., `./DCASE_2023_Challenge_Task_7_Dataset/eval.txt`).
  - **sample_rate**: The sampling rate of the audio data (e.g., 22,050 Hz).
  - **audio_length**: The length of each audio sample in number of samples (e.g., 88,200).
  - **n_workers**: Number of workers for data loading (e.g., 4).

### Model Settings
- **model_dir**: Directory where model checkpoints and logs will be saved (e.g., `train_logs`).
  - **sequential**: Type of sequential model used. Available choices are: `mamba`, `lstm`, `attn`).
  - **factors**: List of factors used in the model's architecture (e.g., `[2, 2, 3, 5, 5]`).
  - **dims**: List of layer dimensions in the model.
  - **bottleneck_layers**: Number of bottleneck layers in the architecture (e.g., 1).
  - **bidirectional_bottleneck**: Whether the bottleneck is bidirectional or not (`true` or `false`).

### Condition Settings
- **time_emb_dim**: Dimensionality of the time embedding (e.g., 512).
  - **class_emb_dim**: Dimensionality of the class embedding (e.g., 512).
  - **film_type**: Type of FiLM used, available choices are: `block`, `temporal`, `film` and `mamba`).
  - **event_type**: Type of event used for conditioning (e.g., `rms`).
  - **cond_prob**: Probability of applying conditioning at each stage of the model (e.g., `[0.1, 0.1]`).

### Training Settings
- **lr**: Learning rate (e.g., 0.0001).
  - **batch_size**: Batch size for training (e.g., 16).
  - **ema_rate**: Exponential moving average rate (e.g., 0.999).
  - **n_epochs**: Total number of epochs to train (e.g., 500).

### Logging Settings
- **n_epochs_to_checkpoint**: Number of epochs between saving model checkpoints (e.g., 15).
  - **n_steps_to_log**: Number of steps between logging updates (e.g., 300). This should be changed accordingly to different batch sizes related to the batch size.

## Training Script usage
```bash
 python scripts/python/training/train_model.py 
```
