import os
import torch
from torch.utils.data import DistributedSampler
from modules.data.sources.original import AudioDataset
from modules.data.sources.precomputed_conditioning import CondAudioDataset


def dataset_from_path(data_dirs, params, labels, distributed=False, cond_dirs=None):
    params_train = params.training if hasattr(params, 'training') else params
    params_data = params.data if hasattr(params, 'data') else params

    if cond_dirs is None:
        dataset = AudioDataset(data_dirs, params, labels)
    else:
        dataset = CondAudioDataset(
            audio_paths=data_dirs,
            cond_paths=cond_dirs,
            params=params,
            labels=labels
        )

    if distributed:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=params_train.batch_size,
            collate_fn=None,
            shuffle=False,
            num_workers=params_data.n_workers,
            pin_memory=True,
            drop_last=True,
            sampler=DistributedSampler(dataset)
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params_train.batch_size,
        collate_fn=None,
        shuffle=True,
        num_workers=os.cpu_count()//4,
        pin_memory=True,
        drop_last=True
    )
