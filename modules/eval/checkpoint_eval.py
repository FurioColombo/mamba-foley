import os.path
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from modules.utils.utilities import plot_env, normalize
from modules.utils.audio import high_pass_filter, get_event_cond

class CheckpointEvaluator:
    def __init__(self, test_set, labels:list, audio_length: int, device, writer_dir:str or Path, sampler=None, event_type='rms'):
        self.test_set = test_set
        self.labels = labels
        self.loss_fn = nn.MSELoss()
        self.summary_writer = None
        self.audio_length = audio_length
        self.writer_dir = os.path.abspath(writer_dir)
        self.sampler=sampler
        self.device = device
        self.event_type = event_type

    def test_checkpoint_inference(self, step, conditioned=True, cond_scale=3., sampler=None):
        # update sampler
        self.sampler = sampler if sampler is not None else self.sampler
        assert self.sampler is not None

        # test features
        if conditioned:
            test_feature = self.test_set.dataset.get_random_sample()
            test_event = test_feature["event"].unsqueeze(0).to(self.device)
        else:
            test_feature = None
            test_event = None

        # Summary Writer
        writer = self.summary_writer or SummaryWriter(self.writer_dir, purge_step=int(step))
        writer.add_audio(f"test_sample/audio", test_feature["audio"], step, sample_rate=22050)
        writer.add_image(f"test_sample/envelope", plot_env(test_feature["audio"]), step, dataformats='HWC')

        event_loss = []
        for class_idx in range(len(self.labels)):
            noise = torch.randn(1, self.audio_length, device=self.device)
            classes = torch.tensor([class_idx], device=self.device)

            sample = sampler.predict(noise, 100, classes, test_event, cond_scale=cond_scale)
            if torch.isnan(sample).any():
                print(f'WARNING: NaN detected at step {step} - class: {self.labels[class_idx]} | SKIPPED LOGS FOR THIS ONE!')

                continue

            sample = sample.flatten().cpu()
            sample = normalize(sample)
            sample = high_pass_filter(sample, sr=22050)

            event_loss.append(self.loss_fn(test_event.squeeze(0).cpu(), get_event_cond(sample, self.event_type)))
            writer.add_audio(f"{self.labels[class_idx]}/audio", sample, step, sample_rate=22050)
            writer.add_image(f"{self.labels[class_idx]}/envelope", plot_env(sample), step, dataformats='HWC')


        event_loss = sum(event_loss) / len(event_loss) if len(event_loss) > 0 else 100
        writer.add_scalar(f"test/event_loss", event_loss, step)
        writer.flush()

        return event_loss

    def set_sampler(self, sampler):
        self.sampler = sampler
