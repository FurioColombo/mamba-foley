import torch
from modules.data.dataset import AbstractAudioDataset, parse_filelist

class CondAudioDataset(AbstractAudioDataset):
    def __init__(self, audio_paths, params, labels, cond_paths=None):
        super().__init__(params, labels)
        self.audio_filenames = []
        self.cond_filenames = []
        self.load_conditioning = audio_paths is not None

        if self.load_conditioning:
            for audio_path, cond_path in zip(audio_paths, cond_paths):
                self.audio_filenames += parse_filelist(audio_path)
                self.cond_filenames += parse_filelist(cond_path)

    def __len__(self):
        return len(self.audio_filenames)

    def __getitem__(self, idx):
        audio_filename = self.audio_filenames[idx]
        signal = self._load_audio(audio_filename)

        # extract class cond
        cls = self._extract_class_cond(audio_filename)

        cond_filename = self.cond_filenames[idx]
        event = torch.load(cond_filename)

        return {
            'audio': signal,
            'class': cls,
            'event': event
        }