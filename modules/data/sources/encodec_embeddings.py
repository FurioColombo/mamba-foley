import torch

from modules.data.dataset import AbstractAudioDataset, parse_filelist
from modules.utils.utilities import get_event_cond

from encodecWrapper import EncodecModel
from encodecWrapper.utils import convert_audio


class EmbeddingDataset(AbstractAudioDataset):
    def __init__(self, paths, params, labels):
        super().__init__(params, labels)
        self.codec = EncodecModel.encodec_model_24khz()
        self.codec.set_target_bandwidth(6.0)
        self.filenames = []
        for path in paths:
            self.filenames += parse_filelist(path)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        signal = self._load_audio(audio_filename)

        # extract class cond
        cls = self._extract_class_cond(audio_filename)

        # extract temporal conditioning
        event = signal.clone().detach()
        event = get_event_cond(event, self.event_type)

        # extract audio embedding
        audio = signal.clone().detach()
        audio = convert_audio(audio, self.sample_rate)
        audio = audio.unsqueeze(0)
        with torch.no_grad():
            audio_embedding = self.codec.encode(audio)

        return {
            'audio': signal,
            'class': cls,
            'event': event,
            'embedding': audio_embedding
        }
