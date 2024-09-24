from modules.data.dataset import AbstractAudioDataset, parse_filelist
from modules.utils.audio import get_event_cond

class AudioDataset(AbstractAudioDataset):
    def __init__(self, paths, params, labels):
        super().__init__(params, labels)
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

        event = signal.clone().detach()
        event = get_event_cond(event, self.event_type)

        return {
            'audio': signal,
            'class': cls,
            'event': event
        }
