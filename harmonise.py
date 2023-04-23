from train import Trainer, TrainingConfig
from data.dataset import DatasetInfo

import pypianoroll as pr
import pretty_midi
import numpy as np
import torch

dataset_config = DatasetInfo()
training_config = TrainingConfig()

save_directory = "harmonised"
input_midi = "lamb.mid"
tracks = [0, None, None, None]

if __name__ == "__main__":
    model = torch.load(
        f"{training_config.save_directory}/{training_config.epochs-1:04}/model.pth"
    )
    trainer = Trainer(
        model,
        config=TrainingConfig(),
        dataset_config=dataset_config
    )
    midi = pretty_midi.PrettyMIDI(input_midi)
    # dataset_config.qpm = midi.estimate_tempo()
    pianoroll = np.stack(
        [
            instrument.get_piano_roll(fs=dataset_config.bpm / 60 * dataset_config.resolution)
            for instrument in midi.instruments
        ],
        axis=0
    )
    pianoroll = np.stack(
        [
            np.zeros(pianoroll.shape[1:], dtype=bool)
            if idx is None
            else pianoroll[idx]
            for idx in tracks
        ],
        axis=0
    )
    pianoroll = pianoroll[:, dataset_config.min_pitch-12:dataset_config.max_pitch + 1-12]
    supermask = np.stack(
        [
            np.ones_like(pianoroll[i])
            if tracks[i] is None else
            np.zeros_like(pianoroll[i])
            for i in range(len(tracks))
        ],
        axis=0
    )
    samples = trainer.generate_samples(1, original_pianoroll=pianoroll, supermask=supermask)
    for i, sample in enumerate(samples):
        dataset_config.save_pianoroll(
            sample.cpu().permute(
                (2, 1, 0)).numpy(),
                f"{save_directory}/{i:02d}.mid"
            )