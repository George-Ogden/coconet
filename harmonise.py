from train import Trainer, TrainingConfig
from data.dataset import DatasetInfo

import numpy as np
import torch

import pretty_midi
import os

dataset_config = DatasetInfo()
training_config = TrainingConfig()

save_directory = "harmonised"
input_midi = "input.mid"
model_directory = "pretrained"
# tracks are the index of the track in the midi, None means to be generated
tracks = [0, None, None, None]

if __name__ == "__main__":
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # load model
    model = torch.load(
        f"{model_directory}/model.pth"
    )
    # create trainer
    trainer = Trainer(
        model,
        config=TrainingConfig(),
        dataset_config=dataset_config
    )
    # load midi
    midi = pretty_midi.PrettyMIDI(input_midi)
    # convert midi to pianoroll
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
    # create a mask over selected tracks
    supermask = np.stack(
        [
            np.ones_like(pianoroll[i])
            if tracks[i] is None else
            np.zeros_like(pianoroll[i])
            for i in range(len(tracks))
        ],
        axis=0
    )
    # harmonise samples
    samples = trainer.generate_samples(16, original_pianoroll=pianoroll, supermask=supermask)
    # save each sample
    for i, sample in enumerate(samples):
        dataset_config.save_pianoroll(
            sample.cpu().permute(
                (2, 1, 0)).numpy(),
                f"{save_directory}/{i:02d}.mid"
            )