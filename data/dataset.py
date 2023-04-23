from torch.utils.data import Dataset
import pypianoroll as pr
import numpy as np

from functools import cached_property
from dataclasses import dataclass
from copy import copy
import os.path
import json

from typing import List

path = os.path.join(os.path.dirname(__file__), "jsb-chorales-16th.json")


@dataclass
class DatasetInfo:
    name: str = "Jsb16thSeparated"
    min_pitch: int = 36
    max_pitch: int = 81
    resolution: float = 16 # 16th notes
    num_instruments: int = 4
    piece_length: int = 64
    bpm: int = 15

    @property
    def num_pitches(self):
        return self.max_pitch - self.min_pitch + 1

    def save_pianoroll(self, pianoroll: np.ndarray, filename: str):
        """save pianoroll to midi file

        Args:
            pianoroll (np.ndarray): numpy array of shape (num_timesteps, num_pitches, num_tracks)
            filename (str): midi filename
        """
        pianoroll = np.pad(pianoroll, ((0, 0), (self.min_pitch, 127-self.max_pitch), (0, 0)), mode="constant", constant_values=0)
        tracks = [pr.BinaryTrack(pianoroll=track) for track in pianoroll.transpose(2, 0, 1)]
        multitrack = pr.Multitrack(tracks=tracks, tempo=self.bpm, resolution=self.resolution // 2)
        multitrack.write(filename)
    
    def to_pianoroll(
        self,
        piece: List[List[int]],
    ) -> np.ndarray:
        """convert piece to binary pianoroll

        Args:
            piece (List[List[int]]): each element of the list is a chord, each chord is a list of pitches so the list should be of shape (num_timesteps, num_tracks)

        Returns:
            np.ndarray: binary pianoroll of shape (num_timesteps, num_pitches, num_tracks)
        """        
        pianoroll = np.zeros((len(piece), self.num_pitches, self.num_instruments), dtype=bool)
        for i, chord in enumerate(piece):
            for j, track in enumerate(chord):
                assert self.min_pitch <= track <= self.max_pitch, "pitch out of range"
                pianoroll[i, track - self.min_pitch, j] = True
        return pianoroll


class Jsb16thSeparatedDataset(Dataset):
    def __init__(self, data: List[List[List[int]]], info: DatasetInfo = DatasetInfo()):
        self.info = copy(info)

        # save all the info as attributes
        self.min_pitch = info.min_pitch
        self.max_pitch = info.max_pitch
        self.resolution = info.resolution
        self.num_instruments = info.num_instruments
        self.qpm = info.bpm

        # convert each piece to a pianoroll
        self.data = [self.info.to_pianoroll(piece) for piece in data]

    def _random_crop(self, pianoroll: np.ndarray) -> np.ndarray:
        if len(pianoroll) < self.info.piece_length:
            raise ValueError(f"Piece length is too short: {len(pianoroll)}")
        # pick a random start index (the piece starts, rather than ends with an upbeat)
        start = np.random.choice(
            np.arange(
                len(pianoroll) % self.info.piece_length,
                len(pianoroll),
                self.info.piece_length,
            )
        )
        return pianoroll[start : start + self.info.piece_length]

    def __getitem__(self, idx: int) -> np.ndarray:
        # return a random crop of the piece
        return self._random_crop(self.data[idx])

    def __len__(self) -> int:
        return len(self.data)


class Jsb16thSeparatedDatasetFactory:
    """create datasets from a json file"""
    def __init__(self, path: str = path, info: DatasetInfo = DatasetInfo()):
        with open(path) as f:
            self.data = json.load(f)
        self.info = info

    @cached_property
    def train_dataset(self):
        return Jsb16thSeparatedDataset(self.data["train"], self.info)

    @cached_property
    def val_dataset(self):
        return Jsb16thSeparatedDataset(self.data["valid"], self.info)

    @cached_property
    def test_dataset(self):
        return Jsb16thSeparatedDataset(self.data["test"], self.info)


if __name__ == "__main__":
    factory = Jsb16thSeparatedDatasetFactory()

    # check lengths of all datasets
    train_dataset = factory.train_dataset
    for i in range(len(train_dataset)):
        train_dataset[i]

    val_dataset = factory.val_dataset
    for i in range(len(val_dataset)):
        val_dataset[i]

    test_dataset = factory.test_dataset
    for i in range(len(test_dataset)):
        test_dataset[i]
