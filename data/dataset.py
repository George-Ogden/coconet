from torch.utils.data import Dataset
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
    shortest_duration: float = 0.125
    num_instruments: int = 4
    piece_length: int = 64
    qpm: int = 60

    @property
    def num_pitches(self):
        return self.max_pitch - self.min_pitch + 1


class Jsb16thSeparatedDataset(Dataset):
    def __init__(self, data: List[List[List[int]]], info: DatasetInfo = DatasetInfo()):
        self.info = copy(info)

        self.min_pitch = info.min_pitch
        self.max_pitch = info.max_pitch
        self.shortest_duration = info.shortest_duration
        self.num_instruments = info.num_instruments
        self.qpm = info.qpm

        self.data = [self.to_pianoroll(piece) for piece in data]

    def to_pianoroll(
        self,
        piece: List[List[int]],
    ) -> np.ndarray:
        pianoroll = np.zeros((len(piece), self.info.num_pitches, 4), dtype=bool)
        for i, chord in enumerate(piece):
            for j, track in enumerate(chord):
                pianoroll[i, track - self.min_pitch, j] = True
        return pianoroll

    def random_crop(self, pianoroll: np.ndarray) -> np.ndarray:
        if len(pianoroll) < self.info.piece_length:
            raise ValueError(f"Piece length is too short: {len(pianoroll)}")
        start = np.random.choice(
            list(
                range(
                    len(pianoroll) % self.info.piece_length,
                    len(pianoroll),
                    self.info.piece_length,
                )
            )
        )
        return pianoroll[start : start + self.info.piece_length]

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.random_crop(self.data[idx])

    def __len__(self) -> int:
        return len(self.data)
