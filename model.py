import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import DatasetInfo
from typing import Tuple, Union


class SeparableConv2d(nn.Module):
    # modified from https://stackoverflow.com/a/65155106/12103577
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, int] = "same",
        bias: bool = False,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=bias,
            dilation=dilation,
            padding=padding,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Model(nn.Module):
    skip_every = 2
    normal_conv_layers = 2
    num_dilation_blocks = 3
    filters = 128

    def __init__(self, info: DatasetInfo = DatasetInfo()):
        """CNN based on original coconet paper"""
        super().__init__()
        self.layers = nn.ModuleList(
            [
                *self._make_input_layers(info),
                *self._make_dilation_blocks(info),
                *self._make_output_layers(info),
            ]
        )

    def _make_input_layers(self, info: DatasetInfo):
        return nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=info.num_instruments  * 2,
                        out_channels=self.filters,
                        kernel_size=3,
                        dilation=1,
                        padding="same",
                    ),
                    nn.BatchNorm2d(self.filters),
                )
            ]
        )

    def _make_dilation_blocks(self, info: DatasetInfo):
        blocks = nn.ModuleList()
        max_time_dilation = math.ceil(math.log2(info.piece_length)) - 1
        max_pitch_dilation = math.ceil(math.log2(info.piece_length)) - 1
        max_dilation_level = max(max_time_dilation, max_pitch_dilation)
        for _ in range(self.num_dilation_blocks):
            for level in range(max_dilation_level + 1):
                # dilate in powers of 2
                time_dilation = 2 ** min(level, max_time_dilation)
                pitch_dilation = 2 ** min(level, max_pitch_dilation)
                blocks.append(
                    nn.Sequential(
                        SeparableConv2d(
                            in_channels=self.filters,
                            out_channels=self.filters,
                            kernel_size=3,
                            dilation=(pitch_dilation, time_dilation),
                        ),
                        nn.BatchNorm2d(self.filters),
                    )
                )
        return blocks

    def _make_output_layers(self, info: DatasetInfo):
        return nn.ModuleList(
            [
                nn.Sequential(
                    SeparableConv2d(
                        in_channels=self.filters,
                        out_channels=self.filters,
                        kernel_size=2,
                        dilation=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(self.filters),
                ),
                nn.Sequential(
                    SeparableConv2d(
                        in_channels=self.filters,
                        out_channels=info.num_instruments,
                        kernel_size=2,
                        dilation=1,
                        padding=0,
                    )
                ),
            ]
        )

    def forward(self, pianoroll: torch.Tensor) -> torch.Tensor:
        skip = (0, pianoroll)
        for i, layer in enumerate(self.layers):
            # pass through layer
            pianoroll = layer(pianoroll)

            # skip connection
            index, tensor = skip

            # don't skip while size is growing
            if tensor.shape != pianoroll.shape:
                skip = (i, pianoroll)
            else:
                # skip every nth layer
                if index + self.skip_every == i and (i != len(self.layers) - 1):
                    pianoroll = pianoroll + tensor
                    skip = (i, pianoroll)

            # relu after every layer except the last
            if i != len(self.layers) - 1:
                pianoroll = F.relu(pianoroll)
        return pianoroll
