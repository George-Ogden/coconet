from data.dataset import DatasetInfo
from diffusers import UNet2DModel

info = DatasetInfo()
model = UNet2DModel(
    sample_size=(info.num_pitches, info.piece_length),
    in_channels=info.num_instruments,
    out_channels=info.num_instruments,
)