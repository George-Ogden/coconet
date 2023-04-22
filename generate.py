from train import Trainer, TrainingConfig
from data.dataset import DatasetInfo
from model import Model

import torch

dataset_config = DatasetInfo()
training_config = TrainingConfig()

save_directory = "generated"

if __name__ == "__main__":
    model = torch.load(
        f"{training_config.save_directory}/{training_config.epochs-1:04}/model.pth"
    )
    trainer = Trainer(
        model,
        config=TrainingConfig(),
        dataset_config=dataset_config
    )
    samples = trainer.generate_samples(16)
    for i, sample in enumerate(samples):
        dataset_config.save_pianoroll(
            sample.cpu().permute(
                (2, 1, 0)).numpy(),
                f"{save_directory}/{i:02d}.mid"
            )