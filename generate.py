from train import Trainer, TrainingConfig
from data.dataset import DatasetInfo

import torch
import os

dataset_config = DatasetInfo()
training_config = TrainingConfig()

save_directory = "generated"
model_directory = f"{training_config.save_directory}/{training_config.epochs-1:04}/model.pth"

if __name__ == "__main__":
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # load model
    model = torch.load(
       model_directory
    )
    # create trainer
    trainer = Trainer(
        model,
        config=TrainingConfig(),
        dataset_config=dataset_config
    )
    # generate samples
    samples = trainer.generate_samples(16)
    # save each sample
    for i, sample in enumerate(samples):
        dataset_config.save_pianoroll(
            sample.cpu().permute(
                (2, 1, 0)).numpy(),
                f"{save_directory}/{i:02d}.mid"
            )