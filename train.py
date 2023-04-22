import torch.distributions as distributions
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

from copy import copy
from tqdm import tqdm
import wandb

from dataclasses import dataclass
from typing import Union
import os

from data.dataset import Jsb16thSeparatedDataset, Jsb16thSeparatedDatasetFactory, DatasetInfo
from model import Model


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-2
    timesteps: int = 100
    min_beta: float = 1e-2
    max_beta: float = 9e-1
    save_directory: str = "checkpoints"


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig=TrainingConfig(),
        dataset_config: DatasetInfo=DatasetInfo()
    ):
        # setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # save config and model
        self.model = model.to(self.device)
        self.config = copy(config)
        self.timesteps = config.timesteps
        self.save_directory = config.save_directory

        # precompute flipping probabilities
        self.probabilities = torch.linspace(config.min_beta, config.max_beta, self.timesteps).to(self.device)

        # split into train and val
        self.dataset_config = copy(dataset_config)

    @staticmethod
    def select(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return torch.index_select(x, 0, idx)

    def mask(self, x: torch.Tensor, p: Union[torch.Tensor, float]) -> torch.Tensor:
        # convert probs to tensor
        if isinstance(p, float) or p.ndim == 0:
            p = torch.tensor([p] * len(x), device=self.device)
        # create mask with bernoulli
        p = torch.tile(p[(slice(None),) + (np.newaxis,) * (x.ndim - 1)], (1, *x.shape[1:]))
        mask = distributions.Bernoulli(p).sample()
        # apply mask and generate random values over mask
        y = torch.cat(
            (
                torch.where(
                    mask.bool(),
                    distributions.Bernoulli(torch.ones_like(x) / 2).sample(),
                    x,
                ),
                mask
            ),
            dim=1
        )
        return y

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        # sample timestep
        t = torch.randint(0, self.timesteps, size=(len(batch),)).to(self.device)

        # generate distribution
        batch = batch.to(self.device).float().permute((0, 3, 2, 1))
        p = self.select(self.probabilities, t)
        
        # compute loss
        predicted = self.model(self.mask(batch, p))
        loss = F.binary_cross_entropy_with_logits(predicted, batch)
        return loss

    @torch.no_grad()
    def generate_samples(self, batch_size: int) -> torch.Tensor:
        distribution = self.mask(
            torch.zeros(
                (batch_size, self.dataset_config.num_instruments, self.dataset_config.num_pitches, self.dataset_config.piece_length),
                device=self.device,
                dtype=torch.float32,
            ),
            1.
        )
        for t in tqdm(
            reversed(range(self.timesteps)), desc="Generating", total=self.timesteps
        ):
            # predict distribution
            distribution = distributions.Bernoulli(
                F.sigmoid(self.model(distribution))
            ).sample()
            if t > 0:
                # mask some notes
                distribution = self.mask(distribution, self.probabilities[t])
        return distribution

    def train(self, train_dataset: Jsb16thSeparatedDataset, val_dataset: Jsb16thSeparatedDataset):
        wandb.init(project="coconet", config=self.config, dir=self.save_directory)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # set up dataloaders
        self.train_loader = data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4
        )
        self.val_loader = data.DataLoader(
            [val_dataset[i] for i in range(len(val_dataset))],
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        for epoch in range(self.config.epochs):
            # train
            self.model.train()
            train_loss = 0.
            for batch in tqdm(self.train_loader, desc="Training", total=len(self.train_loader)):
                optimizer.zero_grad()
                loss = self.compute_loss(batch)
                train_loss += loss.detach().cpu().item()
                loss.backward()
                optimizer.step()
            train_loss /= len(self.train_loader)

            # validate
            self.model.eval()
            val_loss = 0.
            with torch.no_grad(), torch.random.fork_rng(devices=[self.device]):
                # randomly seed for repeatable loss
                torch.random.manual_seed(0)
                for batch in tqdm(self.val_loader, desc="Validating", total=len(self.val_loader)):
                    val_loss += self.compute_loss(batch).item()
            val_loss /= len(self.val_loader)

            # generate samples
            samples = self.generate_samples(4)

            # save
            os.makedirs(f"{self.save_directory}/{epoch:04d}/samples/", exist_ok=True)
            torch.save(model, f"{self.save_directory}/{epoch:04d}/model.pth")
            for i, sample in enumerate(samples):
                self.dataset_config.save_pianoroll(sample.cpu().permute((2, 1, 0)).numpy(), f"{self.save_directory}/{epoch:04d}/samples/{i:02d}.mid")
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})


if __name__ == "__main__":
    factory = Jsb16thSeparatedDatasetFactory()
    model = Model(factory.info)
    trainer = Trainer(
        model,
        config=TrainingConfig(),
        dataset_config=factory.info,
    )

    trainer.train(factory.train_dataset, factory.val_dataset)