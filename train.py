import torch.distributions as distributions
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import math

from copy import copy
from tqdm import tqdm
import wandb

from dataclasses import dataclass
import os

from data.dataset import Jsb16thSeparatedDataset, Jsb16thSeparatedDatasetFactory, DatasetInfo
from model import Model


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    diffusion_timesteps: int = 300
    min_beta: float = 1e-4
    max_beta: float = 1e-2
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
        self.timesteps = config.diffusion_timesteps
        self.save_directory = config.save_directory

        # precompute flipping probabilities
        self.probabilities = torch.linspace(config.min_beta, config.max_beta, self.timesteps).to(self.device)
        self.cumulative_probabilities = (
            1 - torch.cumprod((1 - self.probabilities), dim=0)
        )

        # split into train and val
        self.dataset_config = copy(dataset_config)

    @staticmethod
    def select(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return torch.index_select(x, 0, idx)

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        # sample timestep
        t = torch.randint(0, self.timesteps, size=(len(batch),)).to(self.device)

        # generate distribution
        batch = batch.to(self.device).float().permute((0, 3, 2, 1))
        uncorrupted = batch.clone()
        p = self.select(self.cumulative_probabilities, t)[:, np.newaxis, np.newaxis, np.newaxis].tile((1,) + batch.shape[1:])
        distribution = distributions.Bernoulli(p)
        corrupted = distribution.sample().bool()
        batch[corrupted] = distributions.Bernoulli(torch.ones_like(batch[corrupted]) / 2).sample()

        # compute loss
        predicted = self.model(distribution.sample(), t)
        loss = F.binary_cross_entropy_with_logits(predicted, uncorrupted)
        return loss

    @torch.no_grad()
    def generate_samples(self, batch_size: int) -> torch.Tensor:
        distribution = distributions.Bernoulli(
            torch.ones(
                (batch_size, self.dataset_config.num_instruments, self.dataset_config.num_pitches, self.dataset_config.piece_length),
                device=self.device,
                dtype=torch.float32,
            ) / 2
        ).sample()
        for t in tqdm(
            reversed(range(self.timesteps)), desc="Generating", total=self.timesteps
        ):
            # corrupt some values
            p = self.probabilities[t][(np.newaxis,) * distribution.ndim].tile(distribution.shape)
            corrupted = distributions.Bernoulli(p).sample().bool()
            distribution[corrupted] = distributions.Bernoulli(torch.ones_like(distribution[corrupted]) / 2).sample()
            # clamp to [0, 1]
            distribution = distributions.Bernoulli(
                F.sigmoid(self.model(distribution, t))
            ).sample()
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