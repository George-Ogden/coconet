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
from typing import Tuple

from data.dataset import Jsb16thSeparatedDataset, Jsb16thSeparatedDatasetFactory
from model import model


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    diffusion_timesteps: int = 300
    min_beta: float = 1e-3
    max_beta: float = 0.5

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        datasets: Tuple[Jsb16thSeparatedDataset, Jsb16thSeparatedDataset],
        config: TrainingConfig,
    ):
        # setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # save config and model
        self.model = model.to(self.device)
        self.config = copy(config)
        self.timesteps = config.diffusion_timesteps

        # precompute betas
        uncertainties = np.exp(np.linspace(np.log(config.min_beta), np.log(config.max_beta), config.diffusion_timesteps))
        certainties = 1 - uncertainties
        cumprod_certainties = np.concatenate(((1,), np.cumprod(certainties)))
        delta_alphas = 0.5 + cumprod_certainties / 2
        self.alphas = torch.tensor(
            1 / (.5 + certainties / 2) - 1, dtype=torch.float32
        ).to(
            self.device
        )
        self.cumulative_alphas = torch.tensor(
            1 / delta_alphas - 1, dtype=torch.float32
        ).to(
            self.device
        )
        # limit numerical instability
        self.cumulative_alphas[self.cumulative_alphas < 1e-3] = 1e-3

        # split into train and val
        train_dataset, val_dataset = datasets
        self.dataset_config = copy(train_dataset.info)
        self.train_loader = data.DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
        )
        self.val_loader = data.DataLoader(
            [val_dataset[i] for i in range(len(val_dataset))],
            batch_size=config.batch_size,
            shuffle=False,
        )

    @staticmethod
    def select(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return torch.index_select(x, 0, idx)

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        # sample timestep
        t = torch.randint(0, self.timesteps, size=(len(batch),)).to(self.device)

        # generate distribution
        batch = batch.to(self.device).float().permute((0, 3, 2, 1))
        alphas = torch.ones_like(batch)
        betas = torch.ones_like(batch)
        for i in range(len(batch)):
            alphas[i, batch[i] == 0] = self.cumulative_alphas[t[i]]
            betas[i, batch[i] == 1] = self.cumulative_alphas[t[i]]
        distribution = distributions.Beta(alphas, betas)

        # compute loss
        predicted = self.model(distribution.sample(), t).sample
        loss = F.binary_cross_entropy_with_logits(predicted, batch)
        return loss

    @torch.no_grad()
    def generate_samples(self, batch_size: int) -> torch.Tensor:
        predictions = torch.rand(
            batch_size,
            self.dataset_config.num_instruments,
            self.dataset_config.num_pitches,
            self.dataset_config.piece_length,
            device=self.device,
            dtype=torch.float32,
        )
        for t in tqdm(
            reversed(range(self.timesteps)), desc="Generating", total=self.timesteps
        ):
            # create distribution
            binary_predictions = distributions.Bernoulli(torch.sigmoid(predictions)).sample()
            alphas = torch.ones_like(predictions)
            betas = torch.ones_like(predictions)
            alphas[binary_predictions == 0] = self.alphas[t]
            betas[binary_predictions == 1] = self.alphas[t]
            distribution = distributions.Beta(alphas, betas).sample()

            predictions = self.model(distribution, t).sample
        return predictions > 0
    
    def train(self):
        wandb.init(project="coconet", config=self.config)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
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
            model.save_pretrained(f"checkpoints/{epoch:04d}")
            for i, sample in enumerate(samples):
                self.dataset_config.save_pianoroll(sample.cpu().permute((2, 1, 0)).numpy(), f"checkpoints/{epoch:04d}/sample-{i:02d}.mid")
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})


if __name__ == "__main__":
    factory = Jsb16thSeparatedDatasetFactory()
    trainer = Trainer(
        model, [factory.train_dataset, factory.val_dataset], TrainingConfig()
    )

    trainer.train()