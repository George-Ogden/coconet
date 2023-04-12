from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from accelerate import Accelerator

import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn
import torch

from copy import copy
from tqdm import tqdm
import wandb

from dataclasses import dataclass
import os

from data.dataset import Jsb16thSeparatedDataset, Jsb16thSeparatedDatasetFactory, DatasetInfo

@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    diffusion_timesteps: int = 300
    beta_schedule: str = "linear"
    prediction_type: str = "epsilon"
    save_directory: str = "checkpoints"

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig=TrainingConfig(),
        dataset_config: DatasetInfo=DatasetInfo()
    ):
        # save config and model
        self.model = model
        self.config = copy(config)
        self.timesteps = config.diffusion_timesteps
        self.save_directory = config.save_directory

        # setup noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.config.diffusion_timesteps,
            beta_schedule=self.config.beta_schedule,
            prediction_type=self.config.prediction_type,
        )

        # split into train and val
        self.dataset_config = copy(dataset_config)

        # setup accelerator
        self.accelerator = Accelerator(
            log_with="wandb",
            logging_dir=config.save_directory
        )

    def train(self, train_dataset: Jsb16thSeparatedDataset):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)

        train_dataloader = data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4
        )

        self.model, optimizer, train_dataloader = self.accelerator.prepare(
            self.model, optimizer, train_dataloader
        )

        run = os.path.split(__file__)[-1].split(".")[0]
        self.accelerator.init_trackers(run)

        global_step = 0
        for epoch in range(self.config.epochs):
            self.model.train()
            progress_bar = tqdm(train_dataloader, total=len(train_dataloader))
            progress_bar.set_description(f"Epoch {epoch}")
            for batch in progress_bar:

                # convert correct shape
                clean_images = batch.float().permute((0, 3, 2, 1))
                # Sample noise that we'll add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bsz = clean_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, self.config.diffusion_timesteps, (bsz,), device=clean_images.device
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

                with self.accelerator.accumulate(self.model):
                    # Predict the noise residual
                    model_output = self.model(noisy_images, timesteps).sample

                    if self.config.prediction_type == "epsilon":
                        loss = F.mse_loss(model_output, noise)  # this could have different weights!
                    elif self.config.prediction_type == "sample":
                        alpha_t = _extract_into_tensor(
                            self.noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                        )
                        snr_weights = alpha_t / (1 - alpha_t)
                        loss = snr_weights * F.mse_loss(
                            model_output, clean_images, reduction="none"
                        )  # use SNR weighting from distillation paper
                        loss = loss.mean()
                    else:
                        raise ValueError(f"Unsupported prediction type: {self.config.prediction_type}")

                    self.accelerator.backward(loss)

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                logs = {"loss": loss.detach().item(), "step": global_step}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

            # Generate sample images for visual inspection
            unet = self.accelerator.unwrap_model(self.model)

            pipeline = DDPMPipeline(
                unet=unet,
                scheduler=self.noise_scheduler,
            )

            generator = torch.Generator(device=pipeline.device).manual_seed(0)
            # run pipeline in inference (sample random noise and denoise)
            images = pipeline(
                generator=generator,
                batch_size=10,
                num_inference_steps=self.config.diffusion_timesteps,
                output_type="numpy",
            ).images

            # denormalize the images and save to tensorboard
            images_processed = (images * 255).round().astype("uint8")

            # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
            self.accelerator.get_tracker("wandb").log(
                {"test_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
                step=global_step,
            )

            # save the model
            unet = self.accelerator.unwrap_model(self.model)

            pipeline = DDPMPipeline(
                unet=unet,
                scheduler=self.noise_scheduler,
            )

            pipeline.save_pretrained(self.config.save_directory)
            for i, sample in enumerate(images_processed):
                self.dataset_config.save_pianoroll((sample > 0).transpose((2, 1, 0)), f"{self.save_directory}/{epoch:04d}-{i:02d}.mid")

        self.accelerator.end_training()

if __name__ == "__main__":
    factory = Jsb16thSeparatedDatasetFactory()
    model = UNet2DModel(
        in_channels=factory.info.num_instruments,
        out_channels=factory.info.num_instruments,
        sample_size=(factory.info.num_pitches, factory.info.piece_length),
    )
    trainer = Trainer(
        model,
        config=TrainingConfig(),
        dataset_config=factory.info,
    )

    trainer.train(factory.train_dataset)