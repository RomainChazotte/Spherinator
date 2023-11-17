import gc

import matplotlib
import torch
import torchvision.transforms.functional as functional
from lightning.pytorch.callbacks import Callback
from matplotlib import figure

matplotlib.use('Agg')


class LogReconstructionCallback(Callback):
    def __init__(self, num_samples=4):
        super().__init__()
        self.num_samples = num_samples

    def on_train_epoch_end(self, trainer, pl_module):
        # Return if no wandb logger is used
        if trainer.logger is None or trainer.logger.__class__.__name__ not in [
            "WandbLogger",
            "MyLogger",
        ]:
            return

        # Generate some random samples from the validation set
        samples = next(iter(trainer.train_dataloader))["image"]
        samples = samples[: self.num_samples]
        samples = samples.to(pl_module.device)

        # Generate reconstructions of the samples using the model
        with torch.no_grad():
            batch_size = samples.shape[0]
            losses = torch.zeros(batch_size, pl_module.rotations)
            images = torch.zeros(
                batch_size,
                3,
                pl_module.input_size,
                pl_module.input_size,
                pl_module.rotations,
            )
            recons = torch.zeros(
                batch_size,
                3,
                pl_module.input_size,
                pl_module.input_size,
                pl_module.rotations,
            )
            coords = torch.zeros(batch_size, pl_module.z_dim, pl_module.rotations)
            for r in range(pl_module.rotations):
                rotate = functional.rotate(
                    samples, 360.0 / pl_module.rotations * r, expand=False
                )
                crop = functional.center_crop(
                    rotate, [pl_module.crop_size, pl_module.crop_size]
                )
                scaled = functional.resize(
                    crop, [pl_module.input_size, pl_module.input_size], antialias=False
                )

                (z_mean, _), (_, _), _, recon = pl_module(scaled)

                losses[:, r] = pl_module.reconstruction_loss(scaled, recon)
                images[:, :, :, :, r] = scaled
                recons[:, :, :, :, r] = recon
                coords[:, :, r] = z_mean

            min_idx = torch.min(losses, dim=1)[1]

        # Plot the original samples and their reconstructions side by side
        fig = figure.Figure(figsize=(6, 2 * self.num_samples))
        ax = fig.subplots(self.num_samples, 2)
        for i in range(self.num_samples):
            ax[i, 0].imshow(images[i, :, :, :, min_idx[i]].cpu().detach().numpy().T)
            ax[i, 0].set_title("Original")
            ax[i, 0].axis("off")
            ax[i, 1].imshow(recons[i, :, :, :, min_idx[i]].cpu().detach().numpy().T)
            ax[i, 1].set_title("Reconstruction")
            ax[i, 1].axis("off")
        fig.tight_layout()

        # Log the figure at W&B
        trainer.logger.log_image(key="Reconstructions", images=[fig])

        # Clear the figure and free memory
        # Memory leak issue: https://github.com/matplotlib/matplotlib/issues/27138
        for i in range(self.num_samples):
            ax[i, 0].clear()
            ax[i, 1].clear()
        fig.clear()
        gc.collect()
