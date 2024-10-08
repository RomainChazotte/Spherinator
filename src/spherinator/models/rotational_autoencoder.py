import math
from typing import Optional

import torch
import torch.linalg
import torch.nn as nn
from torch.optim import Adam

from .convolutional_decoder import ConvolutionalDecoder
from .convolutional_encoder import ConvolutionalEncoder
from .spherinator_module import SpherinatorModule


class RotationalAutoencoder(SpherinatorModule):
    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        z_dim: int = 3,
        image_size: int = 91,
        input_size: int = 128,
        rotations: int = 36,
        norm_brightness: bool = False,
    ):
        """Initializer

        Args:
            z_dim (int, optional): dimension of the latent representation. Defaults to 2.
            image_size (int, optional): size of the input images. Defaults to 91.
            rotations (int, optional): number of rotations. Defaults to 36.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])

        if encoder is None:
            encoder = ConvolutionalEncoder(latent_dim=z_dim)
        if decoder is None:
            decoder = ConvolutionalDecoder(latent_dim=z_dim)

        self.encoder = encoder
        self.decoder = decoder
        self.z_dim = z_dim
        self.image_size = image_size
        self.input_size = input_size
        self.rotations = rotations
        self.norm_brightness = norm_brightness

        self.crop_size = int(self.image_size * math.sqrt(2) / 2)
        self.total_input_size = self.input_size * self.input_size * 3

        self.example_input_array = torch.randn(1, 3, self.input_size, self.input_size)

    def get_input_size(self):
        return self.input_size

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def training_step(self, batch, batch_idx):

        best_scaled_image, _, _, _ = self.find_best_rotation(batch)
        recon = self.forward(best_scaled_image)
        loss = self.reconstruction_loss(best_scaled_image, recon)

        # divide by the brightness of the image
        if self.norm_brightness:
            loss = (
                loss / torch.sum(best_scaled_image, (1, 2, 3)) * self.total_input_size
            )

        loss = loss.mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        return loss

    def configure_optimizers(self):
        """Default Adam optimizer if missing from the configuration file."""
        return Adam(self.parameters(), lr=1e-3)

    def project(self, images):
        z = self.encode(images)
        return z

    def reconstruct(self, coordinates):
        return self.decode(coordinates)

    def reconstruction_loss(self, images, reconstructions):
        return torch.sqrt(
            nn.MSELoss(reduction="none")(
                reconstructions.reshape(-1, self.total_input_size),
                images.reshape(-1, self.total_input_size),
            ).mean(dim=1)
        )
