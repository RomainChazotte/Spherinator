import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os.path

import lightning.pytorch as pl

from .embedding import ZernikeEmbedding
from .zernike_encoder_auto import ZernikeEncoderAuto
from .zernike_decoder_auto import ZernikeDecoderAuto
import torchvision.transforms.functional as functional
import torchvision

class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        image_size: int = 28,
        n_max = 32,
        device='cuda'
    ):
        super().__init__()
        
    


        self.Embedding_Function = ZernikeEmbedding(image_size, n_max, device)

        self.encoder = ZernikeEncoderAuto(n_max, 2, 3, 8*2, device=device, image_size=image_size,Very_small=True,Very_large=True)
        self.decoder = ZernikeDecoderAuto(n_max, 2, 3, 8*2, device=device, image_size=image_size,Very_small=True,Very_large=True)
        
        self.image_size = image_size
        self.input_size = image_size
        self.reduce_size = False
        self.crop_size = int(self.image_size * math.sqrt(2) / 2)
        self.total_input_size = self.input_size * self.input_size * 3
        self.step = 0
        self.projecting_mask = self.mask()

        self.example_input_array = torch.randn(64, 1, self.input_size, self.input_size).to(device)

        self.example_input_array = self.Embedding_Function.embed(self.example_input_array)

        self.rand_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        self.criterion = nn.MSELoss()
        self.eps = 0.00000000000001

    def get_input_size(self):
        return self.input_size

    def encode(self, x):  
        x = self.encoder(x)
        return x
    def decode(self, x):  
        x = self.decoder(x)
        return x

    def forward(self, x):

        z = self.encode(x)
        z = self.decode(z)

        return z

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            x = batch

            z = self.Embedding_Function.embed(x)

        out = self.forward(z)
        loss = self.criterion(out,z) 

        img = self.Embedding_Function.decode(out)
        img_loss = self.criterion(img,x*self.projecting_mask)
        img_loss_track = self.criterion(torch.clip(img,0,1)*self.projecting_mask,x*self.projecting_mask)
        img_loss_projection = self.criterion(img,self.Embedding_Function.decode(z))
        self.log("ImageMSE",img_loss_track)
        self.log("ImageMSE_project",img_loss_projection)
        self.log("train_loss", loss, prog_bar=True)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])

        return img_loss



    def configure_optimizers(self):
        """Default Adam optimizer if missing from the configuration file."""
        return Adam(self.parameters(), lr=1e-3)

    def project(self, images):
        pre_model = images
        x = self.Embedding_Function.embed(pre_model).unsqueeze(1)
        z = self.encode(x)
        return z


    def mask(self):

        grid_extend = 1
        z = x = np.linspace(-grid_extend, grid_extend, self.image_size)
        z, x = np.meshgrid(z, x)
        y = (x**2+z**2)
        y =  np.where(y<1,1,0)

        return torch.tensor(y,device='cuda')

