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

from power_spherical import HypersphericalUniform, PowerSpherical

class VAE(pl.LightningModule):
    def __init__(
        self,
        image_size: int = 28,
        n_max = 32,
        device='cuda'
    ):
        super().__init__()

        self.beta = 0.001




        self.Embedding_Function = ZernikeEmbedding(image_size, n_max, device)

        self.encoder = ZernikeEncoderAuto(n_max, 2, 3, 8*2, device=device, image_size=image_size,Very_small=True,Very_large=True)
        self.decoder = ZernikeDecoderAuto(n_max, 2, 3, 8*2, device=device, image_size=image_size,Very_small=True,Very_large=True)
        self.z_dim = 2
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


        self.fc_location = nn.Linear(2, 3)
        self.inv_fc_location = nn.Linear(3,2)
        self.fc_scale = nn.Linear(2, 1)

    def get_input_size(self):
        return self.input_size

    def encode(self, x):
        x = self.encoder(x)
        return x
    def decode(self, x):
        x = self.decoder(x)
        return x

    def reparameterize(self, z_location, z_scale):
        q_z = PowerSpherical(z_location, z_scale)
        p_z = HypersphericalUniform(self.z_dim, device=z_location.device)
        return q_z, p_z

    def dezernify(self, x):
        # breakpoint()
        angle = x[...,1,0]/(x[...,1,1]+x[...,1,0]+self.eps)
        x = torch.sum(x**2, dim=-1,keepdim=False)
        # mean = torch.sqrt(torch.sum(x, dim=-1,keepdim=False)).unsqueeze(-1)
        x = torch.sqrt(x)
        # x = x/mean
        return x, angle
    def zernify(self, z, angle):
        z = z.unsqueeze(-1)
        z2 = z.clone()
        z2[...,1,0] = z[...,1,0]*(1-angle)
        z3 = z.clone()
        z3[...,1,0] = z[...,1,0]*angle
        z2[...,0,0] *= 0
        # z =torch.cat((z,z2),dim=-1)
        z =torch.cat((z3,z2),dim=-1)
        return z

    def make_correct_format(self, x):
        z_location = self.fc_location(x)
        z_location = torch.nn.functional.normalize(z_location, p=2.0, dim=-1)
        # SVAE code: the `+ 1` prevent collapsing behaviors
        z_scale = F.softplus(self.fc_scale(x)) + 1e-5
        return z_location, z_scale

    def forward(self, x):
        # with torch.autograd.set_detect_anomaly(True):
        z = self.encode(x)
        # print(z.size())
        z_location, angle = self.dezernify(z)

        # z_scale = F.softplus(z_scale) + 1e-3
        z_location, z_scale = self.make_correct_format(z_location)



        q_z, p_z = self.reparameterize(z_location, z_scale.squeeze(-1))
        z_out = q_z.rsample()


        z_out = self.inv_fc_location(z_out)
        '''

        z_location = self.fc_location(z_location)
        z_location = torch.nn.functional.normalize(z_location, p=2.0, dim=-1)

        z_out = z_location
        z_out = self.inv_fc_location(z_out)

        '''
        z_out = self.zernify(z_out, angle)

        # print(z_out.size())

        recon = self.decode(z_out)

        # print(recon.size())

        return recon#,q_z, p_z

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            x = batch

            z = self.Embedding_Function.embed(x)

        out = self.forward(z)
        # out,q_z, p_z = self.forward(z)
        loss = self.criterion(out,z)
        # loss_KL = self.beta * torch.distributions.kl.kl_divergence(q_z, p_z).mean()


        img = self.Embedding_Function.decode(out)
        img_loss = self.criterion(img,x*self.projecting_mask)
        img_loss_track = self.criterion(torch.clip(img,0,1)*self.projecting_mask,x*self.projecting_mask)
        img_loss_projection = self.criterion(img,self.Embedding_Function.decode(z))
        self.log("ImageMSE",img_loss_track)
        self.log("ImageMSE_project",img_loss_projection)
        self.log("train_loss", loss, prog_bar=True)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        # self.log("KL_loss", loss_KL)

        return img_loss#+loss_KL



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
