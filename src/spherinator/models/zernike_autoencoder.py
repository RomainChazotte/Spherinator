import argparse
import math
import random

import numpy
import numpy as np
import scipy
import scipy.special as sp
import seaborn as sns
import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as functional
from numpy import convolve
from power_spherical import HypersphericalUniform, PowerSpherical
from scipy.constants import physical_constants
from torch.optim import Adam

from .spherinator_module import SpherinatorModule
from .zernike_decoder import ConvolutionalDecoder
from .zernike_encoder import ConvolutionalEncoder


class ZernikeAutoencoder(SpherinatorModule):
    def __init__(
        self,
        encoder: nn.Module = ConvolutionalEncoder(),
        decoder: nn.Module = ConvolutionalDecoder(),
        h_dim: int = 256,
        z_dim: int = 2,
        image_size: int = 91,
        input_size: int = 128,
        rotations: int = 36,
        beta: float = 1.0,
    ):
        """RotationalVariationalAutoencoderPower initializer

        Args:
            h_dim (int, optional): dimension of the hidden layers. Defaults to 256.
            z_dim (int, optional): dimension of the latent representation. Defaults to 2.
            image_size (int, optional): size of the input images. Defaults to 91.
            rotations (int, optional): number of rotations. Defaults to 36.
            beta (float, optional): factor for beta-VAE. Defaults to 1.0.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])

        self.encoder = encoder
        self.decoder = decoder
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.image_size = image_size
        self.input_size = input_size
        self.rotations = rotations
        self.beta = beta
        self.reduce_size = False
        self.crop_size = int(self.image_size * math.sqrt(2) / 2)
        self.total_input_size = self.input_size * self.input_size * 3

        self.example_input_array = torch.randn(1, 3, self.input_size, self.input_size)
        self.Zernike_size = int(((33+1)*33/2)/2+math.ceil(33/4))
        self.fc_location = nn.Linear(h_dim, z_dim)
        self.fc_scale = nn.Linear(h_dim, 1)
        self.fc2 = nn.Linear(z_dim, h_dim)
        self.criterion = nn.L1Loss()
        self.Embedding_Function = Zernike_embedding(32)
        self.Decoding_Function =  Zernike_decode(32)

        with torch.no_grad():
            self.fc_scale.bias.fill_(1.0e3)

    def get_input_size(self):
        return self.input_size

    def encode(self, x):
        x = self.encoder(x)


        return x

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        x = self.Embedding_Function(x)
        z = self.encode(x)
        recon = self.decode(z)
        return recon,x


    def training_step(self, batch, batch_idx):
        with torch.no_grad():


            crop = functional.center_crop(batch, [self.crop_size, self.crop_size])
            scaled = functional.resize(
                crop, [self.input_size, self.input_size], antialias=True
            )

        out,x = self.forward(scaled)

        loss = self.criterion(x, out)

        self.log("train_loss", loss, prog_bar=True)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        return loss

    def configure_optimizers(self):
        """Default Adam optimizer if missing from the configuration file."""
        return Adam(self.parameters(), lr=1e-3)

    def project(self, images):
        #pre_model = torch.sum(images,-3)
        pre_model = images
        x = self.Embedding_Function(pre_model).unsqueeze(1)
        #print('start')
        #print(x)

        z = self.encode(x)
        return z

    def reconstruct(self, coordinates):
        return self.Decoding_Function(self.decode(coordinates))



class Zernike_embedding(nn.Module):
    def __init__(self, n_max = 30 ):
        super().__init__()
        self.Zernike_matrix = self.create_filter(n_max)
        size = self.calc_size(n_max)
        #self.device = 'cuda:2'
    def calc_size(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        return lengh

    def M_embedding_generator(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        Basis = np.zeros(lengh)
        for m1 in range(0, n_max+1):
            m1_lengh = lengh - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
            count=0
            for n1 in range(m1,n_max+1,2):
                Basis[m1_lengh+count] = m1
                count+=1
        return Basis


    def Radial_function(self,n,m, n_max):
        faktor = []
        scaling = []
        for i in range(n_max+n_max+1):
            scaling.append(1/((2*n_max-i)**2+2))

        for i in range(n_max-n):
            faktor.append(0)

        for k in range(int((n-m)/2+1)):
            faktor.append((-1)**k * math.factorial(n-k) /(math.factorial(k) * math.factorial(int((n+m)/2-k))* math.factorial(int((n-m)/2-k)))   )
            if k != int((n-m)/2):
                faktor.append(0)
            #exp.append(n-2*k)

        for i in range(m):
            faktor.append(0)
        scale = convolve(faktor,faktor)
        scale = np.einsum('i,i', scaling,scale)

        faktor = np.array(faktor/scale)
        #faktor = np.array(faktor)
        return np.flip(faktor)

    def Zernicke_embedding_generator(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        Basis = np.zeros((lengh,n_max+1))
        for m1 in range(0, n_max+1):
            m1_lengh = lengh - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
            count=0
            for n1 in range(m1,n_max+1,2):
                Basis[m1_lengh+count,:] = self.Radial_function(n1,m1,n_max)
                count+=1
        return Basis
    def mask(self,x,z):
        y = (x**2+z**2)
        return np.where(y<1,1,0)


    def create_filter(self,n_max):
        Zernike_functions = self.Zernicke_embedding_generator(n_max)

        grid_extend = 1
        #grid_resolution = 680
        z = x = np.linspace(-grid_extend, grid_extend, 128)
        z, x = np.meshgrid(z, x)

        #print(Zernike_functions)
        # Use epsilon to avoid division by zero during angle calculations
        functions = []
        for i in range(len(Zernike_functions)):

            functions.append(numpy.polynomial.polynomial.Polynomial(Zernike_functions[i]))

        eps = np.finfo(float).eps
        out = []
        M = self.M_embedding_generator(n_max)
        for i in range(len(Zernike_functions)):
            out.append([functions[i](np.sqrt((x ** 2 + z ** 2)))*np.cos(M[i]*np.arctan2(x , (z ))),functions[i](np.sqrt((x ** 2 + z ** 2)))*np.sin(M[i]*np.arctan2(x ,(z  )))])
        #print(out[0])
        # Add restriction to r<1
        out_mask = self.mask(x,z)
        out = torch.tensor(np.array(out*out_mask),dtype=torch.float, device =  'cuda:2')

        norm = []
        for i in range(len(Zernike_functions)):
            norm.append([functions[i](np.sqrt((x ** 2 + z ** 2)))*np.cos(M[i]*np.arctan2(x , (z ))),functions[i](np.sqrt((x ** 2 + z ** 2)))*np.cos(M[i]*np.arctan2(x ,(z  )))])

        norm = torch.tensor(np.array(norm*out_mask),dtype=torch.float, device =  'cuda:2')

        norm = (torch.sum(torch.abs(norm),dim= (-1,-2),keepdim = True))
        return out/norm

    def forward(self,input):
        #norm = (torch.sum(torch.abs(self.Zernike_matrix),dim= (-1,-2),keepdim = True))
        #eps = 0.0000005
        #self.Zernike_matrix = self.Zernike_matrix/(norm+eps)
        #This should be implemented in init, do this later
        #print(self.Zernike_matrix.size())
        #print(input.size())
        out = torch.einsum('ijkl,...kl->...ij',self.Zernike_matrix,input)
        #print(out.size())
        return out*100




class Zernike_decode(nn.Module):
    def __init__(self, n_max = 30 ):
        super().__init__()
        self.Zernike_matrix = self.create_filter(n_max)
        size = self.calc_size(n_max)

    def calc_size(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        return lengh

    def M_embedding_generator(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        Basis = np.zeros(lengh)
        for m1 in range(0, n_max+1):
            m1_lengh = lengh - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
            count=0
            for n1 in range(m1,n_max+1,2):
                Basis[m1_lengh+count] = m1
                count+=1
        return Basis


    def Radial_function(self,n,m, n_max):
        faktor = []
        scaling = []
        for i in range(n_max+n_max+1):
            scaling.append(1/((2*n_max-i)**2+2))

        for i in range(n_max-n):
            faktor.append(0)

        for k in range(int((n-m)/2+1)):
            faktor.append((-1)**k * math.factorial(n-k) /(math.factorial(k) * math.factorial(int((n+m)/2-k))* math.factorial(int((n-m)/2-k)))   )
            if k != int((n-m)/2):
                faktor.append(0)
            #exp.append(n-2*k)

        for i in range(m):
            faktor.append(0)
        scale = convolve(faktor,faktor)
        scale = np.einsum('i,i', scaling,scale)

        faktor = np.array(faktor/scale)
        #faktor = np.array(faktor)
        return np.flip(faktor)

    def Zernicke_embedding_generator(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        Basis = np.zeros((lengh,n_max+1))
        for m1 in range(0, n_max+1):
            m1_lengh = lengh - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
            count=0
            for n1 in range(m1,n_max+1,2):
                Basis[m1_lengh+count,:] = self.Radial_function(n1,m1,n_max)
                count+=1
        return Basis
    def mask(self,x,z):
        y = (x**2+z**2)
        return np.where(y<1,1,0)


    def create_filter(self,n_max):
        Zernike_functions = self.Zernicke_embedding_generator(n_max)

        grid_extend = 1
        #grid_resolution = 680
        z = x = np.linspace(-grid_extend, grid_extend, 128)
        z, x = np.meshgrid(z, x)

        #print(Zernike_functions)
        # Use epsilon to avoid division by zero during angle calculations
        functions = []
        for i in range(len(Zernike_functions)):

            functions.append(numpy.polynomial.polynomial.Polynomial(Zernike_functions[i]))

        eps = np.finfo(float).eps
        out = []
        M = self.M_embedding_generator(n_max)
        for i in range(len(Zernike_functions)):
            out.append([functions[i](np.sqrt((x ** 2 + z ** 2)))*np.cos(M[i]*np.arctan2(x , (z ))),functions[i](np.sqrt((x ** 2 + z ** 2)))*np.sin(M[i]*np.arctan2(x ,(z  )))])
        #print(out[0])
        # Add restriction to r<1
        out_mask = self.mask(x,z)
        out = torch.tensor(np.array(out*out_mask),dtype=torch.float, device =  'cuda:2')
        norm = []
        for i in range(len(Zernike_functions)):
            norm.append([functions[i](np.sqrt((x ** 2 + z ** 2)))*np.cos(M[i]*np.arctan2(x , (z ))),functions[i](np.sqrt((x ** 2 + z ** 2)))*np.cos(M[i]*np.arctan2(x ,(z  )))])

        norm = torch.tensor(np.array(norm*out_mask),dtype=torch.float, device =  'cuda:2')

        norm = (torch.sum(torch.abs(norm),dim= (-1,-2),keepdim = True))
        return out/norm

    def forward(self,input):
        #norm = (torch.sum(torch.abs(self.Zernike_matrix),dim= (-1,-2),keepdim = True))
        #eps = 0.0000005
        #self.Zernike_matrix = self.Zernike_matrix/(norm+eps)
        #This should be implemented in init, do this later
        out = torch.einsum('ijkl,...ij->...kl',self.Zernike_matrix,input)
        return out*100