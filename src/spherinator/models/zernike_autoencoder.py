import argparse
import math
import os.path
import random

import numpy
import numpy as np
#import scipy
#import scipy.special as sp
#import seaborn as sns
import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as functional
from numpy import convolve
from power_spherical import HypersphericalUniform, PowerSpherical
from skimage.measure import block_reduce
#from scipy.constants import physical_constants
from torch.optim import Adam

#import lightning.pytorch as pl
from .spherinator_module import SpherinatorModule
from .zernike_decoder import ZernikeDecoder
from .zernike_encoder import ZernikeEncoder


class ZernikeAutoencoder(SpherinatorModule):
    def __init__(
        self,
        encoder: nn.Module = ZernikeEncoder(32,1,10,device = 'cuda:2'),
        decoder: nn.Module = ZernikeDecoder(32,1,10,device = 'cuda:2'),
        image_size: int = 91,
        input_size: int = 128,
        rotations: int = 36,
        beta: float = 1.0,
        n_max = 32
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


        #print(self.device)
        device = 'cuda:2'
        self.Embedding_Function = Zernike_embedding(32, device, numerical_expand = 16)
        #decoder = ZernikeDecoder(32,1,10,device = 'cpu')
        #encoder =  ZernikeEncoder(32,1,10,device = 'cpu')
        self.input_mask= self.mask().to(device)
        self.encoder = encoder
        self.decoder = decoder
        self.image_size = image_size
        self.input_size = input_size
        self.rotations = rotations
        self.beta = beta
        self.reduce_size = False
        self.crop_size = int(self.image_size * math.sqrt(2) / 2)
        self.total_input_size = self.input_size * self.input_size * 3

        self.example_input_array = torch.randn(2, 3, self.input_size, self.input_size)

        self.criterion = nn.L1Loss()
        self.criterion2 = nn.MSELoss()
        #self.Embedding_Function = Zernike_embedding(32, device, numerical_expand = 16)
        #self.Decoding_Function =  Zernike_decode(32, device)


    def get_input_size(self):
        return self.input_size

    def encode(self, x):
        x = self.encoder(x)


        return x

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, picture):
        #sum_in = torch.sum(torch.abs(x),dim=(-1,-2,-3))

        #print(torch.cuda.mem_get_info())
        #print('encode')
        with torch.no_grad():
            x = self.Embedding_Function.embed(picture)
        #print(torch.cuda.mem_get_info())
        #import pdb
        #breakpoint()
            #print((x.detach())[0,0,i:i+10])
        '''
        print(y.detach())
        print(x.detach().size())
        print(x.detach()[0,:,0:3])
        print(y.detach()[0,:,0:3])
        '''

        # with torch.no_grad():
        #     print(torch.sum(torch.abs(x)))
        z = self.encode(x)
        # with torch.no_grad():
        #     print(torch.sum(torch.abs(z)))
        #print('encoding_layer')
        #print(torch.cuda.mem_get_info())
        recon = self.decode(z)
        # with torch.no_grad():
        #     print(torch.sum(torch.abs(recon)))
        #print('decoded')
        #print(torch.cuda.mem_get_info())
        #sum_out = torch.sum(torch.abs(recon),dim=(-1,-2,-3))
        #eps = 0.00000000000000000000000001
        #recon = torch.einsum('ijkl,i,i->ijkl',recon,1/(sum_out+eps),sum_in)
        return recon,x


    def training_step(self, batch, batch_idx):
        #torch.autograd.set_detect_anomaly(True)

        with torch.no_grad():
            crop = functional.center_crop(batch, [self.crop_size, self.crop_size])
            picture = functional.resize(
                crop, [self.input_size, self.input_size], antialias=True
            )
            picture = torch.einsum('...ij,ij->...ij',picture,self.input_mask)
            picture = picture/torch.sum(torch.abs(picture),dim =(-1,-2),keepdim=True)
        #sum_in = torch.sum(torch.abs(picture),dim=(-1,-2,-3))
        #picture = picture/sum_in

        #print( torch.sum(torch.abs(picture),dim=(-1,-2),keepdim=True)[0])
        #picture = picture/norm
        out,x = self.forward(picture)

        #lr = self.optimizers().param_groups[0]["lr"]
        #loss = self.criterion(self.Decoding_Function(out),picture)


        # out = self.Decoding_Function(out)
        # picture = self.Decoding_Function(x)
        # print(torch.sum(torch.abs(picture),dim=(-1,-2))[0])
        # norm = torch.sum(torch.abs(out),dim=(-1,-2),keepdim=True)
        # out = out/norm
        # loss = self.criterion(out,picture)
        # loss = self.criterion(x, out)

        #input_picture = self.Decoding_Function(x)
        #norm = torch.sum(torch.abs(picture),dim=(-1,-2),keepdim=True)

        with torch.no_grad():
            #print(torch.sum(torch.abs(x)), torch.sum(torch.abs(out)))
            out_pic = self.Embedding_Function.decode(out)
            #rec_pic = self.Embedding_Function.decode(x)
            #print(torch.sum(torch.abs(picture)), torch.sum(torch.abs(out_pic)), torch.sum(torch.abs(rec_pic)))


        #out_norm = torch.sum(torch.abs(out_pic),dim=(-1,-2),keepdim=True)
        #out_pic = out_pic*norm/out_norm


        #loss = self.criterion(out_pic, input_picture)
        # if lr <   0.0002:

        #     loss = self.criterion(input_picture,out_pic)
        # else:
        #     loss = self.criterion2(input_picture,out_pic)

        #out_pic = self.Embedding_Function.decode(out)
        #loss = self.criterion(out_pic, picture)*1000
        loss = self.criterion(x, out)*1000

        #loss = torch.mean(torch.square(out_pic-input_picture),dim=(-1,-2,-3))
        #loss = torch.mean(loss)
        self.log("train_loss", loss, prog_bar=True)
        self.log("image_loss", self.criterion(out_pic, picture)*1000, prog_bar=True)
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
    def reconstruction_loss(self,x):
        print('please implement a reconstruction loss')
        return x
    def reconstruct(self, coordinates):
        return self.Decoding_Function(self.decode(coordinates))

    def mask(self):

        grid_extend = 1
        #grid_resolution = 680
        z = x = np.linspace(-grid_extend, grid_extend, 128)
        z, x = np.meshgrid(z, x)
        y = (x**2+z**2)
        y =  np.where(y<1,1,0)
        return torch.tensor(y)

class Zernike_embedding(nn.Module):
    def __init__(self, n_max = 30 , device = 'cuda:2', numerical_expand = 16 ):
        super().__init__()
        self.num = numerical_expand
        if os.path.isfile('Zernike_decode_encode{}'.format(n_max)) :
            self.Zernike_matrix = torch.load('Zernike_decode_encode{}'.format(n_max))
        else:
            self.Zernike_matrix = self.create_filter(n_max)
            torch.save(self.Zernike_matrix,'Zernike_decode_encode{}'.format(n_max))
        #size = self.calc_size(n_max)


        #self.Zernike_matrix = self.create_filter(n_max)
        self.Zernike_matrix = self.Zernike_matrix.to(device)*16
        #self.norm_matrix = np.array(self.norm_matrix)
        #self.Zernike_matrix= torch.nn.parameter.Parameter(self.Zernike_matrix,requires_grad=False)
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
        z = x = np.linspace(-grid_extend, grid_extend, int(128*self.num))
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
        #out_mask = self.mask(x,z)
        #out = torch.tensor(np.array(out*out_mask),dtype=torch.float)#, device =  'cuda:2')

        # norm = []
        # for i in range(len(Zernike_functions)):
        #     norm.append([functions[i](np.sqrt((x ** 2 + z ** 2)))*np.cos(M[i]*np.arctan2(x , (z )))+eps,functions[i](np.sqrt((x ** 2 + z ** 2)))*np.sin(M[i]*np.arctan2(x ,(z  )))+eps])

        # norm = torch.tensor(np.array(norm*out_mask),dtype=torch.float)

        # norm = torch.sqrt((torch.sum((norm)**2,dim= (-1,-2),keepdim = True)))*self.num


        # out = out/norm
        out = np.array(out)
        out =torch.tensor( block_reduce(out,(1,1, self.num, self.num),func=np.sum),dtype=torch.float)


        z = x = np.linspace(-grid_extend, grid_extend, int(128))
        z, x = np.meshgrid(z, x)
        out_mask = torch.tensor( self.mask(x,z))
        out= torch.einsum('ijkl,kl->ijkl',out,out_mask)

        norm = torch.sqrt((torch.sum((out)**2,dim= (-1,-2),keepdim = True)))+eps
        out = out/norm
        return out#*self.num

    def embed(self,input):
        out = torch.einsum('ijkl,...kl->...ij',self.Zernike_matrix,input)
        return out

    def decode(self,input):
        out = torch.einsum('ijkl,...ij->...kl',self.Zernike_matrix,input)
        return out
