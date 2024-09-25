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
from sympy import ord0
#from scipy.constants import physical_constants
from torch.optim import Adam

from .convolutional_encoder import ConvolutionalEncoder
#import lightning.pytorch as pl
from .spherinator_module import SpherinatorModule
from .zernike_decoder import ZernikeDecoder
from .zernike_encoder import ZernikeEncoder
from .zernike_encoder_classify import ZernikeEncoderClassify


class ZernikeClassifier(SpherinatorModule):
    def __init__(
        self,
        encoder: nn.Module = ZernikeEncoderClassify(8,0,10,device = 'cuda:0'),
        #encoder: nn.Module = ConvolutionalEncoder(2),
        #encoder: nn.Module = ZernikeEncoder(32,1,10,device = 'cuda:2'),
        #decoder: nn.Module = ZernikeDecoder(32,1,10,device = 'cuda:2'),
        image_size: int = 91,
        input_size: int = 28,
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
        #encoder = ZernikeEncoderClassify(8,0,10,device = 'cuda:2')

        #print(self.device)
        device = 'cuda:0'
        self.input_mask= self.mask().to(device)
        self.encoder = encoder
        #self.decoder = decoder
        self.image_size = image_size
        self.input_size = input_size
        self.rotations = rotations
        self.beta = beta
        self.reduce_size = False
        self.crop_size = int(self.image_size * math.sqrt(2) / 2)
        self.total_input_size = self.input_size * self.input_size * 3
        self.step = 0

        self.example_input_array = torch.randn(2, 1, self.input_size, self.input_size)

        self.criterion2 = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()
        self.Embedding_Function = Zernike_embedding(32, device)
        self.mask = self.encoder.Product0.create_mask_decrease(32,8)
        self.mask2 = self.encoder.Product0.create_mask_increase(16,32)
        self.dropout = nn.Dropout(p=0.3)


    def get_input_size(self):
        return self.input_size

    def encode(self, x):
        x = self.encoder(x)


        return x


    def forward(self, x):
        #sum_in = torch.sum(torch.abs(x),dim=(-1,-2,-3))

        #eps = np.finfo(float).eps
        with torch.no_grad():
            x = self.Embedding_Function.embed(x)
            x  = x[:,:,self.mask,:]

        z = self.encode(x)
        #norm = torch.sum(torch.abs(z),dim=1,keepdim=True)+eps
        #print(norm)
        #z = z/norm
        #print(z)
        #z = F.softmax(z,dim=1)
        z = F.sigmoid(z)
        #norm = torch.sum(torch.abs(z),dim=1,keepdim=True)+eps
        #z = z/norm
        #z = F.log_softmax(z, dim=1)
        #z = self.calculate_normalized_outputs(z)
        return z,x


    def training_step(self, batch, batch_idx):

        #sum_in = torch.sum(torch.abs(picture),dim=(-1,-2,-3))
        #picture = picture/sum_in
        with torch.no_grad():
            #print(batch)
            x = batch[0]
            y = batch[1]
            #print(y)
            #compare = torch.ones_like(y,dtype=torch.float)
            y = F.one_hot(y,num_classes=10).to(torch.float)
            # tens= torch.zeros(29,29, device='cuda:2')
            # for i in range(29):
            #     for j in range(29):
            #         tens[i,j]= j-14
            # magnitude = torch.sum(x,dim=(-1,-2))
            # print(torch.max(torch.einsum('...ij,ij->...',x,tens)/(magnitude)))
            #x = x + torch.randn(29,29,device='cuda:3')/2
            #x = x * random.random()
            picture = torch.einsum('...ij,ij->...ij',x,self.input_mask)
            #picture = picture/torch.sum(torch.abs(picture),dim=(-1,-2),keepdim=True)
            #print(y)
        #picture = self.dropout(picture)
        #print( torch.sum(torch.abs(picture),dim=(-1,-2),keepdim=True)[0])
        #picture = picture/norm
        out,_ = self.forward(picture)

        #lr = self.optimizers().param_groups[0]["lr"]
        #loss = self.criterion(self.Decoding_Function(out),picture)
        '''
        if lr <   0.001:

            loss = self.criterion(self.Decoding_Function(out),picture)
        else:
            loss = self.criterion(x, out)
        '''
        #out = self.Decoding_Function(out)
        #picture = self.Decoding_Function(x)
        #print(torch.sum(torch.abs(picture),dim=(-1,-2))[0])
        #norm = torch.sum(torch.abs(out),dim=(-1,-2),keepdim=True)
        #out = out/norm
        #loss = self.criterion(out,picture)
        #print(y.size(),out[:,:,0,0].squeeze().size())
        # print(self.criterion( out,y))
        # print( (torch.max(out,dim=-1).values))
        # print( (torch.max(out,dim=-1).values).mean())
        # print(y[0])
        # print(out[0])
        # print(self.calculate_normalized_outputs(out)[0])
        #loss = torch.sqrt(self.criterion( out,y))
        loss = (self.criterion( out,y))
         #- (torch.min(out,dim=-1).values).mean()/10
        #loss = 50* self.criterion( out,y) + self.criterion2(out,y)
        # if self.step > 40:
        #     # if self.step< 60:
        #     #     print(self.optimizers().param_groups[0]["lr"])
        #     self.optimizers().param_groups[0]["lr"] = 0.001
        # else:
        #     loss = self.criterion( out,y)


        #+ (0.31*(torch.abs(torch.sum(out,dim=1)-1))).mean()
        #loss = F.nll_loss(out[:,:,0,0].squeeze(),y)
        # if loss <0.15 and self.optimizers().param_groups[0]["lr"] > 0.001:
        #     self.optimizers().param_groups[0]["lr"] = 0.001

        #print(y[0])
        #print(out[0,:,0,0])
        # with torch.no_grad():
        #     #out = torch.max(out)
        #     #print(out[0])
        #     out = torch.argmax(out, dim=1)#, keepdim=True)
        #     #print(out[0])
        #     y = torch.abs(y-1)
        #     #print(y[0])
        #     #out = y[out]#.mean()
        #     #print(out.size())
        #     x = 0
        #     for i in range(1024):
        #         #print(y[i,out[i]])
        #         x += y[i,out[i]]
        #     x = x/1024
        #     #print(x)

            #out = torch.abs((y-out)[y==1]).mean()
        self.log("train_loss", loss, prog_bar=True)
        #self.log("prediction_error", x, prog_bar=True)
        #self.log("image_loss", self.criterion(out_pic, picture), prog_bar=True)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, batch_idx):

        #sum_in = torch.sum(torch.abs(picture),dim=(-1,-2,-3))
        #picture = picture/sum_in
        with torch.no_grad():
            x = batch[0]
            y = batch[1]
            #y = y[:,0]
        #compare = torch.ones_like(y,dtype=torch.float)
        y = F.one_hot(y).to(torch.float)
        #x = x# + torch.ones(29,29,device='cuda:3')/4
        with torch.no_grad():
            #picture = torch.einsum('...ij,ij->...ij',x,self.input_mask)
            #print( torch.sum(torch.abs(picture),dim=(-1,-2),keepdim=True)[0])
            #picture = picture/norm
            out,_ = self.forward(x)

            loss = (self.criterion( out,y))
        with torch.no_grad():
            #out = torch.max(out)
            #print(out[0])
            out = torch.argmax(out, dim=1)#, keepdim=True)
            #print(out[0])
            y = torch.abs(y-1)
            #print(y[0])
            #out = y[out]#.mean()
            #print(out.size())
            size = out.size(0)
            x = 0
            for i in range(size):
                #print(y[i,out[i]])
                x += y[i,out[i]]
            x = x/size
            #print(x)
        self.log("validation_loss", loss, prog_bar=True)
        self.log("validation_accuracy",x, prog_bar=True)
        #self.log("Max value avarage", (torch.max(out,dim=-1).values).mean(), prog_bar=True)
        #self.log("image_loss", self.criterion(out_pic, picture), prog_bar=True)
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
        z = x = np.linspace(-grid_extend, grid_extend, 28)
        z, x = np.meshgrid(z, x)
        y = (x**2+z**2)
        y =  np.where(y<1,1,0)
        return torch.tensor(y)

class Zernike_embedding(nn.Module):
    def __init__(self, n_max = 30 , device = 'cuda:2', numerical_expand = 4 ):
        super().__init__()
        self.num = numerical_expand
        if os.path.isfile('Zernike_decode_encode_size28_{}'.format(n_max)) :
            self.Zernike_matrix = torch.load('Zernike_decode_encode_size28_{}'.format(n_max))
        else:
            self.Zernike_matrix = self.create_filter(n_max)
            torch.save(self.Zernike_matrix,'Zernike_decode_encode_size28_{}'.format(n_max))
        #size = self.calc_size(n_max)


        #self.Zernike_matrix = self.create_filter(n_max)
        self.Zernike_matrix = self.Zernike_matrix.to(device)#*16
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
        z = x = np.linspace(-grid_extend, grid_extend, int(28*self.num))
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


        z = x = np.linspace(-grid_extend, grid_extend, int(28))
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
