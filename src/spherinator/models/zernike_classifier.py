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
from .zernike_encoder_classify import ZernikeEncoderClassify


class ZernikeClassifier(SpherinatorModule):
    def __init__(
        self,
        encoder: nn.Module = ZernikeEncoderClassify(8,0,10,device = 'cuda:3'),
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
        device = 'cuda:3'
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

        #self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()
        self.Embedding_Function = Fourier_embedding(16, device)
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

        z = self.encode(x)
        #norm = torch.sum(torch.abs(z),dim=1,keepdim=True)+eps
        #print(norm)
        #z = z/norm
        #print(z)
        z = F.softmax(z,dim=1)
        #z = F.sigmoid(z)
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
            y2 = y
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
            accuracy = self.accuracy(out,y2)
            x = -(accuracy-1)
            num_wrong = self.num_wrong(out,y2)
            # out = torch.argmax(out, dim=1)#, keepdim=True)
            # #print(out[0])
            # y = torch.abs(y-1)
            # #print(y[0])
            # #out = y[out]#.mean()
            # #print(out.size())
            # size = out.size(0)
            # x = 0
            # for i in range(size):
            #     #print(y[i,out[i]])
            #     x += y[i,out[i]]
            # x = x/size
            # #print(x)
        self.log("validation_loss", loss, prog_bar=True)
        self.log("validation_accuracy",x, prog_bar=True)
        #self.log("Max value avarage", (torch.max(out,dim=-1).values).mean(), prog_bar=True)
        #self.log("image_loss", self.criterion(out_pic, picture), prog_bar=True)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        #return loss

    def test_step(self, batch, batch_idx):

        #sum_in = torch.sum(torch.abs(picture),dim=(-1,-2,-3))
        #picture = picture/sum_in
        with torch.no_grad():
            x = batch[0]
            y = batch[1]
            #y = y[:,0]
            y2 = y
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
            accuracy = self.accuracy(out,y2)
            x = -(accuracy-1)
            num_wrong = self.num_wrong(out,y2)
            #num_right = # out = torch.argmax(out, dim=1)#, keepdim=True)
            # #print(out[0])
            # y = torch.abs(y-1)
            # #print(y[0])
            # #out = y[out]#.mean()
            # #print(out.size())
            # size = out.size(0)
            # x = 0
            # for i in range(size):
            #     #print(y[i,out[i]])
            #     x += y[i,out[i]]
            # x = x/size
            # #print(x)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy",x, prog_bar=True)
        self.log("num_wrong",num_wrong, prog_bar=True)
        #self.log("num_right",num_right, prog_bar=True)

        #self.log("Max value avarage", (torch.max(out,dim=-1).values).mean(), prog_bar=True)
        #self.log("image_loss", self.criterion(out_pic, picture), prog_bar=True)
        #self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        #return loss

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

    def accuracy(self,predictions, targets):
        if predictions.shape[1] > 1:
            predictions = predictions.argmax(dim=1)
        else:
            predictions = (predictions > 0.)

        predictions = predictions.to(dtype=targets.dtype)
        accuracy = float((targets == predictions).sum()) / predictions.numel()
        return accuracy

    def num_wrong(self,predictions, targets):
        if predictions.shape[1] > 1:
            predictions = predictions.argmax(dim=1)
        else:
            predictions = (predictions > 0.)

        predictions = predictions.to(dtype=targets.dtype)
        accuracy = float((targets == predictions).sum()) #/ predictions.numel()
        return (predictions.numel() -accuracy)#, accuracy
class Fourier_embedding(nn.Module):
    def __init__(self, n_max = 30 , device = 'cuda:2', numerical_expand = 4 ):
        super().__init__()
        self.num = numerical_expand
        if os.path.isfile('Fourier_decode_encode_size28_{}'.format(n_max)) :
            self.Fourier_matrix = torch.load('Fourier_decode_encode_size28_{}'.format(n_max))
        else:
            self.Fourier_matrix = self.create_filter(n_max+1)
            torch.save(self.Fourier_matrix,'Fourier_decode_encode_size28_{}'.format(n_max))
        #size = self.calc_size(n_max)


        #self.Fourier_matrix = self.create_filter(n_max)
        self.Fourier_matrix = self.Fourier_matrix.to(device)#*16
        self.Fourier_matrix = torch.transpose(self.Fourier_matrix, 1,2)
        #self.norm_matrix = np.array(self.norm_matrix)
        #self.Fourier_matrix= torch.nn.parameter.Parameter(self.Fourier_matrix,requires_grad=False)
        #self.device = 'cuda:2'


    def create_filter(self,n_max):
        grid_extend = 1
        #grid_resolution = 680
        z = x = np.linspace(-grid_extend, grid_extend, int(28*self.num))
        z, x = np.meshgrid(z, x)


        eps = np.finfo(float).eps
        out = np.empty((n_max,n_max,2,2,int(28*self.num),int(28*self.num)))
        for i in range(n_max):
            for j in range(n_max):
                out[i,j] = [[np.cos(i*x)*np.cos(j*z),np.cos(i*x)*np.sin(j*z)],[np.sin(i*x)*np.cos(j*z),np.sin(i*x)*np.sin(j*z)]]

        out = np.array(out)
        out =torch.tensor( block_reduce(out,(1,1,1,1, self.num, self.num),func=np.sum),dtype=torch.float)



        norm = torch.sqrt((torch.sum((out)**2,dim= (-1,-2),keepdim = True)))+eps
        out = out/norm
        return out#*self.num

    def embed(self,input):
        #print(self.Fourier_matrix.size())

        out = torch.einsum('abijkl,...kl->...abij',self.Fourier_matrix,input)
        return out

    def decode(self,input):
        out = torch.einsum('abijkl,...abij->...kl',self.Fourier_matrix,input)
        return out









class Zernike_embedding(nn.Module):
    def __init__(self, n_max = 30 , device = 'cuda:2', numerical_expand = 4 ):
        super().__init__()
        print('empty class')

    def embed(self,input):

        out = 1
        return out

    def decode(self,input):
        out = 1
        return out