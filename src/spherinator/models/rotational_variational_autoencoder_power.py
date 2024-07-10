import math

import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
from power_spherical import HypersphericalUniform, PowerSpherical
from torch.optim import Adam
import torchvision.transforms.v2.functional as functional

from .convolutional_decoder import ConvolutionalDecoder
from .convolutional_encoder import ConvolutionalEncoder
from .spherinator_module import SpherinatorModule


from scipy.constants import physical_constants
import scipy.special as sp
import seaborn as sns
import numpy as np
import argparse

import numpy
import numpy as np
from numpy import convolve
import scipy
import math
import torch
import torch.nn as nn

import torch.nn.functional as F

import random


# Take out this import
#import matplotlib.pyplot as plt


class RotationalVariationalAutoencoderPower(SpherinatorModule):
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
        #self.encoder = ConvolutionalEncoder(),
        #self.decoder = ConvolutionalDecoder(),
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
        self.Zernike_size = lengh = int(((33+1)*33/2)/2+math.ceil(33/4))
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
        #return z
    def reparameterize(self, z_location, z_scale):
        q_z = PowerSpherical(z_location, z_scale)
        p_z = HypersphericalUniform(self.z_dim, device=z_location.device)
        return q_z, p_z

    def forward(self, x):
        #print(x.size())
        #q_z, p_z = self.reparameterize(z_location, z_scale.squeeze())
        #z = q_z.rsample()
        #print(z[0])
        '''
        pre_model = torch.sum(x,-3)
        x = self.Embedding_Function(pre_model).unsqueeze(1)
        '''
        x = self.Embedding_Function(x)

        #print('start')
        #print(x)

        z = self.encode(x)
        #print('z')
        #print(z)
        #print(z.size())
        recon = self.decode(z)

        #return recon, x

        return recon,x


    def training_step(self, batch, batch_idx):
        with torch.no_grad():


            crop = functional.center_crop(batch, [self.crop_size, self.crop_size])
            scaled = functional.resize(
                crop, [self.input_size, self.input_size], antialias=True
            )
        '''
        pre_model = torch.sum(scaled,-3)
        import matplotlib.pyplot as plt
            #print('hi')
        reconstruction = self.Embedding_Function(pre_model)
        reconstruction = self.Decoding_Function(reconstruction)
        for i in range(30):
            plt.figure()
            plt.imshow(pre_model[i].cpu().float().numpy())
            plt.savefig('prepic_new{}.png'.format(i))
            plt.close()
            plt.figure()
            plt.imshow(reconstruction[i].cpu().float().numpy())
            plt.savefig('postpic_new{}.png'.format(i))
            plt.close()


        import matplotlib.pyplot as plt
        #print('hi')

        #print(pre_model.size())
        #print(x.size())
        #print('hi')


        #reconstruction = self.Decoding_Function(out)

        for i in range(1,50):
            pre_model = torch.sum(scaled,-3)
            plt.figure()
            plt.imshow(pre_model[2].cpu().float().numpy())
            plt.savefig('prepic_gif{}.png'.format(0))
            plt.close()

            Embed = Zernike_embedding(i)
            Reconstruct =  Zernike_decode(i)
            x = Embed(pre_model)
            reconstruction = Reconstruct(x)

            plt.figure()
            plt.imshow(reconstruction[2].cpu().float().numpy())
            plt.savefig('postpic_gif{}.png'.format(i))
            plt.close()
        donkey
        '''
        #a = random.randint(3, 19)
        #print(scaled.size())
        out,x = self.forward(scaled)
        #print('finishing')
        #print(out.size())
        #print(x.size())
        '''
        print(out.size())
        print(x.size())
        for i in range(10):
            print(torch.sum(out[0]-out[i]))
        '''

        #a = 0
        #if self.optimizers().param_groups[0]["lr"] < 0.001:

        #recon, pre_model = self.forward(scaled)
        #print('start')
        #best_scaled_image, _, _, _ = self.find_best_rotation(batch)
        #print('for testing')
        #print(out)
        loss = self.criterion(x, out)
        ##################################################################################################
        # Try this loss
        #loss = self.reconstruction_loss(x, out)
        ###################################################################################################################################



        #loss_recon = torch.sum(out)- torch.sum(x)
        #pre_model = torch.sum(scaled,-3)
        #pre_model = torch.einsum('...jk,ljk->...l',pre_model,self.waves)
        #reconstruction = self.reconstruct.forward(pre_model.detach().cpu()).float()
        #pre_reconstruction = self.reconstruct.forward(scaled.detach().cpu()).float()
        #pre_reconstruction =  torch.sum(scaled,-3).detach().cpu()



        #if self.optimizers().param_groups[0]["lr"] < 0.001:
        #loss = loss_recon.mean()
        #print('Hi')
        '''
        if loss < 1.0:
            #print(x[0,0,0:5],out[0,0,0:5])
            import matplotlib.pyplot as plt
            a = random.randint(3, 19)
            #print('hi')

            plt.figure()
            plt.imshow((torch.sum(scaled,dim=-3))[0].cpu().float().numpy())
            plt.savefig('prepica{}.png'.format(a))
            plt.close()

            reconstruction = self.Decoding_Function(x).squeeze().detach()
            plt.figure()
            plt.imshow(reconstruction[0].cpu().float().numpy())
            plt.savefig('recpica{}.png'.format(a))
            plt.close()
            #print(reconstruction[...,50:55,50:55])
            #print(torch.sum(scaled,dim=-3)[...,50:55,50:55])

            reconstruction = self.Decoding_Function(out).squeeze().detach()
            plt.figure()
            plt.imshow(reconstruction[0].cpu().float().numpy())
            plt.savefig('postpica{}.png'.format(a))
            plt.close()

            plt.figure()
            plt.imshow((torch.transpose(scaled,-1,-3))[0].cpu().float().numpy())
            plt.savefig('prepica{}.png'.format(a))
            plt.close()

            reconstruction = self.Decoding_Function(x).squeeze().detach()
            plt.figure()
            plt.imshow(torch.transpose(reconstruction,-1,-3)[0].cpu().float().numpy())
            plt.savefig('recpica{}.png'.format(a))
            plt.close()

            reconstruction = self.Decoding_Function(out).squeeze().detach()
            plt.figure()
            plt.imshow(torch.transpose(reconstruction,-1,-3)[0].cpu().float().numpy())
            plt.savefig('postpica{}.png'.format(a))
            plt.close()
        '''

        #loss = loss_recon.mean()
        '''
        sum_out = torch.sum(out)
        sum_rec = torch.sum(x)
        self.log("Out_sum", sum_out, prog_bar=True)
        self.log("Rec_sum", sum_rec, prog_bar=True)
        '''
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

    def reconstruction_loss(self, images, reconstructions):

        return torch.sqrt(
            nn.MSELoss(reduction="none")(
                reconstructions.reshape(-1, self.Zernike_size),
                images.reshape(-1, self.Zernike_size),
            ).mean(dim=1)
        )


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