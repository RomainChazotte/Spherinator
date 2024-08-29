import math

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Zernike_layer import Lintrans3, Multilin, Zernike_layer


class ZernikeEncoder(pl.LightningModule):
    def __init__(self, n_in, n_output, num_channels):
        super().__init__()
        test_dimens = False

        self.Product0 = Zernike_layer( n_max = 32, n_out=32,multichanneled = 'independant',in_channels = 3 ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens)#, normalize=True)

        self.Product1 = Zernike_layer( n_max = 32, n_out=32,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens)#, normalize=True)
        #self.Input1 =  torch.nn.parameter.Parameter(Init_zero(3,n_in))

        self.Product2 = Zernike_layer( n_max = 32, n_out=16,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens)
        #self.Input2 =  torch.nn.parameter.Parameter(Init_zero(num_channels,n_in))

        size = self.Product2.calc_size(n_output)
        self.Lin_1 = Multilin(num_channels,num_channels,size,Non_lin=True)
        self.Lin_2 = Multilin(num_channels,1,size,Non_lin=True)

        self.Product3 = Zernike_layer( n_max = 16, n_out=8, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens)
        #self.Input3 =  torch.nn.parameter.Parameter(Init_zero(num_channels,n_in))
        self.Product4 = Zernike_layer( n_max = 8, n_out=4, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens)


        self.Product5 = Zernike_layer( n_max = 4, n_out=2, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens)

    def forward(self, x) -> torch.tensor:
        eps = 0.0000000000000000000001
        x = self.Product0(x,x)
        #print('encode')
        #print(torch.sum(torch.abs(x),dim =(-1,-2))[0,0:2])
        a = 1/(torch.sum(torch.abs(x),dim =(-1,-2))+eps)
        x = torch.einsum('ijkl,ij->ijkl', x,a)

        x = self.Product1(x,x)
        #print(torch.sum(torch.abs(x),dim =(-1,-2))[0,0:2])
        a = 1/(torch.sum(torch.abs(x),dim =(-1,-2))+eps)
        x = torch.einsum('ijkl,ij->ijkl', x,a)

        x = self.Product2(x,x)
        #print(torch.sum(torch.abs(x),dim =(-1,-2))[0,0:2])
        a = 1/(torch.sum(torch.abs(x),dim =(-1,-2))+eps)
        x = torch.einsum('ijkl,ij->ijkl', x,a)

        x = self.Product3(x,x)
        #print(torch.sum(torch.abs(x),dim =(-1,-2))[0,0:2])
        a = 1/(torch.sum(torch.abs(x),dim =(-1,-2))+eps)
        x = torch.einsum('ijkl,ij->ijkl', x,a)

        x = self.Product4(x,x)
        #print(torch.sum(torch.abs(x),dim =(-1,-2))[0,0:2])
        a = 1/(torch.sum(torch.abs(x),dim =(-1,-2))+eps)
        x = torch.einsum('ijkl,ij->ijkl', x,a)


        x = self.Product5(x,x)
        #print(torch.sum(torch.abs(x),dim =(-1,-2))[0,0:2])
        a = 1/(torch.sum(torch.abs(x),dim =(-1,-2))+eps)
        x = torch.einsum('ijkl,ij->ijkl', x,a)

        #print('encode_done')

        x = self.Lin_2(self.Lin_1(x))
        return x




def Init_zero(in_size,n_max):
    n_max_calc = n_max+1
    zeros = torch.nn.init.normal_(torch.empty((in_size,int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4)),2)))/10
    zeros[0,0,0] = 1
    return zeros
