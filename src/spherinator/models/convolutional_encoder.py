import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from .Zernike_layer import Zernike_layer,Multilin,Lintrans3
class ConvolutionalEncoder(pl.LightningModule):
    def __init__(self, h_dim: int = 256):
        super().__init__()
        test_dimens = False

        self.Product1 = Zernike_layer( n_max = 32, n_out=32,multichanneled = 'independant',in_channels = 3 ,intermediate_channels=20, out_channels =20 ,fast_test_dimensionality=test_dimens)#, normalize=True)
        self.Input1 =  torch.nn.parameter.Parameter(Init_zero(3,32))

        self.Product2 = Zernike_layer( n_max = 32, n_out=32,multichanneled = 'independant',in_channels = 20 ,intermediate_channels=20, out_channels =20 ,fast_test_dimensionality=test_dimens)
        self.Input2 =  torch.nn.parameter.Parameter(Init_zero(20,32))

        self.Lin_1 = Multilin(20,20,16,Non_lin=True)
        self.Lin_2 = Multilin(20,1,16,Non_lin=True)

        self.Product3 = Zernike_layer( n_max = 32, n_out=4, multichanneled = 'independant',in_channels = 20 ,intermediate_channels=20, out_channels =20 ,fast_test_dimensionality=test_dimens)
        self.Input3 =  torch.nn.parameter.Parameter(Init_zero(20,32))
        '''
        self.Product4 = Zernike_layer( n_max = 26, n_out=26, multichanneled = 'independant',in_channels = 10 ,intermediate_channels=10, out_channels =10 ,fast_test_dimensionality=test_dimens)
        self.Input4 =  torch.nn.parameter.Parameter(Init_zero(26))
        self.Product5 = Zernike_layer( n_max = 26, n_out=26, multichanneled = 'independant',in_channels = 10 ,intermediate_channels=10, out_channels =1 ,fast_test_dimensionality=test_dimens)
        self.Input5 =  torch.nn.parameter.Parameter(Init_zero(26))

        self.Product1 = Zernike_layer( n_max = 32, n_out=32, last_layer = False,multichanneled = 'independant',in_channels = 1 ,intermediate_channels=1, out_channels =1 )
        zeros = (torch.zeros((1,int(((33+1)*33/2)/2+math.ceil(33/4)),2)))
        zeros[0,0,0] = 1
        self.Input1 =  torch.nn.parameter.Parameter(zeros)
        '''
    def forward(self, x) -> torch.tensor:
        #print('conv_encoder')
        #print(x.size())
        x = self.Product1(x,self.Input1)

        x = self.Product2(x,self.Input2)
        x = self.Product3(x,self.Input3)
        x = self.Lin_2(self.Lin_1(x))
        #x = self.Product4(x,self.Input4)
        #x = self.Product5(x,self.Input5)
        return x




def Init_zero(in_size,n_max):
    n_max_calc = n_max+1
    #zeros = (torch.zeros((1,int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4)),2)))
    zeros = torch.nn.init.normal_(torch.empty((in_size,int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4)),2)))/10
    zeros[0,0,0] = 1
    return zeros
