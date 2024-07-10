import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from .Zernike_layer import Zernike_layer,Multilin,Lintrans3

class ConvolutionalDecoder(pl.LightningModule):
    def __init__(self, h_dim: int = 256):
        super().__init__()
        test_dimens = False

        n_max = 4
        self.Product1 = Zernike_layer( n_max = n_max,n_max_2=32, n_out=32, multichanneled = 'independant',in_channels = 20 ,intermediate_channels=20, out_channels =20 ,fast_test_dimensionality=test_dimens)
        self.Input1 =  torch.nn.parameter.Parameter(Init_zero(20,32))

        n_max = 32
        self.Product2 = Zernike_layer( n_max = n_max, n_out=32, multichanneled = 'independant',in_channels = 20 ,intermediate_channels=20, out_channels =20 ,fast_test_dimensionality=test_dimens)
        self.Input2 =  torch.nn.parameter.Parameter(Init_zero(20,n_max))

        self.Lin_1 = Multilin(1,20,16,Non_lin=True)
        self.Lin_2 = Multilin(20,20,16,Non_lin=True)

        n_max = 32
        self.Product3 = Zernike_layer( n_max = n_max, n_out=32, multichanneled = 'independant',in_channels = 20 ,intermediate_channels=20, out_channels =3 ,fast_test_dimensionality=test_dimens)
        self.Input3 =  torch.nn.parameter.Parameter(Init_zero(20,n_max))
        '''
        n_max = 26
        self.Product4 = Zernike_layer( n_max = n_max, n_out=26, multichanneled = 'independant',in_channels = 10 ,intermediate_channels=10, out_channels =10 ,fast_test_dimensionality=test_dimens)
        self.Input4 =  torch.nn.parameter.Parameter(Init_zero(n_max))

        n_max = 26
        self.Product5 = Zernike_layer( n_max = n_max, n_out=32,multichanneled = 'independant',in_channels = 10 ,intermediate_channels=10, out_channels =1 ,fast_test_dimensionality=test_dimens)
        self.Input5 =  torch.nn.parameter.Parameter(Init_zero(n_max))

        n_max = 32
        self.Product6 = Zernike_layer( n_max = n_max, n_out=32, multichanneled = 'independant',in_channels = 10 ,intermediate_channels=5, out_channels =1 ,fast_test_dimensionality=test_dimens)
        self.Input6 =  torch.nn.parameter.Parameter(Init_zero(n_max))
        n_max = 32
        self.Product1 = Zernike_layer( n_max = n_max, n_out=32, multichanneled = 'independant',in_channels = 10 ,intermediate_channels=5, out_channels =1 ,fast_test_dimensionality=test_dimens)
        self.Input1 =  torch.nn.parameter.Parameter(torch.rand((1,int(((n_max+2)*(n_max+1)/2)/2+math.ceil((n_max+1)/4)),2)))
        '''

    def forward(self, x) -> torch.tensor:
        x = self.Lin_2(self.Lin_1(x))
        x = self.Product1(x,self.Input1)

        x = self.Product2(x,self.Input2)
        x = self.Product3(x,self.Input3)

        #x = self.Product4(x,self.Input4)
        #x = self.Product5(x,self.Input5)
        #x = self.Product6(x,self.Input6)

        return x






def Init_zero(in_size,n_max):
    n_max_calc = n_max+1
    #zeros = (torch.zeros((1,int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4)),2)))
    zeros = torch.nn.init.normal_(torch.empty((in_size,int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4)),2)))/10
    zeros[0,0,0] = 1
    return zeros
