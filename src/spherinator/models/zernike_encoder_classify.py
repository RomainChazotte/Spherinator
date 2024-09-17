import math

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Zernike_layer import Lintrans3, Multilin, Zernike_layer


class ZernikeEncoderClassify(pl.LightningModule):
    def __init__(self, n_in, n_output, num_channels,device ):
        super().__init__()
        test_dimens = False


        self.norm = torch.nn.LayerNorm(10)
        self.Product0 = Zernike_layer( n_max = 32, n_out=32,multichanneled = 'independant',in_channels = 3 ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

        self.Product01 = Zernike_layer( n_max = 32, n_out=32,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

        self.Product02 = Zernike_layer( n_max = 32, n_out=32,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

        self.Product1 = Zernike_layer( n_max = 32, n_out=32,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)
        #self.Input1 =  torch.nn.parameter.Parameter(Init_zero(3,n_in))

        self.Product2 = Zernike_layer( n_max = 32, n_out=16,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)
        #self.Input2 =  torch.nn.parameter.Parameter(Init_zero(num_channels,n_in))


        self.Product3 = Zernike_layer( n_max = 16, n_out=8, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)
        #self.Input3 =  torch.nn.parameter.Parameter(Init_zero(num_channels,n_in))
        self.Product4 = Zernike_layer( n_max = 8, n_out=4, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)


        self.Product5 = Zernike_layer( n_max = 4, n_out=2, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)

        self.Product50 = Zernike_layer( n_max = 2, n_out=1, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)

        self.Product_classifier = Zernike_layer( n_max = 1, n_out=0, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)

        size = self.Product5.calc_size(n_output)
        self.Lin_2_classifier = Multilin(num_channels,10,size,Non_lin=True)
        self.Lin_1 = Multilin(num_channels,num_channels,size,Non_lin=True)
        self.Lin_2 = Multilin(num_channels,1,size,Non_lin=True)

        #size = self.Product2.calc_size(n_output)
    def forward(self, x) -> torch.tensor:
        eps = 0.0000000000000000000001

        x = self.Product0(x,x)
        ##print('encode')
        ##print(torch.sum(torch.abs(x),dim =(-1,-2))[0,0:2])
        #a = 1/(torch.sum(torch.abs(x),dim =(-1,-2))+eps)
        #x = torch.einsum('ijkl,ij->ijkl', x,a)

        a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        #print(a[0])
        x = x*a
        x = self.Product01(x,x)
        a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        #print(a[0])
        x = x*a
        x = self.Product02(x,x)
        a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        #print(a[0])
        x = x*a
        x = self.Product1(x,x)
        a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        #print(a[0])
        x = x*a

        x = self.Product2(x,x)
        a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        #print(a[0])
        x = x*a

        x = self.Product3(x,x)
        a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        #print(a[0])
        x = x*a

        x = self.Product4(x,x)
        a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        #print(a[0])
        x = x*a


        x = self.Product5(x,x)
        a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        #print(a[0])
        x = x*a


        x = self.Product50(x,x)
        a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        #print(a[0])
        x = x*a

        ##print('encode_done')

        #x = self.Lin_2(self.Lin_1(x))



        x = self.Product_classifier(x,x)
        a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        #print(a[0])
        x = x*a
        x = self.Lin_2_classifier(self.Lin_1(x))
        a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        x = x*a
        #print(a[0])
        #print(x[0])
        x = x[:,:,0,0]
        #x = self.norm(x)
        #print(x[0])
        return x




def Init_zero(in_size,n_max):
    n_max_calc = n_max+1
    zeros = torch.nn.init.normal_(torch.empty((in_size,int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4)),2)))/10
    zeros[0,0,0] = 1
    return zeros
