import math

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Zernike_layer import Lintrans3, Multilin, Zernike_layer


class ZernikeEncoder(pl.LightningModule):
    def __init__(self, n_in, n_output, num_channels,device ):
        super().__init__()
        test_dimens = False
        self.norm00 = nn.BatchNorm2d(10)
        self.norm00.bias.requires_grad = False

        self.norm1 = nn.BatchNorm2d(10)
        self.norm1.bias.requires_grad = False
        self.norm2 = nn.BatchNorm2d(10)
        self.norm2.bias.requires_grad = False

        self.norm3 = nn.BatchNorm2d(10)
        self.norm3.bias.requires_grad = False

        self.norm4 = nn.BatchNorm2d(10)
        self.norm4.bias.requires_grad = False

        self.norm5 = nn.BatchNorm2d(10)
        self.norm5.bias.requires_grad = False

        self.Product0 = Zernike_layer( n_max = 32, n_out=32,multichanneled = 'independant',in_channels = 3 ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)


        #self.Product01 = Zernike_layer( n_max = 32, n_out=32,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

        #self.Product02 = Zernike_layer( n_max = 32, n_out=32,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

        self.Product1 = Zernike_layer( n_max = 32, n_out=32,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)
        #self.Input1 =  torch.nn.parameter.Parameter(Init_zero(3,n_in))

        self.Product2 = Zernike_layer( n_max = 32, n_out=16,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)
        #self.Input2 =  torch.nn.parameter.Parameter(Init_zero(num_channels,n_in))

        size = self.Product2.calc_size(n_output)
        self.Lin_1 = Multilin(num_channels,num_channels,size,Non_lin=True)
        self.Lin_2 = Multilin(num_channels,1,size,Non_lin=True)

        self.Product3 = Zernike_layer( n_max = 16, n_out=8, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)
        #self.Input3 =  torch.nn.parameter.Parameter(Init_zero(num_channels,n_in))
        self.Product4 = Zernike_layer( n_max = 8, n_out=4, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)


        self.Product5 = Zernike_layer( n_max = 4, n_out=2, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)

        self.Product50 = Zernike_layer( n_max = 2, n_out=1, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)


    def forward(self, x) -> torch.tensor:
        eps = 0.0000000000000000000001
        #print(torch.cuda.mem_get_info())
        #print((torch.sum(torch.square(x),dim=(-1,-2),keepdim=True)+eps))
        #x2 = x/(torch.sum(torch.square(x),dim=(-1,-2),keepdim=True)+eps)
        #x = x/300
        #x2 = x/(torch.sum(torch.square(x),dim=(-1,-2),keepdim=True)+eps)
        x = self.Product0(x,x)
        #x = self.norm00(x)

        a = 1/(torch.sum(torch.abs(x),dim=(-1,-2),keepdim=True)+eps)
        x = x*a
        #x2 = self.x2Lin_2(self.x2Lin_1(x2))
        #print('prod0done')
        #print(torch.cuda.mem_get_info())
        #print('encode')
        #print(torch.sqrt(torch.sum(torch.square(x),dim =(-1,-2))[0,0:2])
        # a = 1/(torch.sum(torch.sqrt(torch.sum(torch.square(x),dim =(-1),keepdim=False)),dim=-1)+eps)
        # print('encode')
        # print(a)
        # print(a.size())
        #x = torch.einsum('ijkl,ij->ijkl', x,a)#*10000
        '''
        x = self.Product01(x,x)
        a = 1/(torch.sqrt(torch.sum(torch.square(x),dim =(-1,-2))+eps)
        x = torch.einsum('ijkl,ij->ijkl', x,a)

        x = self.Product02(x,x)
        a = 1/(torch.sqrt(torch.sum(torch.square(x),dim =(-1,-2))+eps)
        x = torch.einsum('ijkl,ij->ijkl', x,a)
        '''
        #print(torch.cuda.mem_get_info())
        #print((torch.sum(torch.square(x),dim=(-1,-2),keepdim=True)+eps))
        #x2 = x/(torch.sum(torch.square(x),dim=(-1,-2),keepdim=True)+eps)
        x = self.Product1(x,x)
        a = 1/(torch.sum(torch.abs(x),dim=(-1,-2),keepdim=True)+eps)
        x = x*a
        #print(torch.cuda.mem_get_info())
        #print(torch.sqrt(torch.sum(torch.square(x),dim =(-1,-2))[0,0:2])
        # a = 1/(torch.sum(torch.sqrt(torch.sum(torch.square(x),dim =(-1),keepdim=False)),dim=-1)+eps)
        # print(a)
        # x = torch.einsum('ijkl,ij->ijkl', x,a)#*100

        #print((torch.sum(torch.square(x),dim=(-1,-2),keepdim=True)+eps))

        #x = self.norm1(x)
        x = self.Product2(x,x)
        a = 1/(torch.sum(torch.abs(x),dim=(-1,-2),keepdim=True)+eps)
        x = x*a
        #print(torch.cuda.mem_get_info())
        #print(torch.sqrt(torch.sum(torch.square(x),dim =(-1,-2))[0,0:2])
        # a = 1/(torch.sum(torch.sqrt(torch.sum(torch.square(x),dim =(-1),keepdim=False)),dim=-1)+eps)
        # print(a)
        # x = torch.einsum('ijkl,ij->ijkl', x,a)#*100

        #print((torch.sum(torch.square(x),dim=(-1,-2),keepdim=True)+eps))
        #x = self.norm2(x)
        x = self.Product3(x,x)
        a = 1/(torch.sum(torch.abs(x),dim=(-1,-2),keepdim=True)+eps)
        x = x*a
        #print(torch.cuda.mem_get_info())
        #print(torch.sqrt(torch.sum(torch.square(x),dim =(-1,-2))[0,0:2])
        # a = 1/(torch.sum(torch.sqrt(torch.sum(torch.square(x),dim =(-1),keepdim=False)),dim=-1)+eps)
        # print(a)
        # x = torch.einsum('ijkl,ij->ijkl', x,a)#*100

        #print((torch.sum(torch.square(x),dim=(-1,-2),keepdim=True)+eps))
        #x = self.norm3(x)
        x = self.Product4(x,x)
        a = 1/(torch.sum(torch.abs(x),dim=(-1,-2),keepdim=True)+eps)
        x = x*a
        #print(torch.cuda.mem_get_info())
        #print(torch.sqrt(torch.sum(torch.square(x),dim =(-1,-2))[0,0:2])
        # a = 1/(torch.sum(torch.sqrt(torch.sum(torch.square(x),dim =(-1),keepdim=False)),dim=-1)+eps)
        # print(a)
        # x = torch.einsum('ijkl,ij->ijkl', x,a)#*100


        #print((torch.sum(torch.square(x),dim=(-1,-2),keepdim=True)+eps))

        #x = self.norm4(x)
        x = self.Product5(x,x)
        a = 1/(torch.sum(torch.abs(x),dim=(-1,-2),keepdim=True)+eps)
        x = x*a
        # a = 1/(torch.sum(torch.sqrt(torch.sum(torch.square(x),dim =(-1),keepdim=False)),dim=-1)+eps)
        # print(a)
        # x = torch.einsum('ijkl,ij->ijkl', x,a)#*100


        #print((torch.sum(torch.square(x),dim=(-1,-2),keepdim=True)+eps).size())

        #x = self.norm5(x)
        x = self.Product50(x,x)
        # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2),keepdim=True)+eps)
        # x = x*a
        #a = 1/(torch.sum(torch.sqrt(torch.sum(torch.square(x),dim =(-1),keepdim=False)),dim=-1)+eps)
        # #print(a)
        # x = torch.einsum('ijkl,ij->ijkl', x,a)#*100
        # print(torch.isnan(x).any())
        # print(torch.min(x))
        # print(torch.max(x))

        #print('encode_done')
        #print(self.norm1.bias)

        #print((torch.sum(torch.square(x),dim=(-1,-2),keepdim=True)+eps))
        x = self.Lin_2(self.Lin_1(x))
        # print(torch.isnan(x).any())
        # print(torch.min(x))
        # print(torch.max(x))

        #print((torch.sum(torch.square(x),dim=(-1,-2),keepdim=True)+eps))
        #print('done')
        return x




def Init_zero(in_size,n_max):
    n_max_calc = n_max+1
    zeros = torch.nn.init.normal_(torch.empty((in_size,int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4)),2)))/10
    zeros[0,0,0] = 1
    return zeros
