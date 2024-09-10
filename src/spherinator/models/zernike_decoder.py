import math

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Zernike_layer import Lintrans3, Multilin, Zernike_layer


class ZernikeDecoder(nn.Module):
    def __init__(self, n_in, n_output, num_channels,device):
        super().__init__()
        test_dimens = False

        n_max = n_output
        self.Product00 = Zernike_layer( n_max = 1,n_max_2=1, n_out=2, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)

        self.Product0 = Zernike_layer( n_max = 2,n_max_2=2, n_out=4, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)

        self.Product1 = Zernike_layer( n_max = 4,n_max_2=4, n_out=8, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)
        #self.Input1 =  torch.nn.parameter.Parameter(Init_zero(num_channels,n_in))

        n_max = n_in
        self.Product2 = Zernike_layer( n_max = 8, n_out=16, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)
        #self.Input2 =  torch.nn.parameter.Parameter(Init_zero(num_channels,n_max))

        size = self.Product2.calc_size(n_output)
        self.Lin_1 = Multilin(1,num_channels,size,Non_lin=True)
        self.Lin_2 = Multilin(num_channels,num_channels,size,Non_lin=True)

        n_max = n_in
        self.Product3 = Zernike_layer( n_max = 16, n_out=32, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)

        self.Product4 = Zernike_layer( n_max = 32, n_out=32, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)

        self.Product5 = Zernike_layer( n_max = 32, n_out=32, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =3 ,fast_test_dimensionality=test_dimens, device = device)
        '''
        self.Product6 = Zernike_layer( n_max = 32, n_out=32, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)

        self.Product7 = Zernike_layer( n_max = 32, n_out=32, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =3 ,fast_test_dimensionality=test_dimens, device = device)
        '''
        size = self.Product2.calc_size(n_in)
        self.output_Lin_1 = Multilin(num_channels,num_channels,size,Non_lin=True)
        self.output_Lin_2 = Multilin(num_channels,3,size,Non_lin=True)
        #self.Input3 =  torch.nn.parameter.Parameter(Init_zero(num_channels,n_max))
    def forward(self, x) -> torch.tensor:
        eps = 0.0000000000000000000001
        #print('decode')
        #print(torch.sum(torch.abs(x),dim =(-1,-2))[0,0:2])
        x = self.Lin_2(self.Lin_1(x))
        #print(torch.sum(torch.abs(x),dim =(-1,-2))[0,0:2])
        x = self.Product00(x,x)
        #print(torch.sum(torch.abs(x),dim =(-1,-2))[0,0:2])
        a = 1/(torch.sum(torch.abs(x),dim =(-1,-2))+eps)
        x = torch.einsum('ijkl,ij->ijkl', x,a)

        x = self.Product0(x,x)
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
        #x = self.output_Lin_2(self.output_Lin_1(x))
        '''

        a = 1/(torch.sum(torch.abs(x),dim =(-1,-2))+eps)
        x = torch.einsum('ijkl,ij->ijkl', x,a)
        x = self.Product6(x,x)


        a = 1/(torch.sum(torch.abs(x),dim =(-1,-2))+eps)
        x = torch.einsum('ijkl,ij->ijkl', x,a)
        x = self.Product7(x,x)
        '''
        #print(torch.sum(torch.abs(x),dim =(-1,-2))[0,0:2])
        #print('decode_done')
        #x = x/torch.sum(torch.abs(x),dim =(-1,-2))
        #print('done')
        return x

'''
class ZernikeDecoder(pl.LightningModule):
    def __init__(self, n_in, n_output, num_channels):
        super().__init__()
        test_dimens = False

        n_max = n_output
        self.Product1 = Zernike_layer( n_max = n_max,n_max_2=n_in, n_out=n_in, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=num_channels, out_channels =num_channels ,fast_test_dimensionality=test_dimens)
        self.Input1 =  torch.nn.parameter.Parameter(Init_zero(num_channels,n_in))

        n_max = n_in
        self.Product2 = Zernike_layer( n_max = n_max, n_out=n_in, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=num_channels, out_channels =num_channels ,fast_test_dimensionality=test_dimens)
        self.Input2 =  torch.nn.parameter.Parameter(Init_zero(num_channels,n_max))

        size = self.Product2.calc_size(n_output)
        self.Lin_1 = Multilin(1,num_channels,size,Non_lin=True)
        self.Lin_2 = Multilin(num_channels,num_channels,size,Non_lin=True)

        n_max = n_in
        self.Product3 = Zernike_layer( n_max = n_max, n_out=n_in, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=num_channels, out_channels =3 ,fast_test_dimensionality=test_dimens)
        self.Input3 =  torch.nn.parameter.Parameter(Init_zero(num_channels,n_max))
    def forward(self, x) -> torch.tensor:
        x = self.Lin_2(self.Lin_1(x))
        x = self.Product1(x,self.Input1)

        x = self.Product2(x,self.Input2)
        x = self.Product3(x,self.Input3)

        return x
'''





def Init_zero(in_size,n_max):
    n_max_calc = n_max+1
    zeros = torch.nn.init.normal_(torch.empty((in_size,int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4)),2)))/10
    zeros[0,0,0] = 1
    return zeros

'''
Encode = ZernikeDecoder(1,1,1).to('cuda:2')
print(Encode.Input1)
input_1 = torch.zeros(1,1,2,2,device='cuda:2')
input_1[0,0,0,0] = 2000
#input_1[0,0,1,0] = 2000
print(Encode(input_1))


input_2 = torch.zeros(1,1,2,2,device='cuda:2')
#input_2[0,0,0,0] = 2000
input_2[0,0,1,1] = 2000
print(Encode(input_2))

input_2 = torch.zeros(1,1,2,2,device='cuda:2')
#input_2[0,0,0,0] = 2000
input_2[0,0,1,0] = 2000
print(Encode(input_2))
'''