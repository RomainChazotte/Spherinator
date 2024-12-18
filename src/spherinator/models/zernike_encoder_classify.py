import math

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Zernike_layer import Lintrans3, Multilin, Zernike_layer


class  Non_linearity(nn.Module):
    '''
    A Non-linearity defined for Zernike objects. The Non-linearity needs to act on the absolute value of sqrt((Z_m^n)**2 + (Z_-m^n)**2), in order to not break equivariance.
    This function replicates the functionallity of ReLU, with a learnable cutoff. It takes the abs value, applies a 1D linear layer to it, and applies a differentiable step-function to it.
    This is multiplied to the input. The step function is used so out is proportional to in, not to in**2, which leads to NaNs appearing.
    '''
    def __init__(self, normalize= False):
        super().__init__()
        self.lin = nn.Linear(1,1)
        # self.lin.weight.data = torch.ones(1,1,dtype=torch.float)
        # self.lin.bias.data =torch.zeros(1,dtype=torch.float)
        self.normalize = normalize
    def forward(self,x):
        square = torch.sum(x**2, dim=-1,keepdim=True)
        a = self.lin(square)
        a = (torch.tanh(a)+1)/2

        if self.normalize:
            norm = torch.sum(torch.sqrt(square),dim=-2,keepdim=True)
            return a*x/norm
        else:
            return a * x

class Multilin_dim2(nn.Module):
    '''
    A class that creates a linear layer with independant weights along it's channels.
    In this framework, this serves to create a linear layer convolving the channels with each other, but doing this in a way that is independent for different m,n.
    Yet, it acts identical on n,m and n,-m; therefore keeping rotation equivariance
    '''
    def __init__(self, inp,out,size,Non_lin=False,normalize=False):
        super().__init__()
        #self.device = 'cuda:2'
        #self.weight_lin = torch.nn.parameter.Parameter(torch.nn.init.normal_(torch.empty(size,inp,out, device = self.device)))#/np.sqrt(inp))
        self.weight_lin = torch.nn.parameter.Parameter(torch.nn.init.normal_(torch.empty(size,out,inp)))#/np.sqrt(inp))
        self.Non_lin_bool = Non_lin
        if Non_lin:
            self.Non_lin = Non_linearity(normalize=normalize)
    def forward(self,x):
        x = torch.einsum('jki,...ji->...jk',self.weight_lin,x)
        if self.Non_lin_bool:
            x = self.Non_lin(x)
        return x
class ZernikeEncoderClassify(pl.LightningModule):
    def __init__(self, n_in, n_output, num_channels,device, local = False ):
        super().__init__()
        test_dimens = False
        num_channels = 10
        self.eps = 1e-10

        n = n_in
        # self.norm2 = torch.nn.LayerNorm(20)
        # self.norm3 = torch.nn.LayerNorm(10)
        self.local=local

        if local:
            #num_channels = 20
            self.Product0 = Zernike_layer( n_max = n, n_out=n,multichanneled = 'independant',in_channels = 1 ,intermediate_channels=num_channels, out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)

            #self.Product01 = Zernike_layer( n_max = n, n_out=n,multichanneled = 'independant',in_channels = 5 ,intermediate_channels=5, out_channels =10 ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

            self.Product02 = Zernike_layer( n_max = n, n_out=n,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=num_channels, out_channels =num_channels,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)
            #self.Product03 = Zernike_layer( n_max = n, n_out=n,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=num_channels, out_channels =num_channels,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

            # self.Product04 = Zernike_layer( n_max = n, n_out=n,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=num_channels, out_channels =num_channels,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)
            # self.Product05 = Zernike_layer( n_max = n, n_out=n,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=num_channels, out_channels =num_channels,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

        else:
            self.Product0 = Zernike_layer( n_max = n, n_out=n,multichanneled = 'independant',in_channels = 3* num_channels +1 ,intermediate_channels=num_channels, out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

            #self.Product01 = Zernike_layer( n_max = 8, n_out=8,multichanneled = 'independant',in_channels = 5 ,intermediate_channels=5, out_channels =10 ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

            self.Product02 = Zernike_layer( n_max = n, n_out=n,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=num_channels, out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

            #self.Product03 = Zernike_layer( n_max = n, n_out=n,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=num_channels, out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

            # self.Product04 = Zernike_layer( n_max = n, n_out=n,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=num_channels, out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

            # self.Product05 = Zernike_layer( n_max = n, n_out=n,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=num_channels, out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)
            #self.Product03 = Zernike_layer( n_max = 8, n_out=8,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=num_channels, out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

        if not local:

            size = self.Product0.calc_size(n)
            self.size = size
            self.Rnorm2 = nn.RMSNorm([size,2],elementwise_affine=False)
            size = self.Product0.calc_size(n)
            self.size = size
            self.Rnorm02 = nn.RMSNorm([size,2],elementwise_affine=False)
            self.Rnorm03 = nn.RMSNorm([size,2],elementwise_affine=False)
            self.Rnorm04 = nn.RMSNorm([size,2],elementwise_affine=False)
            self.Rnorm05 = nn.RMSNorm([size,2],elementwise_affine=False)

            self.linout = nn.Linear(size,10)
            self.Rnorm3 = nn.RMSNorm([10],elementwise_affine=False)
            self.linout2 = nn.Linear(10,1)

            self.linout3 = nn.Linear(num_channels,20, bias=False)
            self.linout4 = nn.Linear(20,20, bias=False)
            self.linout5 = nn.Linear(20,20, bias=False)
            self.linout6 = nn.Linear(20,20, bias=False)
            self.linout7 = nn.Linear(20,10, bias=False)


        if local:
            size = self.Product0.calc_size(n_in)
            self.size = size
            self.Rnorm2 = nn.RMSNorm([size,2],elementwise_affine=False)
            self.Rnorm02 = nn.RMSNorm([size,2],elementwise_affine=False)
            self.Rnorm03 = nn.RMSNorm([size,2],elementwise_affine=False)
            self.Rnorm04 = nn.RMSNorm([size,2],elementwise_affine=False)
            self.Rnorm05 = nn.RMSNorm([size,2],elementwise_affine=False)


            self.local_out2 = Multilin(num_channels,num_channels,size,Non_lin=False)


            self.linout_loc = nn.Linear(size,size)
            self.linout2_loc = nn.Linear(size,1)
            self.Rnorm3_loc = nn.RMSNorm([size],elementwise_affine=False)

            self.collapse_angle = nn.Linear(2*num_channels,num_channels)
            self.collapse_angle_2 = nn.Linear(num_channels,num_channels)
            self.collapse_angle_3 = nn.Linear(num_channels,num_channels)
            self.collapse_angle_4 = nn.Linear(num_channels,num_channels)
            self.collapse_angle_5 = nn.Linear(num_channels,num_channels)
    def forward(self, x) -> torch.tensor:
        eps = 0.0000000000000000000001
        #x = x.unsqueeze(-3)

        if not self.local:
            x = self.Product0(x,x)#+x
            x = self.Rnorm2(x)
            x = self.Product02(x,x)+x
            x = self.Rnorm02(x)
            #x = self.Product03(x,x)+x
            #x = self.Rnorm03(x)
            # x = self.Product04(x,x)+x
            # x = self.Rnorm04(x)
            # x = self.Product05(x,x)+x
            # x = self.Rnorm05(x)
            x = torch.sum(torch.square(x), dim=-1)

            x = F.relu(self.linout(x))
            x = self.Rnorm3(x)
            x = (self.linout2(x)).squeeze()
            x = F.relu(self.linout3(x))
            # x = F.relu(self.linout4(x))
            # x = F.relu(self.linout5(x))
            # x = F.relu(self.linout6(x))
            x = (self.linout7(x))

        # if self.local:
        #     x = self.Product0(x,x)+x
        #     x = self.Rnorm2(x)
        #     x = self.Product02(x,x)+x
        #     x = self.Rnorm02(x)


        #     x = self.local_out2(x)

        #     #print(angle.size())
        #     x = torch.flatten(x,start_dim=-2)
        #     x = F.relu(self.linout_loc(x))
        #     x = self.Rnorm3_loc(x)
        #     x = (self.linout2_loc(x)).squeeze()#.unsqueeze(-3)
        #     #print(x.size())


        #     x = F.relu(self.collapse_angle(x))
        #     x = self.collapse_angle_2(x)




        if self.local:
            x = self.Product0(x,x)#+x
            x = self.Rnorm2(x)
            x = self.Product02(x,x)+x
            x = self.Rnorm02(x)
            #x = self.Product03(x,x)+x
            #x = self.Rnorm03(x)

            # x = self.Product04(x,x)+x
            # x = self.Rnorm04(x)
            # x = self.Product05(x,x)+x
            # x = self.Rnorm05(x)


            x = self.local_out2(x)

            angle = torch.atan2(x[...,3,0],x[...,3,1]+self.eps)#.unsqueeze(-1)
            #print(angle.size())

            x = torch.sum(torch.square(x), dim=-1)
            x = F.relu(self.linout_loc(x))
            x = self.Rnorm3_loc(x)
            x = (self.linout2_loc(x)).squeeze()#.unsqueeze(-3)
            #print(x.size())





            x = torch.cat((x,angle),dim=-1)
            x = F.relu(self.collapse_angle(x))
            # x = F.relu(self.collapse_angle_2(x))
            # x = F.relu(self.collapse_angle_3(x))
            # x = F.relu(self.collapse_angle_4(x))
            x = self.collapse_angle_5(x)






            #print(x.size())
            #x = x.squeeze()
        return x
