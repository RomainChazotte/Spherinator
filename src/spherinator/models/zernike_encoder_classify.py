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
    def __init__(self, n_in, n_output, num_channels,device ):
        super().__init__()
        test_dimens = False
        num_channels = 10
        n = 8
        self.norm = torch.nn.LayerNorm(10)
        self.norm1 = torch.nn.LayerNorm(10)
        # self.norm2 = torch.nn.LayerNorm(20)
        # self.norm3 = torch.nn.LayerNorm(10)
        self.Product0 = Zernike_layer( n_max = n, n_out=4,multichanneled = 'independant',in_channels = 1 ,intermediate_channels=5, out_channels =5 ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

        self.Product01 = Zernike_layer( n_max = 4, n_out=4,multichanneled = 'independant',in_channels = 5 ,intermediate_channels=5, out_channels =10 ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

        #self.Product02 = Zernike_layer( n_max = 8, n_out=4,multichanneled = 'independant',in_channels = 10 ,intermediate_channels=10, out_channels =10 ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

        # self.Product03 = Zernike_layer( n_max = 4, n_out=4,multichanneled = 'independant',in_channels = 10 ,intermediate_channels=5, out_channels =10 ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

        # self.Product04 = Zernike_layer( n_max = 4, n_out=4,multichanneled = 'independant',in_channels = 10 ,intermediate_channels=5, out_channels =10 ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

        # self.Product001 = Zernike_layer( n_max = n, n_out=n,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=num_channels, out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

        # self.Product002 = Zernike_layer( n_max = n, n_out=n,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=num_channels, out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

        # # self.Product0001 = Zernike_layer( n_max = n, n_out=n,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

        # # self.Product0002 = Zernike_layer( n_max = n, n_out=n,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)

        # num_channels2 = 30
        # self.Product1 = Zernike_layer( n_max = n, n_out=n,multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels2), out_channels =num_channels2 ,fast_test_dimensionality=test_dimens, device = device)#, normalize=True)
        # #self.Input1 =  torch.nn.parameter.Parameter(Init_zero(3,n_in))
        # num_channels3 = 30
        # self.Product2 = Zernike_layer( n_max = n, n_out=0,multichanneled = 'independant',in_channels = num_channels2 ,intermediate_channels=int(num_channels2), out_channels =num_channels3 ,fast_test_dimensionality=test_dimens, device = device)
        # #self.Input2 =  torch.nn.parameter.Parameter(Init_zero(num_channels,n_in))


        # self.Product3 = Zernike_layer( n_max = 16, n_out=8, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)
        # #self.Input3 =  torch.nn.parameter.Parameter(Init_zero(num_channels,n_in))
        # self.Product4 = Zernike_layer( n_max = 8, n_out=4, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)


        # self.Product5 = Zernike_layer( n_max = 4, n_out=2, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)

        # self.Product50 = Zernike_layer( n_max = 2, n_out=1, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)

        # self.Product_classifier = Zernike_layer( n_max = 1, n_out=0, multichanneled = 'independant',in_channels = num_channels ,intermediate_channels=int(num_channels/2), out_channels =num_channels ,fast_test_dimensionality=test_dimens, device = device)

        # size = self.Product5.calc_size(0)
        # self.Lin_2_classifier = Multilin(int(num_channels3/4),10,size,Non_lin=True)
        # self.Lin_1 = Multilin(num_channels3,int(num_channels3/4),size,Non_lin=True)
        # self.Lin_2 = Multilin(int(num_channels3/4),int(num_channels3/4),size,Non_lin=True)

        # self.ugly_lin_1 = nn.Linear(size,size)
        # self.ugly_lin_2 = nn.Linear(size,1)
        # self.ugly_lin_11 = nn.Linear(size,size)
        # self.ugly_lin_21 = nn.Linear(size,1)

        #size = self.Product2.calc_size(n_output)
        # self.drop = torch.nn.Dropout1d(p=0.5)
        # #self.mask = self.Product0.create_mask_decrease(32,16)
        # self.linout = nn.Linear(num_channels3,int(num_channels3/4))
        # self.linout2 = nn.Linear(int(num_channels3/4),int(num_channels3/4))
        # self.linout3 = nn.Linear(int(num_channels3/4),10)


        size = self.Product0.calc_size(4)
        self.size = size
        self.Rnorm = nn.RMSNorm([size,2],elementwise_affine=False)
        self.Rnorm2 = nn.RMSNorm([size,2],elementwise_affine=False)
        self.Rnorm3 = nn.RMSNorm([10],elementwise_affine=False)
        #self.Input_lin_0 = Multilin(10,10,size,Non_lin=True)
        # self.Input_lin_1 = Multilin(10,10,size,Non_lin=True)
        # self.Input_lin_2 = Multilin(10,10,size,Non_lin=True)

        self.linout = nn.Linear(size,10)
        self.linout2 = nn.Linear(10,1)#, bias=False)

        # self.linout = Multilin_dim2(size,size,10,Non_lin=False)
        # self.linout2 = Multilin_dim2(size,1,10,Non_lin=False)
        self.linout3 = nn.Linear(10,10, bias=False)
        self.linout4 = nn.Linear(10,10, bias=False)
        #self.linout5 = nn.Linear(20,10)
        self.dropout = nn.Dropout(p=0.2)

        # self.linout20 = nn.Linear(int(num_channels3/4),int(num_channels3/4))
        # self.linout21 = nn.Linear(int(num_channels3/4),int(num_channels3/4))
        # self.linout22 = nn.Linear(int(num_channels3/4),int(num_channels3/4))
        # self.linout23 = nn.Linear(int(num_channels3/4),int(num_channels3/4))
    def forward(self, x) -> torch.tensor:
        eps = 0.0000000000000000000001
        #x = x.unsqueeze(-3)
        x = self.Product0(x,x)
        x = self.Rnorm(x)
        #x = self.Input_lin_0(x)
        # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        # x = x*a *10
        #x = self.Product01(self.Input_lin_2(x),self.Input_lin_2(x))
        x = self.Product01(x,x)
        x = self.Rnorm2(x)
        # a = 1/(torch.sum(torch.abs(x2),dim=(-1,-2,-3),keepdim=True)+eps)
        # x = x2*a *10
        #x = self.Product01(self.Input_lin_2(x),self.Input_lin_2(x))
        # x3 = self.Product02(x2,x2)
        # a = 1/(torch.sum(torch.abs(x3),dim=(-1,-2,-3),keepdim=True)+eps)
        # x3 = x3*a *10
        #x = self.Product01(self.Input_lin_2(x),self.Input_lin_2(x))
        # x4 = self.Product03(x3,x3)
        # a = 1/(torch.sum(torch.abs(x4),dim=(-1,-2,-3),keepdim=True)+eps)
        # x4 = x4*a *10
        # #x =4self.Product01(self.Input_lin_2(x),self.Input_lin_2(x))
        # x5 = self.Product04(x4,x4)
        # a = 1/(torch.sum(torch.abs(x5),dim=(-1,-2,-3),keepdim=True)+eps)
        #x5 = x5*a*10*self.size
        # x = x*a *10
        # x = self.Product01(self.Input_lin_2(x),self.Input_lin_2(x))
        # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        # x = x*a *10
        # x = self.Product02(self.Input_lin_2(x),self.Input_lin_2(x))
        # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        # x = x*a *10
        # x = self.Product001(self.Input_lin_2(x),self.Input_lin_2(x))
        # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        #x = torch.cat((x,x2,x3,x4,x5),-3)
        #x = self.Product01(x,x)
        #x = self.Product02(x,x)


        x = torch.sum(torch.square(x), dim=-1)
        # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2),keepdim=True)+eps)
        # x = x*a*10
        #x = self.dropout(x)
        x = F.relu(self.linout(x))
        x = self.Rnorm3(x)
        # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2),keepdim=True)+eps)
        # x = x*a*10
        #x = self.dropout(x)
        x = F.relu(self.linout2(x)).squeeze()


        x = self.norm(x)
        #x = self.dropout(x)
        x = F.relu(self.linout3(x))
        x = self.norm1(x)
        #x = self.dropout(x)
        x = (self.linout4(x))


        # x = F.relu(self.linout4(x))
        # x = self.norm2(x)
        # x = self.linout5(x)
        #x = self.norm3(x)

        # size = x.size(-2)
        # # print(x.size())
        # x = x.reshape(-1,size,2)
        # x = self.drop(x)
        # x = x.view(-1,10,size,2)
        ##print('encode')
        ##print(torch.sum(torch.abs(x),dim =(-1,-2))[0,0:2])
        #a = 1/(torch.sum(torch.abs(x),dim =(-1,-2))+eps)
        #x = torch.einsum('ijkl,ij->ijkl', x,a)

        # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        # #print(a[0])
        # x = x*a
        # x = self.Product01(x,x)


        # # size = x.size(-2)
        # # x = x.reshape(-1,size,2)
        # # x = self.drop(x)
        # # x = x.view(-1,10,size,2)


        # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        # #print(a[0])
        # x = x*a


        # # size = x.size(-2)
        # # # print(x.size())
        # # x = x.reshape(-1,size,2)
        # # x = self.drop(x)
        # # x = x.view(-1,50,size,2)


        # x = self.Product02(x,x)

        # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        # #print(a[0])
        # x = x*a
        # x = self.Product001(x,x)


        # # size = x.size(-2)
        # # x = x.reshape(-1,size,2)
        # # x = self.drop(x)
        # # x = x.view(-1,10,size,2)


        # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        # #print(a[0])
        # x = x*a
        # x = self.Product002(x,x)

        # # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        # # #print(a[0])
        # # x = x*a
        # # x = self.Product0001(x,x)


        # # # size = x.size(-2)
        # # # x = x.reshape(-1,size,2)
        # # # x = self.drop(x)
        # # # x = x.view(-1,10,size,2)


        # # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        # # #print(a[0])
        # # x = x*a
        # # x = self.Product0002(x,x)

        # # size = x.size(-2)
        # # x = x.reshape(-1,size,2)
        # # x = self.drop(x)
        # # x = x.view(-1,10,size,2)


        # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        # #print(a[0])
        # x = x*a
        # x = self.Product1(x,x)


        # # size = x.size(-2)
        # # x = x.reshape(-1,size,2)
        # # x = self.drop(x)
        # # x = x.view(-1,10,size,2)


        # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        # #print(a[0])
        # x = x*a

        # x = self.Product2(x,x)


        # # size = x.size(-2)
        # # x = x.reshape(-1,size,2)
        # # x = self.drop(x)
        # # x = x.view(-1,10,size,2)

        # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        # #print(a[0])
        # x = x*a

        # # x = self.Product3(x,x)
        # # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        # # #print(a[0])
        # # x = x*a

        # # x = self.Product4(x,x)
        # # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        # # #print(a[0])
        # # x = x*a


        # # x = self.Product5(x,x)
        # # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        # # #print(a[0])
        # # x = x*a


        # x = self.Product50(x,x)
        # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        # #print(a[0])
        # x = x*a

        # ##print('encode_done')

        # #x = self.Lin_2(self.Lin_1(x))



        # x = self.Product_classifier(x,x)
        # a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        # #print(a[0])
        # x = x*a
        #x = x[:,:,0,0]
        #x = self.Lin_2_classifier(self.Lin_1(x))
        #a = 1/(torch.sum(torch.abs(x),dim=(-1,-2,-3),keepdim=True)+eps)
        #x = x*a
        #print(a[0])
        #print(x[0])
        #size = x.size(0)
        #x = x.reshape(size,-1)
        #x = F.relu(self.linout2(F.relu(self.linout(x))))
        # x = F.relu(self.linout20(x))
        # x = F.relu(self.linout21(x))
        # x = F.relu(self.linout22(x))
        # x = F.relu(self.linout23(x))
        #x = self.linout3(x)
        # y = x[:,:,:,0]
        # z = x[:,:,:,1]
        # x = self.ugly_lin_2(F.relu(self.ugly_lin_1(y))).squeeze() + self.ugly_lin_21(F.relu(self.ugly_lin_11(z))).squeeze()
        #x = self.norm(x)
        #print(x[0])
        return x
