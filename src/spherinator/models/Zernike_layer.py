import math
import os.path
from importlib.util import LazyLoader

import numpy
import numpy as np
import scipy
import torch
import torch.nn as nn
from numpy import convolve
from skimage.measure import block_reduce

#import lightning.pytorch as pl


class  Non_linearity(nn.Module):
    '''
    A Non-linearity defined for Fourier objects. The Non-linearity needs to act on the absolute value of sqrt((Z_m^n)**2 + (Z_-m^n)**2), in order to not break equivariance.
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
class Lintrans3(nn.Module):
    def __init__(self, inp,out,Non_lin=False):
        super().__init__()
        self.lin = nn.Linear(inp,out,bias = False)
        self.Non_lin_bool = Non_lin
        if Non_lin:
            self.Non_lin = Non_linearity()
    def forward(self,x):
        x = torch.transpose((self.lin(torch.transpose(x,-1,-3))),-1,-3)
        if self.Non_lin_bool:
            x = self.Non_lin(x)
        return x
class Multilin(nn.Module):
    '''
    A class that creates a linear layer with independant weights along it's channels.
    In this framework, this serves to create a linear layer convolving the channels with each other, but doing this in a way that is independent for different m,n.
    Yet, it acts identical on n,m and n,-m; therefore keeping rotation equivariance
    '''
    def __init__(self, inp,out,size,Non_lin=False,normalize=False):
        super().__init__()
        #self.device = 'cuda:2'
        #self.weight_lin = torch.nn.parameter.Parameter(torch.nn.init.normal_(torch.empty(size,inp,out, device = self.device)))#/np.sqrt(inp))
        self.weight_lin = torch.nn.parameter.Parameter(torch.nn.init.normal_(torch.empty(size,inp,out)))#/np.sqrt(inp))
        self.Non_lin_bool = Non_lin
        if Non_lin:
            self.Non_lin = Non_linearity(normalize=normalize)
    def forward(self,x):
        x = torch.einsum('ijk,...jil->...kil',self.weight_lin,x)
        if self.Non_lin_bool:
            x = self.Non_lin(x)
        return x



class Fourier_layer(nn.Module):
    def __init__(self, n_max = 30,n_max_2=None, n_out=30, multichanneled = False,in_channels = 1 ,intermediate_channels=1, out_channels =1 ,last_layer = False, fast_test_dimensionality = False,normalize=False, device = 'cuda:2'):
        super().__init__()
        #self.device = device
        '''
        The core Fourier layer. The final forward method is essentially yust simple matrix multiplication, most of the things happen in preprocessing.

        The arguments n_max and n_max_2 are the orders of the Fourier Vectors input into the Network. If n_max_2 is not set, it is set to n_max_2 = n_max.
        n_out is the desired Order of the output. All inputs get embedded into the space of the highest of the three n, and the product performed in that space.
        Finally, if n_out is smaller than this n, only the terms corresponding to n_out get output.

        The argument "fast_test_dimensionality" skips that processing to test wether everything else works. This is due to the processing taking rather long, which is annoying for bugfixing.


        '''
        size = max(n_max,n_out)
        self.device = device
        #print(self.device)

        self.increase_in1 = False
        if n_max_2 is not None:
            if n_max_2 < n_max:
                print('For inputs of different Orders, please use in2 as the input of higher order')
                Donkey
        if fast_test_dimensionality:
            print('Warning, this layer is completely unfunctional, yet will load faster, so you can test wether all dimensions of your Tensors add up')
        if n_max < n_out:
            #batch_size=1
            self.increase_in1 = True
            if fast_test_dimensionality:
                self.Fourier_matrix = torch.zeros(out_size,out_size,out_size,4, device = self.device)
            else:
                if os.path.isfile('../Fourier_layer_matrix_29{}'.format(n_out)):
                    self.Fourier_matrix = torch.load('../Fourier_layer_matrix_29{}'.format(n_out))
                else:
                    print('processing Fourier layer')
                    self.Fourier_matrix = torch.tensor(self.Zernicke_matrix_generator(n_out),dtype=torch.float)
                    torch.save(self.Fourier_matrix,'../Fourier_layer_matrix_29{}'.format(n_out))
                #size = self.calc_size(n_max)
                self.Fourier_matrix = torch.tensor(self.Fourier_matrix).to(self.device)
                #self.Fourier_matrix = torch.tensor(self.Zernicke_matrix_generator(n_out),dtype=torch.float, device = self.device)

        self.increase_in2 = False
        if n_max_2 is None:
            if self.increase_in1 == True:
                self.increase_in2 = True
        elif n_max_2 < n_out:
            if self.increase_in1 == False:
                if fast_test_dimensionality:
                    self.Fourier_matrix = torch.zeros(out_size,out_size,out_size,4, device = self.device)
                else:
                    if os.path.isfile('../Fourier_layer_matrix_29{}'.format(n_out)):
                        self.Fourier_matrix = torch.load('../Fourier_layer_matrix_29{}'.format(n_out))
                    else:
                        print('processing Fourier layer')
                        self.Fourier_matrix = torch.tensor(self.Zernicke_matrix_generator(n_out),dtype=torch.float)
                        torch.save(self.Fourier_matrix,'../Fourier_layer_matrix_29{}'.format(n_out))
                    #size = self.calc_size(n_max)
                    self.Fourier_matrix = torch.tensor(self.Fourier_matrix).to(self.device)
            self.increase_in2 = True
        if self.increase_in1 == False and self.increase_in2 == False:
            if fast_test_dimensionality:
                self.Fourier_matrix = torch.zeros(size,size,size,4, device = self.device)
            else:
                if os.path.isfile('../Fourier_layer_matrix_29{}'.format(n_max)):
                    self.Fourier_matrix = torch.load('../Fourier_layer_matrix_29{}'.format(n_max))
                else:
                    print('processing Fourier layer')
                    self.Fourier_matrix = torch.tensor(self.Zernicke_matrix_generator(n_max),dtype=torch.float)
                    torch.save(self.Fourier_matrix,'../Fourier_layer_matrix_29{}'.format(n_max))
                #size = self.calc_size(n_max)
                self.Fourier_matrix = torch.tensor(self.Fourier_matrix).to(self.device)




        Matrix_plus = [[[1/2,0],[0,-1/2]],[[0,1/2],[1/2,0]]]
        Matrix_minus_pos =[[[1/2,0],[0,1/2]],[[0,-1/2],[1/2,0]]]
        Matrix_minus_neut =[[[1/2,0],[0,1/2]],[[0,0],[0,0]]]
        Matrix_minus_neg =[[[1/2,0],[0,1/2]],[[0,1/2],[-1/2,0]]]
        self.transform = torch.tensor(np.array([Matrix_plus,Matrix_minus_pos,Matrix_minus_neut,Matrix_minus_neg]),dtype=torch.float, device = self.device)
        if normalize:
            self.Nonlin = Non_linearity(normalize=True)
        else:
            self.Nonlin = Non_linearity(normalize=False)
        self.weight = torch.nn.parameter.Parameter(torch.nn.init.normal_(torch.empty(n_max+1,n_max+1)))#/size)#/(size**3))#+torch.ones(size,size))#*(size**3))#/size)

        self.reduce = False
        if n_max > n_out:
            self.reduce = True
        self.last_layer = last_layer
        self.multichanneled = False
        if multichanneled != False:
            self.multichanneled = True
        if multichanneled == 'same':
            # Possibly make this independant for each m,n. This would lead to a more pronounced role of multiple channels, yet drastically increase complexitly.
            self.In_Lin1 = Lintrans3(in_channels,intermediate_channels,Non_lin=True)
            self.In_Lin2 = Lintrans3(in_channels,intermediate_channels,Non_lin=True)
            self.Out_Lin  = Lintrans3(intermediate_channels,out_channels)
        if multichanneled == 'independant':
            # Possibly make this independant for each m,n. This would lead to a more pronounced role of multiple channels, yet drastically increase complexitly.
            self.In_Lin1 = Multilin(in_channels,intermediate_channels,size,Non_lin=True,normalize=False)
            self.In_Lin2 = Multilin(in_channels,intermediate_channels,size,Non_lin=True,normalize=False)

            self.Out_Lin  = Multilin(intermediate_channels,out_channels,size)


        if n_max_2 is not None:
            n_maximal = max(n_max,n_max_2)
        else:
            n_maximal = n_max
        if n_out < n_maximal:
            print('implement size decrease')



    def Zernicke_matrix_generator(self,n_max):
        '''
        Iterating over all pairs of inputs to generate the full conversion matrix

        '''
        n_max_calc = n_max+1
        grid = np.zeros((n_max_calc,n_max_calc,n_max_calc,4))
        for m1 in range(0, n_max_calc):

            for m2 in range(0, n_max_calc):
                m_out1 = np.abs(m1-m2)
                m_out2 = np.abs(m1+m2)
                #print(len(x))
                grid[m1,m2,m_out1,0] = 1
                if m_out2 < n_max:
                    if m1> m2:
                        grid[m1,m2,m_out2,1] = 1
                    if m1 == m2:
                        grid[m1,m2,m_out2,2] = 1
                    if m1 < m2:
                        grid[m1,m2,m_out2,3] = 1

        return grid


    def forward(self,in1,in2):
        # if self.increase_in1:
        #     '''
        #     Increasing size of input 1 in case it needs to
        #     '''
        #     in1 = torch.einsum('ij,...jk->...ik',self.in_mask_1,in1)
        # if self.increase_in2:
        #     '''
        #     Increasing size of input 2 in case it needs to
        #     '''
        #     in2 = torch.einsum('ij,...jk->...ik',self.in_mask_2,in2)
        # if self.multichanneled:
        #     '''
        #     If there are multiple channels, we first apply a linear layer along the channel dimension. The nonlinearity gets called within the linear layer
        #     '''
        #     # in1 = self.deep_In_Lin1(in1)
        #     # in2 = self.deep_In_Lin2(in2)
        #     in1 = self.In_Lin1(in1)
        #     in2 = self.In_Lin2(in2)

        # '''
        # We explicitely set all terms of m =-0 to zero
        # '''
        # in1 = torch.einsum('ij,...ij->...ij',self.zero_mask,in1)
        # in2 = torch.einsum('ij,...ij->...ij',self.zero_mask,in2)
        # '''
        # We multiply our inputs to the Fourier matrix. A weight Matrix is added that is able to learn what interactions the model is supposed to favor
        # '''



        in1 = torch.einsum('...axim,ij,ab,...byjn->...abxyijmn', in1,self.weight,self.weight,in2)
        # print(in1.size())
        # print(self.Fourier_matrix.size())
        in1 = torch.einsum('ijkl,...ijmn->...klmn',self.Fourier_matrix,in1)

        in1 = torch.einsum('ijkl,...ijmnabcd->...klmnabcd',self.Fourier_matrix,in1)

        # Do not put anything inbetween, as it might break equivariance
        '''
        We collapse the four different channels that correspond to the different cases considered above to their representation in +- m.
        '''
        out = torch.einsum('lamn,...klmn->...ka', self.transform,in1)
        out = torch.einsum('lamn,...klmnde->...kade', self.transform,out)




        # if not self.last_layer:
        #     '''
        #     Afterwards, we call a Non-linearity
        #     '''
        #     out = self.Nonlin(out)
        # #a = torch.sum(torch.sqrt(torch.sum(torch.square(out),dim =(-1),keepdim=False)),dim=-1)
        # #print(a)
        # if self.multichanneled and not self.last_layer:
        #     '''
        #     Finally, one more linear layer along channel dimension
        #     '''
        #     #out = self.deep_Out_Lin(out)
        #     out = self.Out_Lin(out)


        return out


'''

           __n__n__
    .------`-\00/-'
   /  ##  ## (oo)
  / \## __   ./
     |//YY \|/
     |||   |||

     Renata


A cow (her name is Renata) to improve Code beauty for this terribly coded class


'''


# class Fourier_Norms(nn.Module):
#     '''
#     As the normalization of the polinomials depends on the image size, the norm is calculated numerically here.
#     Normalizing all filters to the same value is useful, as it allows us to recrate images by simply adding the filters*coefficients up.
#     Not having them normalized leads to an overrepresentation of filters of higher norm.

#     '''
#     def __init__(self, n_max = 30, device= 'cuda:2',numerical_expand=4):
#         super().__init__()
#         self.num = numerical_expand
#         #self.device = device
#         #self.norm_output = self.calc_norms(n_max).to(device)



#         if os.path.isfile('../Fourier_norms_29_{}'.format(n_max)):
#             self.norm_output = torch.load('../Fourier_norms_29_{}'.format(n_max))
#         else:
#             print('processing norms')
#             self.norm_output = self.calc_norms(n_max)
#             torch.save(self.norm_output,'../Fourier_norms_29_{}'.format(n_max))
#         #size = self.calc_size(n_max)

#         #self.norm_output= torch.tensor(self.norm_output).to(device)
#         #self.device = 'cuda:2'



#     def Radial_function(self,n,m, n_max):
#         faktor = []

#         for i in range(n_max-n):
#             faktor.append(0)

#         for k in range(int((n-m)/2+1)):
#             faktor.append((-1)**k * math.factorial(n-k) /(math.factorial(k) * math.factorial(int((n+m)/2-k))* math.factorial(int((n-m)/2-k)))   )
#             if k != int((n-m)/2):
#                 faktor.append(0)
#             #exp.append(n-2*k)

#         for i in range(m):
#             faktor.append(0)
#         faktor = np.array(faktor)
#         return np.flip(faktor)

#     def Zernicke_embedding_generator(self,n_max):
#         n_max_calc = n_max+1
#         lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
#         Basis = np.zeros((lengh,n_max+1))
#         #Basis = []
#         Basis = [[None for i in range(int((n_max+1)))]for i in range(int((n_max+1)))]
#         #print(np.shape(Basis))
#         for m1 in range(0, n_max+1):
#             #m1_lengh = lengh - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
#             count=0
#             for n1 in range(m1,n_max+1,2):
#                 #print(n1,m1)
#                 Basis[n1][m1] = self.Radial_function(n1,m1,n_max)
#                 count+=1
#         return Basis
#     def mask(self,x,z):
#         y = (x**2+z**2)
#         return np.where(y<1,1,0)


#     def calc_norms(self,n_max):

#         eps = np.finfo(float).eps

#         Fourier_functions = self.Zernicke_embedding_generator(n_max)

#         grid_extend = 1
#         #grid_resolution = 680
#         z = x = np.linspace(-grid_extend, grid_extend, int(29*self.num))
#         z, x = np.meshgrid(z, x)

#         #print(Fourier_functions)
#         # Use epsilon to avoid division by zero during angle calculations
#         functions = [[[] for i in range(int((n_max+1)))]for i in range(int((n_max+1)))]
#         #print(Fourier_functions)
#         for i in range(len(Fourier_functions)):
#             for j in range(len(Fourier_functions)):
#                 if Fourier_functions[i][j] is None:
#                     functions[i][j] = numpy.polynomial.polynomial.Polynomial([0])
#                     #print('None')
#                 else:
#                     functions[i][j] = (numpy.polynomial.polynomial.Polynomial(Fourier_functions[i][j]))
#         out = [[[] for i in range(int((n_max+1)))]for i in range(int((n_max+1)))]
#         for i in range(len(Fourier_functions)):
#             for j in range(len(Fourier_functions)):
#                 #print(self.device)
#                 out[i][j] = np.array(functions[i][j](np.sqrt((x ** 2 + z ** 2)))*np.cos(j*np.arctan2(x , (z ))))





#         out = np.array(out)
#         out =torch.tensor( block_reduce(out,(1,1, self.num, self.num),func=np.sum),dtype=torch.float)


#         out = torch.tensor(out)

#         z = x = np.linspace(-grid_extend, grid_extend, int(29))
#         z, x = np.meshgrid(z, x)
#         out_mask = torch.tensor( self.mask(x,z))
#         out= torch.einsum('ijkl,kl->ijkl',out,out_mask)

#         #norm = torch.sqrt((torch.sum((out)**2,dim= (-1,-2),keepdim = True)))*self.num+eps




#         norm = [[None for i in range(int((n_max+1)))]for i in range(int((n_max+1)))]
#         for i in range(len(Fourier_functions)):
#             for j in range(len(Fourier_functions)):
#                 norm[i][j] = torch.sqrt(torch.sum((out[i][j])**2,dim= (-1,-2),keepdim = False)).item()#*1000000000000

#         # for i in range(len(Fourier_functions)):
#         #     for j in range(len(Fourier_functions)):
#         #         print(norm[i][j],i,j)
#         return norm+eps
#     def forward(self):
#         return self.norm_output
#         #return self.norm_output
