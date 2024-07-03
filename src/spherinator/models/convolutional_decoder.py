import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy
import numpy as np
from numpy import convolve
import scipy
import math


class ConvolutionalDecoder(pl.LightningModule):
    def __init__(self, h_dim: int = 256):
        super().__init__()
        test_dimens = False
        '''
        n_max = 8
        self.Product1 = Zernike_layer( n_max = n_max, n_out=10,in_channels = 1 ,intermediate_channels=1, out_channels =1 ,fast_test_dimensionality=test_dimens)
        self.Input1 =  torch.nn.parameter.Parameter(torch.rand((1,int(((n_max+2)*(n_max+1)/2)/2+math.ceil((n_max+1)/4)),2)))

        n_max = 10
        self.Product2 = Zernike_layer( n_max = n_max, n_out=14,in_channels = 1 ,intermediate_channels=1, out_channels =1 ,fast_test_dimensionality=test_dimens)
        self.Input2 =  torch.nn.parameter.Parameter(Init_zero(n_max))

        n_max = 14
        self.Product3 = Zernike_layer( n_max = n_max, n_out=20,in_channels = 1 ,intermediate_channels=1, out_channels =1 ,fast_test_dimensionality=test_dimens, normalize=True)
        self.Input3 =  torch.nn.parameter.Parameter(Init_zero(n_max))

        n_max = 20
        self.Product4 = Zernike_layer( n_max = n_max, n_out=26, multichanneled = 'independant',in_channels = 10 ,intermediate_channels=5, out_channels =10 ,fast_test_dimensionality=test_dimens, normalize=True)
        self.Input4 =  torch.nn.parameter.Parameter(Init_zero(n_max))
        '''
        n_max = 26
        self.Product5 = Zernike_layer( n_max = n_max, n_out=32,multichanneled = 'independant',in_channels = 10 ,intermediate_channels=5, out_channels =1 ,fast_test_dimensionality=test_dimens)
        self.Input5 =  torch.nn.parameter.Parameter(Init_zero(n_max))
        '''
        n_max = 32
        self.Product6 = Zernike_layer( n_max = n_max, n_out=32, multichanneled = 'independant',in_channels = 10 ,intermediate_channels=5, out_channels =1 ,fast_test_dimensionality=test_dimens)
        self.Input6 =  torch.nn.parameter.Parameter(Init_zero(n_max))
        n_max = 32
        self.Product1 = Zernike_layer( n_max = n_max, n_out=32, multichanneled = 'independant',in_channels = 10 ,intermediate_channels=5, out_channels =1 ,fast_test_dimensionality=test_dimens)
        self.Input1 =  torch.nn.parameter.Parameter(torch.rand((1,int(((n_max+2)*(n_max+1)/2)/2+math.ceil((n_max+1)/4)),2)))
        '''

    def forward(self, x) -> torch.tensor:
        #x = self.Product1(x,self.Input1)

        #x = self.Product2(x,self.Input2)
        #x = self.Product3(x,self.Input3)

        #x = self.Product4(x,self.Input4)
        x = self.Product5(x,self.Input5)
        #x = self.Product6(x,self.Input6)

        return x





def Init_zero(n_max):
    n_max_calc = n_max+1
    zeros = (torch.zeros((1,int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4)),2)))
    zeros[0,0,0] = 1
    return zeros








class  Non_linearity(nn.Module):
    def __init__(self, normalize= False):
        super().__init__()
        self.lin = nn.Linear(1,1)
        self.normalize = normalize
    def forward(self,x):
        a = self.lin(torch.sum(x**2, dim=-1,keepdim=True))
        b = (torch.tanh(a)+1)/2
        if self.normalize:
            a = torch.sqrt(torch.sum(a,dim=-2,keepdim=True))
            return b*x/a
        else:
            return b * x
class Lintrans3(nn.Module):
    def __init__(self, inp,out,Non_lin=False):
        super().__init__()
        self.lin = nn.Linear(inp,out)
        self.Non_lin_bool = Non_lin
        if Non_lin:
            self.Non_lin = Non_linearity()
    def forward(self,x):
        x = torch.transpose((self.lin(torch.transpose(x,-1,-3))),-1,-3)
        if self.Non_lin_bool:
            x = self.Non_lin(x)
        return x
class Multilin(nn.Module):
    def __init__(self, inp,out,size,Non_lin=False):
        super().__init__()
        self.device = 'cuda:2'
        # try torch.nn.init.normal_  and torch.nn.init.kaiming_uniform_
        self.weight_lin = torch.nn.parameter.Parameter(torch.nn.init.normal_(torch.empty(size,inp,out, device = self.device)))#/np.sqrt(inp))
        #self.bias = torch.nn.parameter.Parameter(torch.zeros(size,1, device = self.device))
        self.Non_lin_bool = Non_lin
        #print(size,inp,out)
        if Non_lin:
            self.Non_lin = Non_linearity()
    def forward(self,x):
        #print(x.size())
        #print(self.weight_lin.size())
        x = torch.einsum('ijk,...jil->...kil',self.weight_lin,x)
        #x = x+ self.bias
        if self.Non_lin_bool:
            x = self.Non_lin(x)
        return x


class Zernike_layer(nn.Module):
    def __init__(self, n_max = 30, n_out=30, multichanneled = False,in_channels = 1 ,intermediate_channels=1, out_channels =1 ,last_layer = False, fast_test_dimensionality = False,normalize=False, device = 'cuda:2'):
        super().__init__()
        self.device = device
        #test_tensor = torch.ones(3, device = self.device)
        #print('initialized_tensor')

        self.increase = False
        if fast_test_dimensionality:
            print('Warning, this layer is completely unfunctional, yet will load faster, so you can test wether all dimensions of your Tensors add up')
        if n_max < n_out:
            #batch_size=1
            Zernike_normalization = Zernike_Norms(n_out)
            self.Zernike_normalization = Zernike_normalization()
            self.increase = True
            self.in_mask = self.create_mask_increase(n_max,n_out)
            out_size= self.calc_size(n_out)
            self.input1_expand = torch.zeros(512,in_channels,out_size,2, device = self.device)
            self.input2_expand = torch.zeros(512,in_channels,out_size,2, device = self.device)
            if fast_test_dimensionality:
                self.Zernike_matrix = torch.zeros(out_size,out_size,out_size,4, device = self.device)
            else:
                self.Zernike_matrix = torch.tensor(self.Zernicke_matrix_generator(n_out),dtype=torch.float, device = self.device)
            size = self.calc_size(n_out)
        else:
            size = self.calc_size(n_max)
            Zernike_normalization = Zernike_Norms(n_max)
            self.Zernike_normalization = Zernike_normalization()
            if fast_test_dimensionality:
                self.Zernike_matrix = torch.zeros(size,size,size,4, device = self.device)
            else:
                self.Zernike_matrix = torch.tensor(self.Zernicke_matrix_generator(n_max),dtype=torch.float, device = self.device)
        Matrix_plus = [[[1/2,0],[0,-1/2]],[[0,1/2],[1/2,0]]]
        Matrix_minus_pos =[[[1/2,0],[0,1/2]],[[0,1/2],[-1/2,0]]]
        Matrix_minus_neg =[[[1/2,0],[0,1/2]],[[0,-1/2],[1/2,0]]]
        Matrix_minus_neut =[[[1/2,0],[0,1/2]],[[0,0],[0,0]]]
        self.transform = torch.tensor(np.array([Matrix_plus,Matrix_minus_pos,Matrix_minus_neut,Matrix_minus_neg]),dtype=torch.float, device = self.device)
        if normalize:
            self.Nonlin = Non_linearity(normalize=True)
        else:
            self.Nonlin = Non_linearity(normalize=False)
        self.weight = torch.nn.parameter.Parameter(torch.nn.init.normal_(torch.empty(size,size, device = self.device))/size)
        self.reduce = False
        if n_max > n_out:
            self.reduce = True
            self.out_mask = self.create_mask_decrease(n_max,n_out)
        self.last_layer = last_layer
        self.multichanneled = False
        if multichanneled != False:
            self.multichanneled = True
        if multichanneled == 'same':
            # Possibly make this independant for each m,n. This would lead to a more pronounced role of multiple channels, yet drastically increase complexitly.
            self.In_Lin1 = Lintrans3(in_channels,intermediate_channels)
            self.In_Lin2 = Lintrans3(in_channels,intermediate_channels)
            self.Out_Lin  = Lintrans3(intermediate_channels,out_channels)
        if multichanneled == 'independant':
            # Possibly make this independant for each m,n. This would lead to a more pronounced role of multiple channels, yet drastically increase complexitly.
            self.In_Lin1 = Multilin(in_channels,intermediate_channels,size,Non_lin=True)
            self.In_Lin2 = Multilin(in_channels,intermediate_channels,size,Non_lin=True)
            self.Out_Lin  = Multilin(intermediate_channels,out_channels,size)
    def calc_size(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        return lengh
    def create_mask_decrease(self,n_max,n_out):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        mask = torch.ones(lengh, device = self.device)
        for m1 in range(0, n_max+1):
            m1_lengh = lengh - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
            count=0
            for n1 in range(m1,n_max+1,2):
                if m1>n_out or n1>n_out:
                    mask[m1_lengh+count] -= 1
                count+=1
        mask = mask.bool()
        return mask
    def create_mask_increase(self,n_max,n_out):
        n_out_calc = n_out+1
        n_max_calc = n_max+1
        lengh = int(((n_out_calc+1)*n_out_calc/2)/2+math.ceil(n_out_calc/4))
        lengh_in = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        mask = torch.zeros(lengh,lengh_in, device = self.device)
        for m1 in range(0, n_out+1):
            m1_lengh = lengh - int(((n_out_calc-m1+1)*(n_out_calc-m1)/2)/2+math.ceil((n_out_calc-m1)/4))
            count=0
            for n1 in range(m1,n_out+1,2):
                m_in_lengh = lengh_in - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
                if not (m1>n_max or n1>n_max):
                    mask[m1_lengh+count, m_in_lengh+count] += 1
                count+=1
        #mask = mask.bool()
        return mask
    def Radial_function(self,n,m, n_max):
        faktor = []
        '''
        scaling = []
        for i in range(n_max+n_max+1):
            scaling.append(1/((2*n_max-i)**2+2))
        '''
        for i in range(n_max-n):
            faktor.append(0)

        for k in range(int((n-m)/2+1)):
            faktor.append((-1)**k * math.factorial(n-k) /(math.factorial(k) * math.factorial(int((n+m)/2-k))* math.factorial(int((n-m)/2-k)))   )
            if k != int((n-m)/2):
                faktor.append(0)
            #exp.append(n-2*k)

        for i in range(m):
            faktor.append(0)

        #print('hi')
        #print(np.shape(scale))
        norm = self.Zernike_normalization[n][m]
        #print(np.shape(norm))
        #print(np.shape(faktor))
        faktor = np.array(faktor)
        faktor = faktor/norm
        #scale = np.einsum('i,i', scaling,scale)

        #faktor = np.array(faktor/scale)
        return np.flip(faktor)




    def Radial_function_matrix(self,m, n_max):
        #scaling = []
        matrix = None
        matrix = []
        empty = np.zeros(n_max+1)
        for i in range(m):
            empty *=0
            empty[n_max-i] = 1
            ##print('hi',empty)
            matrix.append(empty.copy())
        ##print(matrix)
        #for i in range(n_max+n_max+1):
        #    scaling.append(1/((2*n_max-i)**2+2))
        for n in range(m,n_max+1,2):
            faktor = []
            for i in range(int((n_max-n))):
                faktor.append(0)
            for k in range(int((n-m)/2+1)):
                faktor.append((-1)**k * math.factorial(n-k) /(math.factorial(k) * math.factorial(int((n+m)/2-k))* math.factorial(int((n-m)/2-k)))   )
                if k !=int((n-m)/2):
                    faktor.append(0)

            for i in range(m):
                faktor.append(0)
            #scale = convolve(faktor,faktor)
            #print(np.shape(scale))
            #scale = np.einsum('i,i', scaling,scale)
            norm = self.Zernike_normalization[n][m]
            faktor = np.array(faktor)/norm

            matrix.append((faktor.copy()))

            if n != n_max:
                empty *=0
                empty[n_max-n-1] = 1
                matrix.append(empty.copy())
        ##print(matrix)

        ##print(scaling)
        matrix = (np.rot90(numpy.vstack(np.array(matrix))))
        #matrix = np.where(matrix>0.0000000000001, matrix,0)
        #print('done')

        return scipy.linalg.solve_triangular(matrix, np.identity(n_max+1))
    def Multiply(self,x,y,n_max):
        # fix, work in no zero space
        x = np.flip(x)
        y = np.flip(y)
        return np.flip(convolve(x,y)[-n_max-1:])
    def Calculate_matrix_coefficients(self,m1,m2,n1,n2,n_max):
        In1 = self.Radial_function(n1,m1,n_max)
        In2 = self.Radial_function(n2,m2,n_max)
        Mult = self.Multiply(In1,In2,n_max)
        m_out1 = np.abs(m1-m2)
        m_out2 = np.abs(m1+m2)
        m_out2 = np.min((m_out2,n_max+1))
        Mat1 = self.Radial_function_matrix(m_out1,n_max)
        ##print(Mat1)
        if m_out2 > n_max:
            Mat2 = np.zeros((n_max+1,n_max+1))
        else:
            Mat2 = self.Radial_function_matrix(m_out2,n_max)
        lower=0
        higher = 0
        inbetween = 0
        #if not m_out1 == m_out2:
        for i in range(n_max+1,n_max+1-m_out1,-1):
            lower +=math.ceil(i/2)
        for i in range(1,n_max+1-m_out2):
            higher +=math.ceil(i/2)
        for i in range(n_max+1-m_out1-1,n_max+1-m_out2,-1):
            inbetween +=math.ceil(i/2)
        out1 = np.einsum('ij,j->i',Mat1,Mult)[m_out1:]
        out2 = np.einsum('ij,j->i',Mat2,Mult)[m_out2:]
        out1 = out1[::2]
        out2 = out2[::2]
        out_dim_0 = np.zeros(lower,dtype=float)

        if not m_out1 == m_out2:
            out_dim_0 = np.append(out_dim_0,np.zeros(len(out1)))
        out_dim_0 = np.append(out_dim_0,np.zeros(inbetween))
        out_dim_0 = np.append(out_dim_0,out2)
        out_dim_0 = np.append(out_dim_0,np.zeros(higher))
        #print(len(out_dim_0))
        #dackel
        out_dim_1 = np.zeros(lower,dtype=float)
        if m1>m2:
            out_dim_1 = np.append(out_dim_1,out1)
        else:
            out_dim_1 = np.append(out_dim_1,np.zeros(len(out1)))
        out_dim_1 = np.append(out_dim_1,np.zeros(inbetween))

        if not m_out1 == m_out2:
            out_dim_1 = np.append(out_dim_1,np.zeros(len(out2)))
        out_dim_1 = np.append(out_dim_1,np.zeros(higher))

        out_dim_2 = np.zeros(lower,dtype=float)
        if m1==m2:
            out_dim_2 = np.append(out_dim_2,out1)
        else:
            out_dim_2 = np.append(out_dim_2,np.zeros(len(out1)))
        out_dim_2 = np.append(out_dim_2,np.zeros(inbetween))
        if not m_out1 == m_out2:
            out_dim_2 = np.append(out_dim_2,np.zeros(len(out2)))
        out_dim_2 = np.append(out_dim_2,np.zeros(higher))

        out_dim_3 = np.zeros(lower,dtype=float)
        if m1<m2:
            out_dim_3 = np.append(out_dim_3,out1)
        else:
            out_dim_3 = np.append(out_dim_3,np.zeros(len(out1)))
        out_dim_3 = np.append(out_dim_3,np.zeros(inbetween))
        if not m_out1 == m_out2:
            out_dim_3 = np.append(out_dim_3,np.zeros(len(out2)))
        out_dim_3 = np.append(out_dim_3,np.zeros(higher))
        out = np.transpose(np.array([out_dim_0,out_dim_1,out_dim_2,out_dim_3]))
        return out

    def Zernicke_matrix_generator(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        grid = np.zeros((lengh,lengh,lengh,4))
        for m1 in range(0, n_max+1):
            m1_lengh = lengh - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
            #print(lengh)
            #print(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
            #print(m1_lengh)
            for m2 in range(0, n_max+1):
                m2_lengh = lengh - int(((n_max_calc-m2+1)*(n_max_calc-m2)/2)/2+math.ceil((n_max_calc-m2)/4))
                count1=0
                for n1 in range(m1,n_max+1,2):
                    count2=0
                    for n2 in range(m2,n_max+1,2):
                        #print('done')
                        x = self.Calculate_matrix_coefficients(m1,m2,n1,n2,n_max)
                        #print(len(x))
                        grid[m1_lengh+count1,m2_lengh+count2,:,:] = x
                        count2 +=1
                    count1 +=1
        return grid
    def mask(self,x,z):
        y = (x**2+z**2)
        return np.where(y<1,1,0)

    '''
    def Calc_norm(self,input,m):

        grid_extend = 1
        #grid_resolution = 680
        z = x = np.linspace(-grid_extend, grid_extend, 128)
        z, x = np.meshgrid(z, x)

        #print(Zernike_functions)
        # Use epsilon to avoid division by zero during angle calculations
        functions = numpy.polynomial.polynomial.Polynomial(input)

        eps = np.finfo(float).eps
        out=(functions(np.sqrt((x ** 2 + z ** 2)))*np.cos(m*np.arctan2(x , (z + eps))))#,functions(np.sqrt((x ** 2 + z ** 2)))*np.sin(m*np.arctan2(x ,(z  + eps)))])
        #print(out[0])
        # Add restriction to r<1
        out_mask = self.mask(x,z)
        out = torch.tensor(out*out_mask)

        norm = (torch.sum(torch.abs(out),dim= (-1,-2),keepdim = False)).item()
        return norm
    '''
    def forward(self,in1,in2):
        if self.increase:
            #print('size')
            #print(in1.size())
            #print(in2.size())
            '''
            self.input1_expand[:,:,self.in_mask] = in1
            self.input2_expand[:,:,self.in_mask] = in2
            in1 = self.input1_expand
            in2 = self.input2_expand
            '''
            in1 = torch.einsum('ij,...jk->...ik',self.in_mask,in1)
            in2 = torch.einsum('ij,...jk->...ik',self.in_mask,in2)
            #print(in1.size())
            #print(in2.size())
            #print('size')
        if self.multichanneled:
            in1 = self.In_Lin1(in1)
            in2 = self.In_Lin2(in2)

        out = torch.einsum('...im,ijkl,ij,...jn->...klmn', in1,self.Zernike_matrix,self.weight,in2)
        # Do not put anything inbetween, as it might break equivariance
        out = torch.einsum('lamn,...klmn->...ka', self.transform,out)
        if not self.last_layer:
            out = self.Nonlin(out)
        #print(out.size())
        #print(out[0,100:150,0])
        #print('middle')
        #print(out[0,-50:-1,0])
        #print('stop')
        if self.multichanneled and not self.last_layer:
            out = self.Out_Lin(out)
        if self.reduce:
            out = out[:,:,self.out_mask,:]
        #print(out.size())

        return out




##############################################################################################################################################################################################
##This is implemented in a way that is insanely stupid, and should seriously be rebuilt!
#####################################################################################################################################################################################
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
class Zernike_Norms(nn.Module):
    def __init__(self, n_max = 30 ):
        super().__init__()
        self.norm_output = self.calc_norms(n_max)



    def Radial_function(self,n,m, n_max):
        faktor = []
        #scaling = []
        #for i in range(n_max+n_max+1):
        #    scaling.append(1/((2*n_max-i)**2+2))

        for i in range(n_max-n):
            faktor.append(0)

        for k in range(int((n-m)/2+1)):
            faktor.append((-1)**k * math.factorial(n-k) /(math.factorial(k) * math.factorial(int((n+m)/2-k))* math.factorial(int((n-m)/2-k)))   )
            if k != int((n-m)/2):
                faktor.append(0)
            #exp.append(n-2*k)

        for i in range(m):
            faktor.append(0)
        #scale = convolve(faktor,faktor)
        #cale = np.einsum('i,i', scaling,scale)

        #faktor = np.array(faktor/scale)
        faktor = np.array(faktor)
        return np.flip(faktor)

    def Zernicke_embedding_generator(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        Basis = np.zeros((lengh,n_max+1))
        #Basis = []
        Basis = [[None for i in range(int((n_max+1)))]for i in range(int((n_max+1)))]
        #print(np.shape(Basis))
        for m1 in range(0, n_max+1):
            #m1_lengh = lengh - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
            count=0
            for n1 in range(m1,n_max+1,2):
                #print(n1,m1)
                Basis[n1][m1] = self.Radial_function(n1,m1,n_max)
                count+=1
        return Basis
    def mask(self,x,z):
        y = (x**2+z**2)
        return np.where(y<1,1,0)


    def calc_norms(self,n_max):
        Zernike_functions = self.Zernicke_embedding_generator(n_max)

        grid_extend = 1
        #grid_resolution = 680
        z = x = np.linspace(-grid_extend, grid_extend, 128)
        z, x = np.meshgrid(z, x)

        #print(Zernike_functions)
        # Use epsilon to avoid division by zero during angle calculations
        functions = [[[] for i in range(int((n_max+1)))]for i in range(int((n_max+1)))]
        #print(Zernike_functions)
        for i in range(len(Zernike_functions)):
            for j in range(len(Zernike_functions)):
                if Zernike_functions[i][j] is None:
                    functions[i][j] = numpy.polynomial.polynomial.Polynomial([0])
                    #print('None')
                else:
                    functions[i][j] = (numpy.polynomial.polynomial.Polynomial(Zernike_functions[i][j]))

        eps = np.finfo(float).eps
        out = [[[] for i in range(int((n_max+1)))]for i in range(int((n_max+1)))]
        #M = self.M_embedding_generator(n_max)
        for i in range(len(Zernike_functions)):
            for j in range(len(Zernike_functions)):
                out[i][j] = torch.tensor(np.array(functions[i][j](np.sqrt((x ** 2 + z ** 2)))*np.cos(j*np.arctan2(x , (z )))),dtype=torch.float)#,functions[i][j](np.sqrt((x ** 2 + z ** 2)))*np.sin(j*np.arctan2(x ,(z  )))])*self.mask(x,z),dtype=torch.float)

        #print(out[0])
        # Add restriction to r<1
        #out_mask = self.mask(x,z)
        #out = torch.tensor(np.array(out*out_mask),dtype=torch.float)
        norm = [[None for i in range(int((n_max+1)))]for i in range(int((n_max+1)))]
        for i in range(len(Zernike_functions)):
            for j in range(len(Zernike_functions)):
                norm[i][j] = (torch.sum(torch.abs(out[i][j]),dim= (-1,-2),keepdim = False)).item()

        #print(np.shape(norm))
        #donkey
        '''
        import matplotlib.pyplot as plt
        out = np.array(out)
        plt.figure(1)
        for i in range(1,int(26**2)):
            plt.subplot(26, 26, i)
            plt.imshow(out[(i-1)//2,(i-1)%2], origin='lower', extent=(-1, 1, -1, 1))
            plt.axis('off')

        plt.savefig('zerpic_norm_custom.png')
        plt.close()
        '''
        return norm
    def forward(self):

        return self.norm_output
