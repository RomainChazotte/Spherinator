import numpy
import numpy as np
from numpy import convolve
import scipy
import math
import torch
import torch.nn as nn

import torch.nn.functional as F


# Maybe implement normalization (Should be trivial)
class  Non_linearity(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1,1)

    def forward(self,x):
        a = torch.sum(x**2, dim=-1,keepdim=True)
        print(a.size())
        print(x.size())
        a = (torch.tanh(a)+1)/2
        return a * x
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
        # try torch.nn.init.normal_  and torch.nn.init.kaiming_uniform_
        self.weight = torch.nn.parameter.Parameter(torch.ones(size,inp,out)/np.sqrt(inp))
        self.bias = torch.nn.parameter.Parameter(torch.ones(size,1))
        self.Non_lin_bool = Non_lin
        #print(size,inp,out)
        if Non_lin:
            self.Non_lin = Non_linearity()
    def forward(self,x):
        x = torch.einsum('ijk,...jil->...kil',self.weight,x)
        x = x+ self.bias
        if self.Non_lin_bool:
            x = self.Non_lin(x)
        return x


class Zernike_layer(nn.Module):
    def __init__(self, n_max = 30, n_out=30, multichanneled = False,in_channels = 1 ,intermediate_channels=1, out_channels =1 ):
        super().__init__()
        self.increase = False
        if n_max < n_out:
            batch_size=1
            self.increase = True
            self.in_mask = self.create_mask_increase(n_max,n_out)
            out_size= self.calc_size(n_out)
            self.input1_expand = torch.zeros(batch_size,out_size,2)
            self.input2_expand = torch.zeros(1,out_size,2)
            self.Zernike_matrix = torch.tensor(self.Zernicke_matrix_generator(n_out),dtype=torch.float)
            size = self.calc_size(n_out)
        else:
            self.Zernike_matrix = torch.tensor(self.Zernicke_matrix_generator(n_max),dtype=torch.float)
            size = self.calc_size(n_max)
        Matrix_plus = [[[1/2,0],[0,-1/2]],[[0,1/2],[1/2,0]]]
        Matrix_minus_pos =[[[1/2,0],[0,1/2]],[[0,1/2],[-1/2,0]]]
        Matrix_minus_neg =[[[1/2,0],[0,1/2]],[[0,-1/2],[1/2,0]]]
        Matrix_minus_neut =[[[1/2,0],[0,1/2]],[[0,0],[0,0]]]
        self.transform = torch.tensor(np.array([Matrix_plus,Matrix_minus_pos,Matrix_minus_neut,Matrix_minus_neg]),dtype=torch.float)
        self.Nonlin = Non_linearity()
        self.weight = torch.nn.parameter.Parameter(torch.nn.init.normal_(torch.empty(size,size))/size)
        self.reduce = False
        if n_max > n_out:
            self.reduce = True
            self.out_mask = self.create_mask_decrease(n_max,n_out)

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
        mask = torch.ones(lengh)
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
        lengh = int(((n_out_calc+1)*n_out_calc/2)/2+math.ceil(n_out_calc/4))
        mask = torch.ones(lengh)
        for m1 in range(0, n_out+1):
            m1_lengh = lengh - int(((n_out_calc-m1+1)*(n_out_calc-m1)/2)/2+math.ceil((n_out_calc-m1)/4))
            count=0
            for n1 in range(m1,n_out+1,2):
                if m1>n_max or n1>n_max:
                    mask[m1_lengh+count] -= 1
                count+=1
        mask = mask.bool()
        return mask
    def Radial_function(self,n,m, n_max):
        faktor = []
        scaling = []
        for i in range(n_max+n_max+1):
            scaling.append(1/((2*n_max-i)**2+2))

        for i in range(n_max-n):
            faktor.append(0)

        for k in range(int((n-m)/2+1)):
            faktor.append((-1)**k * math.factorial(n-k) /(math.factorial(k) * math.factorial(int((n+m)/2-k))* math.factorial(int((n-m)/2-k)))   )
            if k != int((n-m)/2):
                faktor.append(0)
            #exp.append(n-2*k)

        for i in range(m):
            faktor.append(0)
        scale = convolve(faktor,faktor)
        scale = np.einsum('i,i', scaling,scale)

        faktor = np.array(faktor/scale)
        faktor = np.array(faktor)
        return np.flip(faktor)




    def Radial_function_matrix(self,m, n_max):
        scaling = []
        matrix = None
        matrix = []
        empty = np.zeros(n_max+1)
        for i in range(m):
            empty *=0
            empty[n_max-i] = 1
            ##print('hi',empty)
            matrix.append(empty.copy())
        ##print(matrix)
        for i in range(n_max+n_max+1):
            scaling.append(1/((2*n_max-i)**2+2))
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
            scale = convolve(faktor,faktor)
            scale = np.einsum('i,i', scaling,scale)

            faktor = faktor/scale

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
    def forward(self,in1,in2):
        if self.increase:
            in1 = self.input1_expand[self.in_mask] = in1
            in2 = self.input2_expand[self.in_mask] = in2
        if self.multichanneled:
            in1 = self.In_Lin1(in1)
            in2 = self.In_Lin2(in2)

        out = torch.einsum('...im,ijkl,ij,...jn->...klmn', in1,self.Zernike_matrix,self.weight,in2)
        # Do not put anything inbetween, as it might break equivariance
        out = self.Nonlin(torch.einsum('lamn,...klmn->...ka', self.transform,out))
        print(out[0,0:30,0])
        print('start')
        #print(out[0,100:150,0])
        #print('middle')
        #print(out[0,-50:-1,0])
        #print('stop')
        if self.multichanneled:
            out = self.Out_Lin(out)
        if self.reduce:
            out = out[:,self.out_mask,:]
        print(out.size())
        return out


class Zernike_embedding(nn.Module):
    def __init__(self, n_max = 30 ):
        super().__init__()
        self.Zernike_matrix = torch.tensor(np.array(self.create_filter(n_max)),dtype=torch.float)
        size = self.calc_size(n_max)

    def calc_size(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        return lengh

    def M_embedding_generator(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        Basis = np.zeros(lengh)
        for m1 in range(0, n_max+1):
            m1_lengh = lengh - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
            count=0
            for n1 in range(m1,n_max+1,2):
                Basis[m1_lengh+count] = m1
                count+=1
        return Basis


    def Radial_function(self,n,m, n_max):
        faktor = []
        scaling = []
        for i in range(n_max+n_max+1):
            scaling.append(1/((2*n_max-i)**2+2))

        for i in range(n_max-n):
            faktor.append(0)

        for k in range(int((n-m)/2+1)):
            faktor.append((-1)**k * math.factorial(n-k) /(math.factorial(k) * math.factorial(int((n+m)/2-k))* math.factorial(int((n-m)/2-k)))   )
            if k != int((n-m)/2):
                faktor.append(0)
            #exp.append(n-2*k)

        for i in range(m):
            faktor.append(0)
        scale = convolve(faktor,faktor)
        scale = np.einsum('i,i', scaling,scale)

        faktor = np.array(faktor/scale)
        #faktor = np.array(faktor)
        return np.flip(faktor)

    def Zernicke_embedding_generator(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        Basis = np.zeros((lengh,n_max+1))
        for m1 in range(0, n_max+1):
            m1_lengh = lengh - int(((n_max_calc-m1+1)*(n_max_calc-m1)/2)/2+math.ceil((n_max_calc-m1)/4))
            count=0
            for n1 in range(m1,n_max+1,2):
                Basis[m1_lengh+count,:] = self.Radial_function(n1,m1,n_max)
                count+=1
        return Basis
    def mask(self,x,z):
        y = (x**2+z**2)
        return np.where(y<1,1,0)


    def create_filter(self,n_max):
        Zernike_functions = self.Zernicke_embedding_generator(n_max)

        grid_extend = 1
        #grid_resolution = 680
        z = x = np.linspace(-grid_extend, grid_extend, 128)
        z, x = np.meshgrid(z, x)

        #print(Zernike_functions)
        # Use epsilon to avoid division by zero during angle calculations
        functions = []
        for i in range(len(Zernike_functions)):

            functions.append(numpy.polynomial.polynomial.Polynomial(Zernike_functions[i]))

        eps = np.finfo(float).eps
        out = []
        M = self.M_embedding_generator(n_max)
        for i in range(len(Zernike_functions)):
            out.append([functions[i](np.sqrt((x ** 2 + z ** 2)))*np.cos(M[i]*np.arctan2(x , (z + eps))),functions[i](np.sqrt((x ** 2 + z ** 2)))*np.sin(M[i]*np.arctan2(x ,(z  + eps)))])
        #print(out[0])
        # Add restriction to r<1
        out_mask = self.mask(x,z)
        return out*out_mask

    def forward(self,input):
        out = torch.einsum('ijkl,...kl->...ij',self.Zernike_matrix,input)
        return out
input = torch.ones(4,5,128,128)
x = Zernike_embedding(2)
#print(x(input))
dacekl
x = Zernike_layer( n_max = 20,n_out=15,multichanneled = 'independant',in_channels = 1 ,intermediate_channels=10, out_channels =1 )
one = torch.zeros(1,121,2,dtype=torch.float)
two = torch.zeros(1,121,2,dtype=torch.float)
for i in range(20):
    one[0,i,0] +=1
for j in range(20):
    two[0,16+j,0] +=1
y = (x(one/20,two/20))
print(torch.sum(y))




#print(signtransform(2,1,[1,0],[1,0]))

# Implement n**3 Matrix
# Copy it 2*2*8 times
#perform product independantly for each combination of 2 and 2
# upper triangle goes to 0-3, lower one to 4-7
# reduce back down via sign transform function
# Product is of dimension n x n -> n. Therefore, we have n**2 independant weights. If we eant the model to be invariant towards permutation of input1 and input2, we have (n**2)/2 +n/2 independant features, yet have to properly implement this to autograd. Luckily, this is for the moment not necessary.
# (n x 2) * (n x 2) -> (n x 8)
# (n x 8) * (8 x 2) -> (n x 2)

#Matrix_lower =

#### copy both inputs to have (n x 4) inputs, with one being ordered 1,1,2,2 and the other one being ordered 1,2,1,2
# Do (4 x n)* n**3 x 3 x 2 * (n x 4) product -> n x 4 x 3 x 2
# 2 being n+m or n-m and 3 being sgn(n-m). This is reducible to dim 4 as m+n term is independant of sgn(n-m).
# Multiply with a 4 x 3 x 2 x 2 matrix to collapse it down to feature space
# Optionally multiply it with  4 x 3 x 2 x 4 matrix instead to keep it in computation space


# For multiplicities of the same input Have the n**3 matrix be n**3 x m x m to allow free propagation between Nodes
# Paramater space size = num_layers* n**2  * m **2
# Base storage usage ~n**3
