
import math
import numpy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from numpy import convolve

class Zernike_decode(nn.Module):
    def __init__(self, n_max = 5, device = 'cpu' ):
        super().__init__()
        self.device = device
        self.Zernike_matrix = self.create_filter(n_max)
        max = torch.max(torch.abs(self.Zernike_matrix[0,0]))
        self.Zernike_matrix[0,0,0,0] = -max
        size = self.Zernike_matrix.size(0)
        out = np.array(self.Zernike_matrix)
        print(self.M_embedding_generator(5))

        max = torch.max(torch.abs(self.Zernike_matrix[0,0]))
        for i in range(2):
            for j in range(size):
                list = [0,2,4,1,3,5,2,4,3,5,4,5]
                list_m = self.M_embedding_generator(5)
                #out[:,:,j,i]

                plt.figure()
                plt.imshow(out[j,i,:,:])
                plt.axis('off')
                plt.savefig('Filter_n{}_m{}_sgn{}.png'.format(list[j],list_m[j],i))
                plt.close()
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
            out.append([functions[i](np.sqrt((x ** 2 + z ** 2)))*np.cos(M[i]*np.arctan2(x , (z ))),functions[i](np.sqrt((x ** 2 + z ** 2)))*np.sin(M[i]*np.arctan2(x ,(z  )))])
        #print(out[0])
        # Add restriction to r<1
        out_mask = self.mask(x,z)
        out = torch.tensor(np.array(out*out_mask),dtype=torch.float, device = self.device)
        norm = []
        for i in range(len(Zernike_functions)):
            norm.append([functions[i](np.sqrt((x ** 2 + z ** 2)))*np.cos(M[i]*np.arctan2(x , (z ))),functions[i](np.sqrt((x ** 2 + z ** 2)))*np.cos(M[i]*np.arctan2(x ,(z  )))])

        norm = torch.tensor(np.array(norm*out_mask),dtype=torch.float, device = self.device)

        norm = (torch.sum(torch.abs(norm),dim= (-1,-2),keepdim = True))
        return out/norm

    def forward(self,input):
        #norm = (torch.sum(torch.abs(self.Zernike_matrix),dim= (-1,-2),keepdim = True))
        #eps = 0.0000005
        #self.Zernike_matrix = self.Zernike_matrix/(norm+eps)
        #This should be implemented in init, do this later
        out = torch.einsum('ijkl,...ij->...kl',self.Zernike_matrix,input)
        return out*100
Zernike_decode()