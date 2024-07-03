
from scipy.constants import physical_constants
import scipy.special as sp
import seaborn as sns
import numpy as np
import argparse

import numpy
import numpy as np
from numpy import convolve
import scipy
import math
import torch
import torch.nn as nn

import torch.nn.functional as F

import random

class Zernike_embedding(nn.Module):
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
                out[i][j] = torch.tensor(np.array([functions[i][j](np.sqrt((x ** 2 + z ** 2)))*np.cos(j*np.arctan2(x , (z ))),functions[i][j](np.sqrt((x ** 2 + z ** 2)))*np.sin(j*np.arctan2(x ,(z  )))])*self.mask(x,z),dtype=torch.float)

        #print(out[0])
        # Add restriction to r<1
        #out_mask = self.mask(x,z)
        #out = torch.tensor(np.array(out*out_mask),dtype=torch.float)
        norm = [[[] for i in range(int((n_max+1)))]for i in range(int((n_max+1)))]
        for i in range(len(Zernike_functions)):
            for j in range(len(Zernike_functions)):
                norm[i][j] = (torch.sum(torch.abs(out[i][j]),dim= (-1,-2),keepdim = False))#.item()
                if norm[i][j][1] == 0:
                    norm[i][j][1] = norm[i][j][0]
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


x = Zernike_embedding(50)
x = x()
print(x)
'''
for i in range(676):
    if x[i,1].item() != 0:
        print(x[i,0].item(),x[i,1].item())
    else:
        print(x[i,0].item(),x[i,0].item())
'''