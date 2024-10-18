import argparse
import math
import os.path
import random

import numpy
import numpy as np
#import scipy
#import scipy.special as sp
#import seaborn as sns
import torch
import torch.linalg
import torch.nn as nn
from numpy import convolve
from skimage.measure import block_reduce



class Create_multiscale_filters(nn.Module):
    def __init__(self, n_max = 4 , device = 'cpu', numerical_expand = 16, size=28, filtersize=7 ):
        super().__init__()
        self.num = numerical_expand

        if os.path.isfile('Zernike_multi_size{}_filter{}_nmax{}'.format(size,filtersize,n_max)) :
            self.Zernike_matrix = torch.load('Zernike_multi_size{}_filter{}_nmax{}'.format(size,filtersize,n_max))
        else:
            self.Zernike_matrix = self.create_macro_filter(n_max,size,filtersize)
            torch.save(self.Zernike_matrix,'Zernike_multi_size{}_filter{}_nmax{}'.format(size,filtersize,n_max))
        #size = self.calc_size(n_max)


        #self.Zernike_matrix = self.create_filter(n_max)
        self.Zernike_matrix = self.Zernike_matrix.to(device)#*16
        #self.norm_matrix = np.array(self.norm_matrix)
        #self.Zernike_matrix= torch.nn.parameter.Parameter(self.Zernike_matrix,requires_grad=False)
        #self.device = 'cuda:2'
    def calc_size(self,n_max):
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        return lengh

    def find_angles(self,size):

        z = x = np.linspace(-1, 1, size)
        z, x = np.meshgrid(z, x)
        angles = np.arctan2(x ,(z  ))
        return angles

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
        return np.where(y>1,0,1)


    def create_filter(self,n_max,filtersize,angle):

        Zernike_functions = self.Zernicke_embedding_generator(n_max)

        grid_extend = 1
        #grid_resolution = 680
        z = x = np.linspace(-grid_extend, grid_extend, int(filtersize*self.num))
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
            out.append([functions[i](np.sqrt((x ** 2 + z ** 2)))*np.cos(M[i]*(np.arctan2(x , (z ))+angle)),functions[i](np.sqrt((x ** 2 + z ** 2)))*np.sin(M[i]*(np.arctan2(x ,(z  ))+angle))])
        #print(out[0])
        # Add restriction to r<1
        #out_mask = self.mask(x,z)
        #out = torch.tensor(np.array(out*out_mask),dtype=torch.float)#, device =  'cuda:2')

        # norm = []
        # for i in range(len(Zernike_functions)):
        #     norm.append([functions[i](np.sqrt((x ** 2 + z ** 2)))*np.cos(M[i]*np.arctan2(x , (z )))+eps,functions[i](np.sqrt((x ** 2 + z ** 2)))*np.sin(M[i]*np.arctan2(x ,(z  )))+eps])

        # norm = torch.tensor(np.array(norm*out_mask),dtype=torch.float)

        # norm = torch.sqrt((torch.sum((norm)**2,dim= (-1,-2),keepdim = True)))*self.num


        # out = out/norm
        out = np.array(out)
        out =torch.tensor( block_reduce(out,(1,1, self.num, self.num),func=np.sum),dtype=torch.float)


        z = x = np.linspace(-grid_extend, grid_extend, filtersize)
        z, x = np.meshgrid(z, x)
        out_mask = torch.tensor( self.mask(x,z))
        out= torch.einsum('ijkl,kl->ijkl',out,out_mask)

        norm = torch.sqrt((torch.sum((out)**2,dim= (-1,-2),keepdim = True)))+eps
        out = out/norm
        return out#*self.num
    def calc_size(self,n_max):
        '''
        Calculating the amount of terms in the Zernike decomposition depending on n. This calculates the amount of radial polinomes, the final decomposition will have size= (calc_size(n),2)
        '''
        n_max_calc = n_max+1
        lengh = int(((n_max_calc+1)*n_max_calc/2)/2+math.ceil(n_max_calc/4))
        return lengh

    def create_macro_filter(self,n_max,size,filtersize):
        overlap =int((filtersize//2)*2)
        angles = self.find_angles(size-overlap)
        filters = torch.zeros(size-overlap,size-overlap,self.calc_size(n_max),2,size,size)
        for i in range(size-overlap):
            for j in range(size-overlap):
                filters[i,j,:,:,i:i+filtersize,j:j+filtersize] = self.create_filter(n_max,filtersize,angles[i,j])
        return filters
    def embed(self,input):

        out = torch.einsum('mnijkl,...akl->...mnaij',self.Zernike_matrix,input)

        return out

# a = torch.nn.parameter.Parameter(torch.rand(1024,3,28,28, device='cuda:0'))
# clas = Create_multiscale_filters(device='cuda:0')
# print(torch.cuda.max_memory_allocated()/1024/1024/1024)
# clas.embed(a)
# print(torch.cuda.max_memory_allocated()/1024/1024/1024)

# #print(clas.mask(x,z))
# #print(clas.embed(a))