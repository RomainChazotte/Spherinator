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
    def __init__(self, n_max = 8 , device = 'cpu', numerical_expand = 16, size=28, filtersize=7, spacing = 3 ):
        super().__init__()
        self.num = numerical_expand
        #angles = self.find_angles(5)
        #print(angles/np.pi)

        self.Zernike_matrix = self.create_macro_filter_pad(n_max,size,filtersize,spacing)

        # if os.path.isfile('Zernike_multi_size{}_filter{}_nmax{}_spacing{}_other_rot'.format(size,filtersize,n_max,spacing)) :
        #     self.Zernike_matrix = torch.load('Zernike_multi_size{}_filter{}_nmax{}_spacing{}_other_rot'.format(size,filtersize,n_max,spacing))
        # else:
        #     self.Zernike_matrix = self.create_macro_filter(n_max,size,filtersize,spacing)
        #     torch.save(self.Zernike_matrix,'Zernike_multi_size{}_filter{}_nmax{}_spacing{}_other_rot'.format(size,filtersize,n_max,spacing))


        #size = self.calc_size(n_max)
        # print('hi_start')
        # print(self.create_filter(5,3,0.5*np.pi))
        # print('hi')
        # print(self.create_filter(5,3,0))
        # print('hi_stop')


        #self.Zernike_matrix = self.create_filter(n_max)

        self.Zernike_matrix = self.Zernike_matrix.to(device)#*16
        #self.norm_matrix = np.array(self.norm_matrix)
        #self.Zernike_matrix= torch.nn.parameter.Parameter(self.Zernike_matrix,requires_grad=False)
        #self.device = 'cuda:2'

    def find_angles(self,size):
        eps = np.finfo(float).eps
        z = x = np.linspace(-1, 1, size)
        #print(np.meshgrid(z, x))
        z, x = np.meshgrid(z, x)
        angles = np.arctan2(x  ,(z))#+ np.pi
        #print(angles)
        #donkey
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
            out.append([functions[i](np.sqrt((x ** 2 + z ** 2)))*np.cos(M[i]*(np.arctan2(x , (z ))-angle)),functions[i](np.sqrt((x ** 2 + z ** 2)))*np.sin(M[i]*(np.arctan2(x ,(z  ))-angle))])

        out = np.array(out)
        out =torch.tensor( block_reduce(out,(1,1, self.num, self.num),func=np.sum),dtype=torch.float)


        z = x = np.linspace(-grid_extend, grid_extend, filtersize)
        z, x = np.meshgrid(z, x)
        out_mask = torch.tensor( self.mask(x,z))
        out= torch.einsum('ijkl,kl->ijkl',out,out_mask)
        out = torch.tensor(np.where(np.abs(out)<1e-6,0.,out))

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

    # def find_scaling(self,size,filtersize,x,y):
    #     reference = (filtersize-1)//2
    #     x = min(x,size-x)
    #     y = min(x,size-y)
    #     if x <
    def find_scaling(self,size,filtersize):
        max = size-1
        reference = ((filtersize-1)//2)+1
        filter = filtersize**2

        out = np.zeros((size,size))
        for x_in in range(size):
            for y_in in range(size):
                x = min(x_in,max-x_in)
                y = min(y_in,max-y_in)
                if x < reference:
                    if y < reference:
                        param = (filtersize+x-reference+1)*(filtersize+y-reference+1)
                    else:
                        param = (filtersize+x-reference+1)*(filtersize)
                elif y < reference:
                    param = (filtersize)*(filtersize+y-reference+1)
                else:
                    param = filter
                out[x_in,y_in] = filter/param

        return out
    def create_macro_filter(self,n_max,size,filtersize,spacing):
        overlap =int((filtersize//2)*2)#-2
        out_size = int((size-overlap)/spacing)#-1#+1

        angles = self.find_angles(out_size)
        #print(angles/np.pi)
        filters = torch.zeros(out_size,out_size,self.calc_size(n_max),2,size,size)
        for i in range(out_size):
            for j in range(out_size):
                #print(self.create_filter(n_max,filtersize,angles[i,j]).size())
                filters[i,j,:,:,spacing*i:spacing*i+filtersize,spacing*j:spacing*j+filtersize] = self.create_filter(n_max,filtersize,angles[i,j])
        return filters

    def create_macro_filter_pad(self,n_max,size,filtersize,spacing):
        overlap =int((filtersize//2)*2)#-2
        out_size = int((size-overlap)/spacing)#-1#+1

        angles = self.find_angles(out_size)
        #print(angles/np.pi)
        filters = torch.zeros(out_size,out_size,self.calc_size(n_max),2,size,size)
        scale = self.find_scaling(size-overlap,filtersize)
        for i in range(out_size):
            for j in range(out_size):
                #print(self.create_filter(n_max,filtersize,angles[i,j]).size())
                filters[i,j,:,:,spacing*i:spacing*i+filtersize,spacing*j:spacing*j+filtersize] = self.create_filter(n_max,filtersize,angles[i,j])*scale[spacing*i,spacing*j]

        filters = filters[:,:,:,:,int(overlap/2):int(size-overlap/2),int(overlap/2):int(size-overlap/2)]
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