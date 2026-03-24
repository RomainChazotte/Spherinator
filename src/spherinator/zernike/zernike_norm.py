import os
import math
import numpy as np
import numpy.polynomial.polynomial
from skimage.measure import block_reduce
import torch
import torch.nn as nn


class ZernikeNorms(nn.Module):
    '''
    Numerically computes normalization constants for Zernike polynomials on a discrete image grid.

    Normalizing each (n, m) Zernike mode ensures energy consistency across filters and enables
    faithful image reconstruction from Zernike coefficients. The norms are calculated using a 
    discretized 2D integral over the unit disk.

    Args:
         - n_max (int): maximum order of Zernike polynomials. Default: 30
         - image_size (int): output image size (filters are evaluated on this resolution). Default: 14
         - numerical_expand (int): grid upsampling factor to ensure accurate integration. Default: 64
    """

    '''
    def __init__(self, n_max = 30, image_size = 14, numerical_expand=4):
        super().__init__()
        self.num = numerical_expand
        self.image_size = image_size

        # Load or compute and save the norms
        cache_path = f'.preprocessed/Zernike_norms_{image_size}_{n_max}'
        if os.path.isfile(cache_path):
            self.norm_output = torch.load(cache_path)
        else:
            self.norm_output = self._calc_norms(n_max)
            torch.save(self.norm_output, cache_path)

    def forward(self):
        ''' Returns the precomputed normalization matrix '''
        return self.norm_output

    def _calc_norms(self,n_max):
        ''' Computes the L2 norm for each (n, m) Zernike polynomial on a discretized grid '''

        eps = torch.finfo(torch.float32).eps

        # Generate zernike function
        Zernike_functions = self._get_zernike_embedding_for_norm(n_max)

        functions = [[[] for i in range(int((n_max+1)))]for i in range(int((n_max+1)))]
        for i in range(len(Zernike_functions)):
            for j in range(len(Zernike_functions)):
                if Zernike_functions[i][j] is None:
                    functions[i][j] = numpy.polynomial.polynomial.Polynomial([0])
                else:
                    functions[i][j] = (numpy.polynomial.polynomial.Polynomial(Zernike_functions[i][j]))

        # Polar coordinate grid (upscaled)
        grid_extend = 1
        z = x = np.linspace(-grid_extend, grid_extend, int(self.image_size*self.num))
        z, x = np.meshgrid(z, x)
        r = np.sqrt((x ** 2 + z ** 2))
        theta = np.arctan2(x , (z ))

        # Compute rotated Zernike basis functions    
        out = [[[] for i in range(int((n_max+1)))]for i in range(int((n_max+1)))]
        for i in range(len(Zernike_functions)):
            for j in range(len(Zernike_functions)): 
                out[i][j] = np.array( functions[i][j](r) * np.cos(j*theta) )
        out = np.array(out)
        out = torch.tensor( block_reduce(out, (1, 1, self.num, self.num), func=np.sum), dtype=torch.float ) / ( self.num**2 )
        out = torch.tensor(out)

        # Apply final disk mask
        z = x = np.linspace(-grid_extend, grid_extend, int(self.image_size))
        z, x = np.meshgrid(z, x)
        out_mask = torch.tensor( self._mask(x,z) )
        out = torch.einsum('ijkl,kl->ijkl', out, out_mask)

        # Compute L2 norm for each Zernike basis
        norm = [[None for i in range(int((n_max+1)))]for i in range(int((n_max+1)))]
        for i in range(len(Zernike_functions)):
            for j in range(len(Zernike_functions)):
                norm[i][j] = torch.sqrt( torch.sum( ( out[i][j] ) **2, dim=(-1,-2), keepdim = False ) ).item()+eps
        return norm


    def _get_zernike_embedding_for_norm(self, n_max):
        ''' Generate radial function up to n_max '''
        n_max_calc = n_max + 1
        lengh = int(((n_max_calc + 1) * n_max_calc / 2) / 2 + math.ceil(n_max_calc / 4))
        Basis = np.zeros((lengh, n_max + 1))
        Basis = [[None for i in range(int((n_max+1)))]for i in range(int((n_max+1)))]

        for m1 in range(0, n_max+1):
            count=0
            for n1 in range(m1,n_max+1,2):
                #print(n1,m1)
                Basis[n1][m1] = self._get_radial_function_for_norm(n1,m1,n_max)
                count+=1

        return Basis

    def _mask(self,x,z):
        y = (x**2+z**2)
        return np.where(y<1,1,0)

    def _get_radial_function_for_norm(self, n, m, n_max):              
        ''' Compute the radial function of Zernike polynomials '''

        factor = []

        # Ensure there are zeros for indices smaller than (n - m)
        for i in range(n_max - n):
            factor.append(0)

        # Loop over k to calculate the radial polynomial terms
        for k in range(int((n - m) / 2 + 1)):
            num_ = (-1) ** k * math.factorial(n - k)
            denom = (math.factorial(k) *
                     math.factorial((n + m) // 2 - k) *
                     math.factorial((n - m) // 2 - k))
            
            term = num_ / denom
            factor.append(term)

            # If not at the midpoint, append a zero to maintain the symmetry
            if k != int((n - m) / 2):
                factor.append(0)
                
        # Fill in zeros for higher m terms
        for i in range(m):
            factor.append(0)

        factor = np.array(factor)

        return np.flip(factor)    



    
    