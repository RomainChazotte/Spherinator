import math
import os.path
import numpy as np
import torch
import torch.nn as nn
from skimage.measure import block_reduce
class ZernikeEmbedding(nn.Module):

    '''
    Generates Zernike polynomial-based filters for encoding images into a 
    rotation-aware basis, and allows encoding and decoding using this basis.

    This module computes or loads a matrix of Zernike filters, rotates them
    over a specified number of orientations, and provides efficient encoding ("embed")
    and decoding ("decode") of 2D inputs via inner product (Einstein summation).

    Args:
         - size (int): spatial size of the filters. Default: 7
         - n_max (int): maximum radial order of Zernike polynomials. Default: 30
         - device (str): device on which the model will be run; e.g. 'cuda', 'cpu', or 'mps' 
         (for macOS with Metal programming framework). Default: 'cuda:2'
         - numerical_expand (int): upsampling factor used when generating filters. Default: 64
         - num_angles (int): number of orientations for rotated Zernike bases
    '''

    def __init__(self, size=7, n_max=30 , device='cuda:2', numerical_expand=4, num_angles=1):
        super().__init__()
        self.num = numerical_expand
        self.num_angles = num_angles
        self.device = device

        # Precompute Zernike or load if cached
        self.Zernike_matrix = self._load_or_create_zernike_matrix(n_max, size)
        self.Zernike_matrix = self.Zernike_matrix.to(device)

        if num_angles ==1:
            self.Zernike_matrix = self.Zernike_matrix.squeeze(0)
        self.Zernike_matrix = self.Zernike_matrix.to(device)

    def _load_or_create_zernike_matrix(self, n_max, size):
        ''' Loads a cached Zernike matrix or generates it if not found '''

        file_path = f'.preprocessed/Zernike_filters_size{size}_n_max{n_max}_expand{self.num}_num_angles{self.num_angles}' 
        
        if os.path.isfile(file_path):
            return torch.load(file_path)
        else:
            zernike_matrix = self._create_filter(n_max, size)
            torch.save(zernike_matrix, file_path)

        return zernike_matrix.to(self.device)


    def _create_filter(self, n_max, size):
        ''' Generates rotated Zernike filters on a polar grid '''

        # Generate Zernike polynomials as filters
        zernike_functions = self._get_zernike_embedding(n_max)
        functions = self._generate_zernike_functions(zernike_functions)
        M = self._generate_M_embedding(n_max)

        # Create upsampled polar coordinate grid
        grid_extend = 1
        z = x = np.linspace(-grid_extend, grid_extend, int(size * self.num))
        z, x = np.meshgrid(z, x)
        r = np.sqrt((x ** 2 + z ** 2))
        theta = np.arctan2(x , (z )) 

        angles = [j * np.pi / ( 2 * self.num_angles ) for j in range( self.num_angles )]
        out_exp = np.empty( ( self.num_angles, len(zernike_functions ), 2, int( size*self.num ), int( size*self.num ) ) )

        # Apply each Zernike function at each rotation angle
        for j in range(self.num_angles): 
            out = []
            for i in range(len(zernike_functions)):
                term1 = functions[i](r)*np.cos((M[i]*(theta)+angles[j]))
                term2 = functions[i](r)*np.sin((M[i]*(theta)+angles[j]))
                out.append([term1, term2])
            out_exp[j] = out

        # Apply circular mask and downsample
        out = np.array(out_exp)
        out_mask = self._get_mask(x, z)
        out = out_mask*out
        out = block_reduce(out, block_size=(1, 1, 1, self.num, self.num), func=np.sum)
        out = torch.tensor(out, dtype=torch.float32)

        # Final masking at target resolution
        z = x = np.linspace(-grid_extend, grid_extend, int(size))
        z, x = np.meshgrid(z, x)
        out_mask = torch.tensor(self._get_mask(x, z))
        out = torch.einsum('aijkl,kl->aijkl', out, out_mask)

        # Normalize
        eps = torch.finfo(torch.float32).eps 
        norm = torch.linalg.vector_norm(out, dim=(-1, -2), keepdim=True) + eps
        out = out / norm
    
        return out
    
    def _get_zernike_embedding(self, n_max):
        ''' Generate radial function up to n_max '''
        n_max_calc = n_max + 1
        lengh = int(((n_max_calc + 1) * n_max_calc / 2) / 2 + math.ceil(n_max_calc / 4))
        Basis = np.zeros((lengh, n_max + 1))

        for m1 in range(0, n_max + 1):
            m1_lengh = lengh - int(((n_max_calc - m1 + 1) * (n_max_calc - m1) / 2) / 2 + math.ceil((n_max_calc - m1) / 4))
            count = 0
            for n1 in range(m1, n_max + 1, 2):
                Basis[m1_lengh + count, :] = self._get_radial_function(n1, m1, n_max)
                count += 1
        return Basis

    def _generate_zernike_functions(self, zernike_embedding):
        ''' Create callable polynomial objects from Zernike radial embeddings '''

        return [np.polynomial.polynomial.Polynomial(emb) for emb in zernike_embedding]


    def _get_radial_function(self,n, m, n_max):
        """Compute the radial function of Zernike polynomials."""

        factor = []
        scaling = []
        
        # Create scaling factors based on n_max
        for i in range(n_max + n_max + 1):
            scaling.append(1 / ((2 * n_max - i) ** 2 + 2))

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

        scale = np.convolve(factor, factor)
        scale = np.einsum('i,i', scaling, scale)
 
        factor = np.array(factor / scale)
        
        return np.flip(factor)

    def _generate_M_embedding(self, n_max):
        ''' Generates array of angular frequencies m for each Zernike function '''
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

    def _get_mask(self, x, z):
        ''' Returns a binary mask for points inside the unit circle '''
        y = (x**2+z**2)
        return np.where(y<1,1,0)

    def embed(self, x):
        ''' Projects input onto the Zernike basis '''
        return torch.einsum('ijkl,...kl->...ij',self.Zernike_matrix, x) 

    def decode(self, x):
        ''' Reconstructs an image from Zernike basis coefficients ''' 
        return torch.einsum('ijkl,...ij->...kl',self.Zernike_matrix, x) 




