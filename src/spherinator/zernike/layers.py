import torch
import torch.nn as nn
import torch.nn.functional as F

# from .utils import RMSNorm          
import math
class MultiLinear(nn.Module):
    '''
    Custom linear layer, where the weights are independent for each channel and mantain rotational equivariance.
    This layer performs a linear transformation per basis index while preserving symmetry between positive and 
    negative angular orders (m and -m), which is essential for maintaining rotation-equivariant representations.
    Optionally, a custom non-linearity can be applied after the transformation.

    Args:
    	 - inp (int): number of input channels
    	 - out (int): number of output channels
    	 - size (int): number of Zernike functions or the number of terms in the expansion
    	 - non_lin (bool): defines if a custom non-linearity is applied after the linear operation. Default: False.
    '''

    def __init__(self, inp, out, size, non_lin=False):
        super().__init__()

        stdv = 1./math.sqrt(inp)

        # Learnable weights of shape (size, inp, out)
        # Each basis function has its own independent linear transformation
        # self.weight_lin = torch.nn.parameter.Parameter( torch.nn.init.normal_( torch.empty(size, inp, out) ) )  
        self.weight_lin = torch.nn.parameter.Parameter(torch.nn.init.uniform_(torch.empty(size,inp,out),a=-stdv,b= stdv))
        self.NonLin_bool = non_lin                         
        
        # Optional non-linearity layer
        if self.NonLin_bool:
            self.NonLin = NonLinearity(size=size)
    
    def forward(self,x):

        # Apply per-function linear transformation using Einstein summation
        # 'ijk' indexes weight as [size, inp, out], '...jil' indexes input, where 'i' aligns with size.
        # Essentially is performing a convolution between the channels, where the weights are independent for each channel.
        x = torch.einsum('ijk,...jil->...kil',self.weight_lin,x)

        if self.NonLin_bool:
            x = self.NonLin(x)

        return x

class  NonLinearity(nn.Module):
    '''
    Rotation-equivariant non-linearity for Zernike-based representations. 
    This module computes non-linearity on a pair of Zernike functions (Z_m^n and Z_-m^n) in a way that preserves 
    rotational equivariance. It operates on the magnitude of the functions pair (using the L2 norm) and applies a 
    learnable, smoothed gating function inspired by ReLU.
    The non-linearity is defined as:
        a(x) * x,       where a(x) is a sigmoid-activated linear projection of the L2 norm of each function pair.
    Args:
         - size (int): number of Zernike basis functions
    '''

    def __init__(self, size=16):
        super().__init__()

        self.lin = nn.Linear(size,size)   # Linear projection
        self.sigmoid = nn.Sigmoid()       # Activation
        self.Norm = nn.RMSNorm([size],elementwise_affine=False)        # Normalization

    def forward(self, x):

        square = torch.sum(x**2, dim=-1,keepdim=False)   # Compute the sum of squares along the last dimension (for each element in the batch)
        square = self.Norm(square)                       # Normalize the summed squares
        a = self.lin(square).unsqueeze(-1)               # Apply the linear layer and reshape it to have an extra dimension
        a = self.sigmoid(a)                              # To introduce a differentiable step-function

        prod = a * x                                     # Modulate input with learned gate

        return prod
