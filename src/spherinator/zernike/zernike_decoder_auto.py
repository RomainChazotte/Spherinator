import math

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .zernike_layer import ZernikeLayer
# from .utils import RMSNorm

from .layers import MultiLinear

class ZernikeDecoderAuto(pl.LightningModule):
    '''
    A classification model using Zernike-based layers to encode structured image data.

    This module applies a sequence of equivariant Zernike layers, followed by normalization
    and fully connected layers, to produce a classification output.

    Args:
         - n_in (int): maximum Zernike order for the first layer
         - n_output (int): maximum Zernike order for the second layer
         - in_channels (int): number of input channels in the input tensor
         - num_channels (int): number of intermediate/output channels for Zernike layers
         - device (str): device on which the model will be run; e.g. 'cuda', 'cpu', or 'mps'
         (for macOS with Metal programming framework).
         - image_size (int): size of the input image (assumed square). Default: 14
         - p_dropout (float): dropout probability. Default: 0.5
         - num_classes (int): number of classes for classification task. Default: 10
    '''
    def __init__(self, n_in, n_output, in_channels, num_channels, device,Very_small=True,Very_large=True, image_size=14, p_dropout=0.5, num_classes=10):
        super().__init__()

        test_dimens = False
        self.Very_small = Very_small
        self.Very_large = Very_large
        vast_weights = False
        Skip_connect=True

        # First Zernike layer
        n_temp = int(n_in/2)

        if self.Very_large:
            num_channels = int(num_channels*4)
            self.Product00 = ZernikeLayer(n_max=n_in, n_out=n_in, multichanneled='independant',
                                        in_channels=in_channels, intermediate_channels=int(num_channels/8),
                                        out_channels=int(num_channels/8), image_size=image_size,
                                        fast_test_dimensionality=test_dimens,invert = True,vast_weights=vast_weights,Skip_connect=Skip_connect, device=device)
            size = self.Product00._calc_size(n_in)
            self.Rnorm00 = nn.RMSNorm([size,2], elementwise_affine=False)
            self.SkipLinear_0 = MultiLinear(int(num_channels/8), int(num_channels/8), size, non_lin=True)
            in_channels = int(num_channels/8)


        self.Product01 = ZernikeLayer(n_max=n_in, n_out=n_temp, multichanneled='independant',
                                      in_channels=in_channels, intermediate_channels=int(num_channels/8),
                                      out_channels=int(num_channels/8), image_size=image_size,
                                      fast_test_dimensionality=test_dimens,invert = True,vast_weights=vast_weights,Skip_connect=Skip_connect, device=device)

        self.Product02 = ZernikeLayer(n_max=n_temp, n_out=int(n_temp/2), multichanneled='independant',
                                      in_channels=int(num_channels/8), intermediate_channels=int(num_channels/4),
                                      out_channels=int(num_channels/4), image_size=image_size,
                                      fast_test_dimensionality=test_dimens,invert = True,vast_weights=vast_weights,Skip_connect=Skip_connect, device=device)

        self.Product03 = ZernikeLayer(n_max=int(n_temp/2), n_out=int(n_temp/4), multichanneled='independant',
                                      in_channels=int(num_channels/4), intermediate_channels=int(num_channels/2),
                                      out_channels=int(num_channels/2), image_size=image_size,
                                      fast_test_dimensionality=test_dimens,invert = True,vast_weights=vast_weights,Skip_connect=Skip_connect, device=device)

        self.Product04 = ZernikeLayer(n_max=int(n_temp/4), n_out=int(n_temp/8), multichanneled='independant',
                                      in_channels=int(num_channels/2), intermediate_channels=num_channels,
                                      out_channels=num_channels, image_size=image_size,
                                      fast_test_dimensionality=test_dimens,invert = True,vast_weights=vast_weights,Skip_connect=Skip_connect, device=device)


        self.Product05 = ZernikeLayer(n_max=int(n_temp/8), n_out=int(n_temp/16), multichanneled='independant',
                                      in_channels=num_channels, intermediate_channels=num_channels,
                                      out_channels=num_channels, image_size=image_size,
                                      fast_test_dimensionality=test_dimens,invert = True,vast_weights=vast_weights,Skip_connect=Skip_connect, device=device)
        # Normalization 1
        size = self.Product01._calc_size(n_temp)
        self.Rnorm01 = nn.RMSNorm([size,2], elementwise_affine=False)

        # self.SkipLinear_3 = MultiLinear(int(num_channels/4), int(num_channels/8), size, non_lin=True)

        # Normalization 2
        size = self.Product01._calc_size(int(n_temp/2))
        self.Rnorm02 = nn.RMSNorm([size,2], elementwise_affine=False)

        # self.SkipLinear_2 = MultiLinear(int(num_channels/2), int(num_channels/4), size, non_lin=True)
        size = self.Product01._calc_size(int(n_temp/4))
        self.Rnorm03 = nn.RMSNorm([size,2], elementwise_affine=False)

        # self.SkipLinear_1 = MultiLinear(num_channels, int(num_channels/2), size, non_lin=True)
        size = self.Product01._calc_size(int(n_temp/8))

        self.Rnorm04 = nn.RMSNorm([size,2], elementwise_affine=False)

        if self.Very_small:
            # self.SkipLinear_5 = MultiLinear(num_channels, int(num_channels), size, non_lin=True)
            size = self.Product01._calc_size(int(n_temp/16))
            self.Rnorm05 = nn.RMSNorm([size,2], elementwise_affine=False)


        self.MultiLinear_2 = MultiLinear(int(num_channels), int(num_channels), size, non_lin=True)

        if self.Very_large:
            self.MultiLinear_0 = MultiLinear(1, int(num_channels/16), size, non_lin=True)
            self.MultiLinear_1 = MultiLinear(int(num_channels/16), int(num_channels), size, non_lin=True)
        else:
            self.MultiLinear_1 = MultiLinear(1, int(num_channels), size, non_lin=True)

        # Regularization
        # self.dropout = nn.Dropout(p=p_dropout)
        self.inv_fc_location = nn.Linear(3,2)

    def forward(self, x, angle) -> torch.tensor:
        ''' Forward pass for classification '''

        x = self.inv_fc_location(x)
        x = self.zernify(x, angle)


        if self.Very_large:
            x = self.MultiLinear_0(x)
        x = self.MultiLinear_1(x)
        x = self.MultiLinear_2(x)
        # First Zernike transformation
        if self.Very_small:
            x = self.Rnorm05(x)
            x = self.Product05(x,x)#+self.SkipLinear_5(torch.einsum('ij,...jk->...ik', self.Product05.in_mask_1,x))
        x = self.Rnorm04(x)
        # x = self.Product04(x,x)+self.SkipLinear_1(torch.einsum('ij,...jk->...ik', self.Product04.in_mask_1,x))
        x = self.Product04(x,x)#+self.SkipLinear_1(torch.einsum('ij,...jk->...ik', self.Product04.in_mask_1,x))
        x = self.Rnorm03(x)

        # Second Zernike transformation
        x = self.Product03(x,x)#+self.SkipLinear_2(torch.einsum('ij,...jk->...ik', self.Product03.in_mask_1,x))
        x = self.Rnorm02(x)

        x = self.Product02(x,x)#+self.SkipLinear_3(torch.einsum('ij,...jk->...ik', self.Product02.in_mask_1,x))
        x = self.Rnorm01(x)

        y = self.Product01(x,x)


        if self.Very_large:
            # y += self.SkipLinear_0(torch.einsum('ij,...jk->...ik', self.Product01.in_mask_1,x))
            y = self.Rnorm00(y)
            y = self.Product00(y,y)
        return y

    def zernify(self, z, angle):
        z = z.unsqueeze(-1)
        z = z*angle
        return z