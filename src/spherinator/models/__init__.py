"""
This module is the entry point of the models package used to provide the projections.
It initializes the package and makes its modules available for import.

It contains the following modules:

1. `RotationalAutoencoder`:
    A plain convolutional autoencoder projecting on a sphere with naive rotation invariance.
2. `RotationalVariationalAutoencoder`:
    A convolutional variational autoencoder projecting on a sphere with naive rotation invariance.
3. `RotationalVariationalAutoencoderPower`:
    A convolutional variational autoencoder using power spherical distribution.
"""
from .convolutional_decoder import ConvolutionalDecoder
from .convolutional_decoder_2 import ConvolutionalDecoder2
from .convolutional_decoder_224 import ConvolutionalDecoder224
from .convolutional_decoder_256 import ConvolutionalDecoder256
from .convolutional_encoder import ConvolutionalEncoder
from .convolutional_encoder_2 import ConvolutionalEncoder2
from .rotational2_autoencoder import Rotational2Autoencoder
from .rotational2_variational_autoencoder_power import \
    Rotational2VariationalAutoencoderPower
from .rotational_variational_autoencoder_power import RotationalVariationalAutoencoderPower
from .rotational_autoencoder import RotationalAutoencoder
from .spherinator_module import SpherinatorModule
from .zernike_autoencoder import ZernikeAutoencoder
from .zernike_decoder import ZernikeDecoder
from .zernike_encoder import ZernikeEncoder
from .Zernike_layer import Lintrans3, Multilin, Zernike_layer

__all__ = [
    "ZernikeDecoder",
    "ZernikeEncoder",
    "ConvolutionalDecoder",
    "ConvolutionalDecoder2",
    "ConvolutionalDecoder224",
    "ConvolutionalDecoder256",
    "ConvolutionalEncoder",
    "ConvolutionalEncoder2",
    "Rotational2Autoencoder",
    "Rotational2VariationalAutoencoderPower",
    "RotationalAutoencoder",
    "ZernikeAutoencoder",
    "SpherinatorModule",
    "Zernike_layer",
    "Multilin",
    "Lintrans3"
]
