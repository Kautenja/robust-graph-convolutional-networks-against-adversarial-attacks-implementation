"""
Gaussian graph convolution layer.

Reference:
    Authors: Dingyuan Zhu, Ziwei Zhang, Peng Cui, Wenwu Zhu
    Title: Robust Graph Convolutional Networks Against Adversarial Attacks
    URL: https://doi.org/10.1145/3292500.3330851

"""
from .ggcl import GaussianGraphConvolution


# explicitly define the outward facing API of this package
__all__ = [
    GaussianGraphConvolution.__name__
]
