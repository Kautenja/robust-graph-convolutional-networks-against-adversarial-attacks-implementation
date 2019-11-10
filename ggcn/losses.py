"""
losses for Gaussian graph convolution neural networks.

Reference:
    Authors: Dingyuan Zhu, Ziwei Zhang, Peng Cui, Wenwu Zhu
    Title: Robust Graph Convolutional Networks Against Adversarial Attacks
    URL: https://doi.org/10.1145/3292500.3330851

"""
from keras import backend as K
from tensorflow import distributions


def kl_reg(mean, variance):
    """
    Return the kl_regularization based on mean and variance tensors.

    Args:
        mean: the mean output from the first layer
        variance: the variance output from the first layer

    Returns:
        the KL-divergence between normal distribution and model distribution
    """
    identity = distributions.Normal(K.zeros_like(mean), K.ones_like(mean))
    model = distributions.Normal(mean, variance)
    return distributions.kl_divergence(identity, model)


# explicitly define the outward facing API of this module
__all__ = [kl_reg.__name__]
