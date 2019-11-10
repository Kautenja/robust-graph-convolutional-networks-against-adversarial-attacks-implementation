"""
Gaussian graph convolution layer.

Reference:
    Authors: Dingyuan Zhu, Ziwei Zhang, Peng Cui, Wenwu Zhu
    Title: Robust Graph Convolutional Networks Against Adversarial Attacks
    URL: https://doi.org/10.1145/3292500.3330851

"""
from keras.engine.topology import Layer
from keras import activations, initializers, regularizers, constraints
import keras.backend as K
from tensorflow import distributions


def sample(mean, variance):
    """
    Sample from the given normal distribution.

    Args:
        mean: the mean values
        variance: the variances

    Returns:
        values samples from the distribution

    """
    epsilon = distributions.Normal(K.zeros_like(mean), K.ones_like(mean)).sample()
    return mean + epsilon * variance


class GaussianGraphConvolution(Layer):
    """
    Gaussian graph convolution layer.

    Reference:
        Authors: Dingyuan Zhu, Ziwei Zhang, Peng Cui, Wenwu Zhu
        Title: Robust Graph Convolutional Networks Against Adversarial Attacks
        URL: https://doi.org/10.1145/3292500.3330851

    """

    def __init__(self, units: int,
        is_first: bool = False,
        is_last: bool = False,
        attention_factor: float = 1,
        activation: any = None,
        mean_initializer: any = 'glorot_uniform',
        variance_initializer: any = 'glorot_uniform',
        **kwargs
    ):
        """
        TODO.

        Args:
            units: the number of weights
            is_first: whether this is the first Gaussian graph convolution layer
            is_last: whether this is the last Gaussian graph convolution layer
            attention_factor: the attention factor ([0, 1], 1 is best)
            TODO

        Returns:
            None

        """
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GaussianGraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.is_first = is_first
        self.is_last = is_last
        self.attention_factor = attention_factor
        self.activation = activations.get(activation)
        self.mean_initializer = initializers.get(mean_initializer)
        self.variance_initializer = initializers.get(variance_initializer)
        self.supports_masking = True
        # setup model variables
        self.mean_weight = None
        self.variance_weight = None

    def compute_output_shape(self, input_shape):
        """
        Return the output shape of the layer for given input shape.

        Args:
            input_shape: the input shape to transform to output shape

        Returns:
            the output shape as a function of input shape

        """
        features_shape = input_shape[0]
        output_shape = (features_shape[0], self.units)
        if self.is_last:  # the last layer samples from the distribution
            return output_shape
        # normal layers return a mean and variance tensor
        return [output_shape, output_shape]

    def build(self, input_shape):
        """
        Build the layer for the given input shape.

        Args:
            input_shape: the shape to build the layer with

        Returns:
            None

        """
        features_shape = input_shape[0]
        assert len(features_shape) == 2
        input_dim = features_shape[1]

        self.mean_weight = self.add_weight(
            shape=(input_dim, self.units),
            name='mean',
            initializer=self.mean_initializer)
        self.variance_weight = self.add_weight(
            shape=(input_dim, self.units),
            name='variance',
            initializer=self.variance_initializer)

        self.built = True

    def _call_first(self, inputs):
        """
        Forward pass through the layer (assuming the inputs are vectors.

        Args:
            inputs: the input tensors to pass through the layer

        Returns:
            the output tensors from the layer

        """
        features, basis = inputs
        # calculate the mean and variance
        output = K.dot(basis, features)
        mean = K.dot(output, self.mean_weight)
        variance = K.dot(output, self.variance_weight)
        # sample from the distribution if the last layer
        if self.is_last:
            return self.activation(sample(mean, variance))
        # return the mean and variance through the activation
        return [self.activation(mean), self.activation(variance)]

    def _call_generic(self, inputs):
        """
        Forward pass through the layer for generic cases.

        Args:
            inputs: the input tensors to pass through the layer

        Returns:
            the output tensors from the layer

        """
        mean, variance, basis = inputs
        # calculate the alpha value
        alpha = K.exp(-self.attention_factor * variance)
        # calculate the mean
        mean = K.dot(basis, (mean * alpha))
        mean = K.dot(mean, self.mean_weight)
        # calculate the variance
        variance = K.dot(basis, (variance * alpha**2))
        variance = K.dot(variance, self.variance_weight)
        # sample from the distribution if the last layer
        if self.is_last:
            return self.activation(sample(mean, variance))
        # return the mean and variance through the activation
        return [self.activation(mean), self.activation(variance)]

    def call(self, inputs, **kwargs):
        """
        Forward pass through the layer.

        Args:
            inputs: the input tensors to pass through the layer

        Returns:
            the output tensors from the layer

        """
        if self.is_first:
            return self._call_first(inputs)
        return self._call_generic(inputs)

    def get_config(self):
        """Return the configuration for building the layer."""
        config = dict(
            units=self.units,
            activation=activations.serialize(self.activation),
            mean_initializer=initializers.serialize(self.mean_initializer),
            variance_initializer=initializers.serialize(self.variance_initializer),
        )

        base_config = super(GaussianGraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# explicitly define the outward facing API of this module
__all__ = [GaussianGraphConvolution.__name__]
