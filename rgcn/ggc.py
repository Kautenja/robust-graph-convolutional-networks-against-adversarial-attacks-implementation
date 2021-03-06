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
        attention_factor: float = 1.,
        dropout: float = 0.,
        mean_activation: any = 'elu',
        mean_initializer: any = 'glorot_uniform',
        mean_regularizer: any = regularizers.l2(5e-4),
        mean_constraint: any = None,
        variance_activation: any = 'relu',
        variance_initializer: any = 'glorot_uniform',
        variance_regularizer: any = regularizers.l2(5e-4),
        variance_constraint: any = constraints.NonNeg(),
        last_activation: any = None,
        **kwargs
    ):
        """
        Create a new Gaussian graph convolution layer.

        Args:
            units: the number of units in the layer
            is_first: whether this is the first Gaussian graph convolution layer.
                      If true, the inputs are just the features and graph; if
                      false, the inputs are mean, variance, and graph
            is_last: whether this is the last Gaussian graph convolution layer.
                     If true, the output is sampled from the mean and variance;
                     if false, the outputs are the mean and variance.
            attention_factor: the attention factor ([0, 1], 1 is best)
            dropout: the dropout rate to apply to the mean and variance output
            mean_activation: the activation function for the mean outputs
            mean_initializer: the initializer for the mean weights
            mean_regularizer: the regularize for the mean weights
            mean_constraint: the constraint mechanism for the mean weights
            variance_activation: the activation function for the variance outputs
            variance_initializer: the initializer for the variance weights
            variance_regularizer: the regularize for the variance weights
            variance_constraint: the constraint mechanism for the variance weights
            last_activation: the activation function to apply if is_last is true

        Returns:
            None

        """
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = units
        self.is_first = is_first
        self.is_last = is_last
        self.attention_factor = attention_factor
        self.dropout = dropout
        self.mean_activation = activations.get(mean_activation)
        self.mean_initializer = initializers.get(mean_initializer)
        self.mean_regularizer = regularizers.get(mean_regularizer)
        self.mean_constraint = constraints.get(mean_constraint)
        self.variance_activation = activations.get(variance_activation)
        self.variance_initializer = initializers.get(variance_initializer)
        self.variance_regularizer = regularizers.get(variance_regularizer)
        self.variance_constraint = constraints.get(variance_constraint)
        self.last_activation = activations.get(last_activation)
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
        # setup dimensionality
        features_shape = input_shape[0]
        assert len(features_shape) == 2
        input_dim = features_shape[1]
        # create a weight for the mean values
        self.mean_weight = self.add_weight(
            shape=(input_dim, self.units),
            name='mean_weight',
            initializer=self.mean_initializer,
            regularizer=self.mean_regularizer,
            constraint=self.mean_constraint)
        # create a weight for the variance values
        self.variance_weight = self.add_weight(
            shape=(input_dim, self.units),
            name='variance_weight',
            initializer=self.variance_initializer,
            regularizer=self.variance_regularizer,
            constraint=self.variance_constraint)
        # mark the layer as built
        super().build(input_shape)

    def _call_first(self, inputs):
        """
        Forward pass through the layer (assuming the inputs are vectors.

        Args:
            inputs: the input tensors to pass through the layer

        Returns:
            the output tensors from the layer

        """
        features, graph = inputs
        # calculate the mean and variance
        output = K.dot(graph, features)
        mean = K.dot(output, self.mean_weight)
        variance = K.dot(output, self.variance_weight)
        return mean, variance

    def _call_generic(self, inputs):
        """
        Forward pass through the layer for generic cases.

        Args:
            inputs: the input tensors to pass through the layer

        Returns:
            the output tensors from the layer

        """
        mean, variance, graph = inputs
        # calculate the alpha value
        alpha = K.exp(-self.attention_factor * variance)
        # calculate the mean
        mean = K.dot(graph, (mean * alpha))
        mean = K.dot(mean, self.mean_weight)
        # calculate the variance
        variance = K.dot(graph, (variance * alpha**2))
        variance = K.dot(variance, self.variance_weight)
        return mean, variance

    def call(self, inputs, **kwargs):
        """
        Forward pass through the layer.

        Args:
            inputs: the input tensors to pass through the layer

        Returns:
            the output tensors from the layer

        """
        if self.is_first:  # convert vectors to distributions
            mean, variance = self._call_first(inputs)
        else:  # transform the distributions
            mean, variance = self._call_generic(inputs)
        # pass the mean and variance through the activation
        mean = self.mean_activation(mean)
        variance = self.variance_activation(variance)
        # apply the dropout if enabled
        if self.dropout:
            mean = K.dropout(mean, self.dropout)
            variance = K.dropout(variance, self.dropout)
        # sample from the distribution if the last layer
        if self.is_last:
            dist = distributions.Normal(mean, K.sqrt(variance))
            return self.last_activation(dist.sample())
        return [mean, variance]

    def get_config(self):
        """Return the configuration for building the layer."""
        config = dict(
            units=self.units,
            is_first=self.is_first,
            is_last=self.is_last,
            attention_factor=self.attention_factor,
            dropout=self.dropout,
            mean_activation=activations.serialize(self.mean_activation),
            mean_initializer=initializers.serialize(self.mean_initializer),
            mean_regularizer=regularizers.serialize(self.mean_regularizer),
            mean_constraint=constraints.serialize(self.mean_constraint),
            variance_activation=activations.serialize(self.variance_activation),
            variance_initializer=initializers.serialize(self.variance_initializer),
            variance_regularizer=regularizers.serialize(self.mean_regularizer),
            variance_constraint=constraints.serialize(self.variance_constraint),
            last_activation=activations.serialize(self.last_activation),
        )
        # get items from the parent class and combine them with this layer
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


# explicitly define the outward facing API of this module
__all__ = [GaussianGraphConvolution.__name__]
