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


class GaussianGraphConvolution(Layer):
    """
    Gaussian graph convolution layer.

    Reference:
        Authors: Dingyuan Zhu, Ziwei Zhang, Peng Cui, Wenwu Zhu
        Title: Robust Graph Convolutional Networks Against Adversarial Attacks
        URL: https://doi.org/10.1145/3292500.3330851

    """

    def __init__(self, units: int, num_nodes: int,
        is_first: bool = False,
        activation: any = None,
        mean_initializer: any = 'zeros',
        variance_initializer: any = 'zeros',
        kernel_initializer: any = 'glorot_uniform',
        kernel_regularizer: any = None,
        kernel_constraint: any = None,
        activity_regularizer: any = None,
        **kwargs
    ):
        """
        TODO.

        Args:
            units: the number of weights
            num_nodes: the number of nodes in the graph
            TODO

        Returns:
            None

        """
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GaussianGraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.num_nodes = num_nodes
        self.is_first = is_first
        self.activation = activations.get(activation)
        self.mean_initializer = initializers.get(mean_initializer)
        self.variance_initializer = initializers.get(variance_initializer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.supports_masking = True
        # setup model variables
        self.kernel = None

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
        return output_shape

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

        # self.mean = self.add_weight(
        #     shape=(self.num_nodes, self.units),
        #     name='mean',
        #     initializer=self.mean_initializer,
        #     trainable=True)
        # self.variance = self.add_weight(
        #     shape=(self.num_nodes, self.units),
        #     name='variance',
        #     initializer=self.variance_initializer,
        #     trainable=True)

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        self.built = True

    def _call_first(self, inputs):
        """
        Forward pass through the layer (assuming the inputs are vectors.

        Args:
            inputs: the input tensors to pass through the layer

        Returns:
            the output tensors from the layer

        """
        features = inputs[0]
        basis = inputs[1]
        output = K.dot(basis, features)
        output = K.dot(output, self.kernel)
        return self.activation(output)

    def _call_generic(self, inputs):
        """
        Forward pass through the layer for generic cases.

        Args:
            inputs: the input tensors to pass through the layer

        Returns:
            the output tensors from the layer

        """
        features = inputs[0]
        basis = inputs[1]
        output = K.dot(basis, features)
        output = K.dot(output, self.kernel)
        return self.activation(output)

    def call(self, inputs):
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
            num_nodes=self.num_nodes,
            activation=activations.serialize(self.activation),
            mean_initializer=initializers.serialize(self.mean_initializer),
            variance_initializer=initializers.serialize(self.variance_initializer),
            kernel_initializer=initializers.serialize(self.kernel_initializer),
            kernel_regularizer=regularizers.serialize(self.kernel_regularizer),
            kernel_constraint=constraints.serialize(self.kernel_constraint),
            activity_regularizer=regularizers.serialize(self.activity_regularizer),
        )

        base_config = super(GaussianGraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# explicitly define the outward facing API of this module
__all__ = [GaussianGraphConvolution.__name__]
