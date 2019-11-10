"""
Gaussian graph convolution layer.

Reference:
    Authors: Dingyuan Zhu, Ziwei Zhang, Peng Cui, Wenwu Zhu
    Title: Robust Graph Convolutional Networks Against Adversarial Attacks
    URL: https://doi.org/10.1145/3292500.3330851

"""
# from keras.engine.base_layer import InputSpec
from keras.engine.topology import Layer
from keras import activations, initializers, regularizers, constraints
# from keras.utils import conv_utils
import keras.backend as K


class GaussianGraphConvolution(Layer):
    """
    Gaussian graph convolution layer.

    Reference:
        Authors: Dingyuan Zhu, Ziwei Zhang, Peng Cui, Wenwu Zhu
        Title: Robust Graph Convolutional Networks Against Adversarial Attacks
        URL: https://doi.org/10.1145/3292500.3330851

    """

    def __init__(self, units,
        support: int = 1,
        activation: any = None,
        use_bias: bool = True,
        kernel_initializer: any = 'glorot_uniform',
        bias_initializer: any = 'zeros',
        kernel_regularizer: any = None,
        bias_regularizer: any = None,
        activity_regularizer: any = None,
        kernel_constraint: any = None,
        bias_constraint: any = None,
        **kwargs
    ):
        """
        TODO.

        Args:
            TODO

        Returns:
            None

        """
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GaussianGraphConvolution, self).__init__(**kwargs)
        # TODO
        # self.input_spec = InputSpec(ndim=4)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True
        if support < 1:
            raise ValueError('support must be >= 1')
        self.support = support
        # setup model variables
        self.kernel = None
        self.bias = None

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
        return output_shape  # (batch_size, output_dim)

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

        self.kernel = self.add_weight(
            shape=(input_dim * self.support, self.units),
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None
        # TODO: set input specification for th layer
        # self.input_spec = InputSpec(ndim=4, axes={'channel_axis': input_dim})
        self.built = True

    def call(self, inputs):
        """
        Forward pass through the layer.

        Args:
            inputs: the input tensor to pass through the pyramid pooling module

        Returns:
            the output tensor from the pyramid pooling module

        """
        features = inputs[0]
        basis = inputs[1:]

        supports = list()
        for i in range(self.support):
            supports.append(K.dot(basis[i], features))
        supports = K.concatenate(supports, axis=1)
        output = K.dot(supports, self.kernel)

        if self.bias:
            output += self.bias

        return self.activation(output)

    def get_config(self):
        """Return the configuration for building the layer."""
        config = dict(
            units=self.units,
            support=self.support,
            activation=activations.serialize(self.activation),
            use_bias=self.use_bias,
            kernel_initializer=initializers.serialize(self.kernel_initializer),
            bias_initializer=initializers.serialize(self.bias_initializer),
            kernel_regularizer=regularizers.serialize(self.kernel_regularizer),
            bias_regularizer=regularizers.serialize(self.bias_regularizer),
            activity_regularizer=regularizers.serialize(self.activity_regularizer),
            kernel_constraint=constraints.serialize(self.kernel_constraint),
            bias_constraint=constraints.serialize(self.bias_constraint)
        )

        base_config = super(GaussianGraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# explicitly define the outward facing API of this module
__all__ = [GaussianGraphConvolution.__name__]
