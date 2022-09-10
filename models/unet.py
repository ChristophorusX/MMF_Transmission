from typing import Any, Optional, Union, Callable, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers import Adam


class ConvBlock(layers.Layer):
    """A convolutional block with the given number of filters and kernel size."""

    def __init__(
        self,
        layer_index: int,
        base_num_filters: int,
        kernel_size: int,
        dropout_rate: float,
        padding: str,
        activation: str,
        num_convs: int = 2,
    ):
        """Initialize the convolutional block.

        Args:
            layer_index (int): The index of the layer.
            base_num_filters (int): The base number of filters.
            kernel_size (int): The size of the convolutional kernel.
            dropout_rate (float): The dropout rate.
            padding (str): The padding type.
            activation (str): The activation function.
            num_convs (int, optional): The number of convolutional layers. Defaults to 2.
        """
        super(ConvBlock, self).__init__()
        self.layer_index = layer_index
        self.base_num_filters = base_num_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.activation = activation
        self.num_convs = num_convs

        filters = _get_filter_count(layer_index, self.base_num_filters)

        self.conv2ds = []
        for _ in range(self.num_convs):
            conv = layers.Conv2D(
                filters=filters,
                kernel_size=(kernel_size, kernel_size),
                kernel_initializer=_get_kernel_initializer(
                    filters, kernel_size),
                strides=1,
                padding=padding,
            )
            self.conv2ds.append(conv)

        self.dropouts = []
        for _ in range(self.num_convs):
            dropout = layers.Dropout(rate=dropout_rate)
            self.dropouts.append(dropout)

        self.activations = []
        for _ in range(self.num_convs):
            activation = layers.Activation(activation)
            self.activations.append(activation)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass of the convolutional block.

        Args:
            inputs (tf.Tensor): The input tensor.
            training (bool, optional): The indication of training mode. Defaults to None.

        Returns:
            tf.Tensor: The output tensor.
        """
        x = inputs

        for conv2d, dropout, activation in zip(self.conv2ds, self.dropouts, self.activations):
            x = conv2d(x)
            if training:
                x = dropout(x)
            x = activation(x)

        return x

    def get_config(self) -> dict:
        """Gets the configuration of the convolutional block.

        Returns:
            dict: The configuration of the convolutional block.
        """
        return dict(
            layer_index=self.layer_index,
            base_num_filters=self.base_num_filters,
            kernel_size=self.kernel_size,
            dropout_rate=self.dropout_rate,
            padding=self.padding,
            activation=self.activation,
            **super(ConvBlock, self).get_config(),
        )


class UpconvBlock(layers.Layer):
    """An upsampling block with the given number of filters and kernel size."""

    def __init__(
        self,
        layer_index: int,
        base_num_filters: int,
        kernel_size: int,
        pool_size: int,
        padding: str,
        activation: str,
    ):
        """Initialize the upsampling block.

        Args:
            layer_index (int): The index of the layer.
            base_num_filters (int): The base number of filters.
            kernel_size (int): The size of the convolutional kernel.
            pool_size (int): The size of the max pooling kernel.
            padding (str): The padding type.
            activation (str): The activation function.
        """
        super(UpconvBlock, self).__init__()
        self.layer_index = layer_index
        self.base_num_filters = base_num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.padding = padding
        self.activation = activation

        filters = _get_filter_count(layer_index + 1, self.base_num_filters)
        self.upconv = layers.Conv2DTranspose(
            filters // 2,
            kernel_size=(
                pool_size,
                pool_size,
            ),
            kernel_initializer=_get_kernel_initializer(
                filters,
                kernel_size,
            ),
            strides=pool_size,
            padding=padding,
        )

        self.activation = layers.Activation(activation)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the upsampling block.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor.
        """
        x = inputs
        x = self.upconv(x)
        x = self.activation(x)

        return x

    def get_config(self) -> dict:
        """Gets the configuration of the upsampling block.

        Returns:
            dict: The configuration of the upsampling block.
        """
        return dict(
            layer_index=self.layer_index,
            base_num_filters=self.base_num_filters,
            kernel_size=self.kernel_size,
            pool_size=self.pool_size,
            padding=self.padding,
            activation=self.activation,
            **super(UpconvBlock, self).get_config(),
        )


class ConcatBlock(layers.Layer):
    """A concatenation block."""

    def call(self, x: tf.Tensor, concatenated_layer: tf.Tensor) -> tf.Tensor:
        """Forward pass of the concatenation block.

        Args:
            x (tf.Tensor): The input tensor.
            concatenated_layer (tf.Tensor): The output tensor of the corresponding downsampling block.

        Returns:
            tf.Tensor: The output tensor.
        """
        downsampled_shape = tf.shape(concatenated_layer)
        upsampled_shape = tf.shape(x)

        height_diff = (downsampled_shape[1] - upsampled_shape[1]) // 2
        width_diff = (downsampled_shape[2] - upsampled_shape[2]) // 2

        concatenated_layer_cropped = concatenated_layer[:,
                                                        height_diff: (upsampled_shape[1] + height_diff),
                                                        width_diff: (upsampled_shape[2] + width_diff),
                                                        :]

        x = tf.concat([concatenated_layer_cropped, x], axis=-1)
        return x


def construct_model(x_dims: Optional[int] = None,
                    y_dims: Optional[int] = None,
                    channels: int = 1,
                    layer_depth: int = 5,
                    base_num_filters: int = 64,
                    kernel_size: int = 3,
                    pool_size: int = 2,
                    dropout_rate: int = 0.5,
                    padding: str = "same",
                    activation: Union[str, Callable] = "relu") -> Model:
    """Constructs the U-Net model.

    Args:
        x_dims (Optional[int], optional): The input dimension on x-axis. Defaults to None.
        y_dims (Optional[int], optional): The input dimension on y-axis. Defaults to None.
        channels (int, optional): The number of channels of the input image. Defaults to 1.
        layer_depth (int, optional): The depth of the U-Net model. Defaults to 5.
        base_num_filters (int, optional): The number of convolutional filters at input layer. Defaults to 64.
        kernel_size (int, optional): The size of convolutional kernel. Defaults to 3.
        pool_size (int, optional): The size of maxpooling. Defaults to 2.
        dropout_rate (int, optional): The dropout rate. Defaults to 0.5.
        padding (str, optional): The padding type. Defaults to "same".
        activation (Union[str, Callable], optional): The activation function. Defaults to "relu".

    Returns:
        Model: A U-Net model.
    """
    inputs = Input(shape=(x_dims, y_dims, channels), name="inputs")

    x = inputs
    contracting_layers = {}
    # Contracting path
    for layer_index in range(layer_depth - 1):
        x = ConvBlock(
            layer_index=layer_index,
            base_num_filters=base_num_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            padding=padding,
            activation=activation,
        )(x)
        contracting_layers[layer_index] = x
        x = layers.MaxPooling2D((pool_size, pool_size))(x)
        print(f"Contracting layer {layer_index} shape: {x.shape}")
    # Bottleneck
    x = ConvBlock(
        layer_index + 1, base_num_filters=base_num_filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        padding=padding,
        activation=activation,)(x)
    print(f"Bottleneck layer shape: {x.shape}")
    # Expansive path
    for layer_index in range(layer_index, -1, -1):
        x = UpconvBlock(
            layer_index=layer_index,
            base_num_filters=base_num_filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            padding=padding,
            activation=activation,
        )(x)
        x = ConcatBlock()(x, contracting_layers[layer_index])
        x = ConvBlock(
            layer_index=layer_index,
            base_num_filters=base_num_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            padding=padding,
            activation=activation,
        )(x)
        print(f"Expansive layer {layer_index} shape: {x.shape}")
    # Output layer
    outputs = layers.Conv2D(
        filters=channels,
        kernel_size=(1, 1),
        activation=activation,
        padding=padding,
    )(x)

    return Model(inputs=inputs, outputs=outputs, name="unet")


def configure_model(model: Model,
                    loss: Optional[Union[Callable, str]
                                   ] = losses.categorical_crossentropy,
                    optimizer: Any = None,
                    metrics: Optional[List[Union[Callable, str]]] = None,
                    auc: bool = True,
                    learning_rate: float = 1e-4,
                    ):
    """Configures the model.

    Args:
        model (Model): The model to be configured.
        loss (Optional[Union[Callable, str] ], optional): The loss function. Defaults to losses.categorical_crossentropy.
        optimizer (Any, optional): The optimizer. Defaults to None.
        metrics (Optional[List[Union[Callable, str]]], optional): The metrics for tracking. Defaults to None.
        auc (bool, optional): Whether to track AUC. Defaults to True.
        learning_rate (float, optional): The learning rate. Defaults to 1e-4.
    """

    if optimizer is None:
        optimizer = Adam(learning_rate=learning_rate)

    if metrics is None:
        metrics = [
            'mse',
        ]

    if auc:
        metrics += [tf.keras.metrics.AUC()]

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
        run_eagerly=True,
    )


def _get_filter_count(layer_index: int, base_num_filters: int) -> int:
    """Gets the number of filters for a given layer.

    Args:
        layer_index (int): The layer index.
        base_num_filters (int): The number of filters at the first layer.

    Returns:
        int: The number of filters.
    """
    return 2 ** layer_index * base_num_filters


def _get_kernel_initializer(filters: int, kernel_size: int) -> Any:
    """Gets the kernel initializer.

    Args:
        filters (int): The number of filters.
        kernel_size (int): The size of the kernel.

    Returns:
        Any: A kernel initializer.
    """
    std = np.sqrt(2 / (kernel_size ** 2 * filters))
    return TruncatedNormal(stddev=std)


if __name__ == "__main__":
    model = construct_model()
    model.summary()

    configure_model(model)

    dummy_input = tf.convert_to_tensor(np.random.rand(32, 256, 256, 1))
    output = model.predict(dummy_input)
    print(f"Output shape: {output.shape}")
