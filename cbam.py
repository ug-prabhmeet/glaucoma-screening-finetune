"""
cbam.py — Convolutional Block Attention Module (CBAM).

Implements channel + spatial attention as Keras layers.
Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
"""

import tensorflow as tf
from tensorflow.keras import layers


class ChannelAttention(layers.Layer):
    """Squeeze‑and‑excitation style channel attention."""

    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.dense1 = None
        self.dense2 = None
        self.channels = None
        self.reduced_channels = None

    def build(self, input_shape):
        self.channels = int(input_shape[-1])
        self.reduced_channels = max(self.channels // self.reduction_ratio, 1)

        self.dense1 = layers.Dense(
            self.reduced_channels,
            activation="relu",
            kernel_initializer="he_normal",
            use_bias=True,
        )
        self.dense2 = layers.Dense(
            self.channels,
            kernel_initializer="he_normal",
            use_bias=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)

        avg_pool = tf.reduce_mean(x, axis=[1, 2])
        max_pool = tf.reduce_max(x, axis=[1, 2])

        avg_out = self.dense2(self.dense1(avg_pool))
        max_out = self.dense2(self.dense1(max_pool))

        attention = tf.nn.sigmoid(avg_out + max_out)
        attention = tf.reshape(attention, [-1, 1, 1, self.channels])
        attention = tf.cast(attention, inputs.dtype)

        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


class SpatialAttention(layers.Layer):
    """7×7 convolution‑based spatial attention."""

    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = None

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)

        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)

        attention = tf.nn.sigmoid(self.conv(concat))
        attention = tf.cast(attention, inputs.dtype)

        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


class CBAM(layers.Layer):
    """Convolutional Block Attention Module (channel → spatial)."""

    def __init__(self, reduction_ratio=16, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(
            reduction_ratio=reduction_ratio,
        )
        self.spatial_attention = SpatialAttention(
            kernel_size=kernel_size,
        )

    def build(self, input_shape):
        self.channel_attention.build(input_shape)
        self.spatial_attention.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "kernel_size": self.kernel_size,
        })
        return config
