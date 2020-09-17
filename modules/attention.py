import tensorflow as tf
from tensorflow.keras import Model

from tensorflow.keras.applications import (
    EfficientNetB7,
    EfficientNetB5
)

layers = tf.keras.layers
reg = tf.keras.regularizers


class AttentionModel(tf.keras.Model):
    def __init__(self, kernel_size=1, decay=0.0001, name='attention'):
        """Initialization of attention model.
        Args:
            kernel_size: int, kernel size of convolutions.
            decay: float, decay for l2 regularization of kernel weights.
            name: str, name to identify model.
        """
        super(AttentionModel, self).__init__(name=name)

        # First convolutional layer (called with relu activation).
        self.conv1 = layers.Conv2D(
                512,
                kernel_size,
                kernel_regularizer=reg.l2(decay),
                padding='same',
                name='attn_conv1')
        self.bn_conv1 = layers.BatchNormalization(axis=3, name='bn_conv1')

        # Second convolutional layer, with softplus activation.
        self.conv2 = layers.Conv2D(
                1,
                kernel_size,
                kernel_regularizer=reg.l2(decay),
                padding='same',
                name='attn_conv2')
        self.activation_layer = layers.Activation('softplus')

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.bn_conv1(x, training=training)
        x = tf.nn.relu(x)

        score = self.conv2(x)
        prob = self.activation_layer(score)

        # L2-normalize the featuremap before pooling.
        inputs = tf.nn.l2_normalize(inputs, axis=-1)
        feat = tf.reduce_mean(tf.multiply(inputs, prob), [1, 2], keepdims=False)

        return feat, prob, score

def att_func():

    return AttentionModel(name='attention')


