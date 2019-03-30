import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import mean_squared_error

from . import game


def model_for_key(key) -> tf.keras.Model:
    if key == 'residual_conv_net':
        return residual_conv_net


def residual_conv_net():
    board = Input(shape=[6, 6, 1],
                  dtype=tf.float32,
                  name='board')

    features = _conv_layer(board, 3, 3)
    features = _residual_layer(features, 3, 3)
    features = _residual_layer(features, 3, 3)

    value = _value_head(features)
    policy = _policy_head(features)

    model = tf.keras.Model(inputs=[board],
                           outputs=[policy, value])
    model.compile('adam',
                  loss=[_policy_loss, mean_squared_error])
    return model


def _policy_loss(mc_values, pred_values):
    # mc_values = tf.reshape(mc_values,
    #                        tf.stack([tf.shape(mc_values)[0], -1]))
    # pred_values = tf.reshape(pred_values,
    #                          tf.stack([tf.shape(pred_values)[0], -1]))
    pred_values = tf.where(tf.equal(mc_values, tf.zeros_like(mc_values)),
                           tf.tile(tf.expand_dims(tf.expand_dims(-tf.float32.max, 0), 1),
                                   tf.shape(pred_values)),
                           pred_values)
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=mc_values,
                                                      logits=pred_values)


def _value_head(inputs):
    convd = _conv_layer(inputs, 1, 1)
    flattened = Flatten()(convd)
    out = Dense(20,
                use_bias=False,
                activation=None,
                kernel_regularizer=l2(1e-3))(flattened)
    out = Dense(1,
                name='value_head',
                activation='tanh')(out)
    return out


def _policy_head(inputs):
    out = _conv_layer(inputs, 4, 1)
    out = Flatten()(out)
    out = Dense(game.PROBABILITY_OUT_DIM,
                use_bias=False,
                activation=None,
                name='policy_head')(out)
    return out


def _conv_layer(inputs, filters, kernel_size):
    layer = Conv2D(filters,
                   kernel_size,
                   use_bias=False)
    out = layer(inputs)
    out = BatchNormalization(axis=-1)(out)
    out = LeakyReLU()(out)
    return out


def _residual_layer(inputs, filters, kernel_size):
    layer = Conv2D(filters,
                   kernel_size,
                   use_bias=False,
                   kernel_regularizer=l2(1e-3),
                   padding='same')
    out = layer(inputs)
    out = BatchNormalization(axis=-1)(out)
    out = tf.keras.layers.add([inputs, out])
    out = LeakyReLU()(out)
    return out
