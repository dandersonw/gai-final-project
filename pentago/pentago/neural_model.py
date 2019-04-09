import tensorflow as tf
import numpy as np

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Flatten
from keras.regularizers import l2
from keras.losses import mean_squared_error

from . import game


def model_for_key(key) -> tf.keras.Model:
    if key == 'residual_conv_net':
        return residual_conv_net


DEFAULT_MODEL_PARAMS = {'dense_regularization_const': 1e-2,
                        'regularization_const': 0,
                        'num_layers': 5,
                        'kernel_size': 3,
                        'num_filters': 64}


def residual_conv_net(**config):
    board = Input(shape=[6, 6, 1],
                  dtype=tf.float32,
                  name='board')

    features = _conv_layer(board,
                           config['num_filters'],
                           config['kernel_size'],
                           config)
    for i in range(config['num_layers']):
        features = _residual_layer(features,
                                   config['num_filters'],
                                   config['kernel_size'],
                                   config)

    value = _value_head(features, config)
    policy = _policy_head(features, config)

    model = tf.keras.Model(inputs=[board],
                           outputs=[policy, value])
    model.compile('adam',
                  loss=[_policy_loss, mean_squared_error])
    return model


def _policy_loss(mc_values, pred_values):
    pred_values = tf.where(tf.equal(mc_values, tf.zeros_like(mc_values)),
                           tf.tile(tf.expand_dims(tf.expand_dims(-tf.float32.max, 0), 1),
                                   tf.shape(pred_values)),
                           pred_values)
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=mc_values,
                                                      logits=pred_values)


def _value_head(inputs, config):
    convd = _conv_layer(inputs, 1, 1, config)
    flattened = Flatten()(convd)
    layer = Dense(20,
                  kernel_regularizer=l2(config['dense_regularization_const']))
    out = layer(flattened)
    layer = Dense(1,
                  name='value_head',
                  activation='tanh')
    out = layer(out)
    return out


def _policy_head(inputs, config):
    out = _conv_layer(inputs, 8, 1, config)
    out = Flatten()(out)
    out = Dense(game.PROBABILITY_OUT_DIM,
                use_bias=False,
                activation=None,
                name='policy_head')(out)
    return out


def _conv_layer(inputs, filters, kernel_size, config):
    layer = Conv2D(filters,
                   kernel_size,
                   padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(config['regularization_const']))
    out = layer(inputs)
    out = BatchNormalization(axis=-1)(out)
    out = LeakyReLU()(out)
    return out


def _residual_layer(inputs, filters, kernel_size, config):
    layer = Conv2D(filters,
                   kernel_size,
                   use_bias=False,
                   kernel_regularizer=l2(config['regularization_const']),
                   padding='same')
    out = layer(inputs)
    out = BatchNormalization(axis=-1)(out)
    out = tf.keras.layers.add([inputs, out])
    out = LeakyReLU()(out)
    return out
