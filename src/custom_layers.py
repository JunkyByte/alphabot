from keras.layers import *
from keras.regularizers import l2
from keras.models import Model
from keras.utils import multi_gpu_model
import tensorflow as tf

# Convolutional Block
def conv_block(in_layer, name, filters, kernel_size=(3, 3), bn=True, relu=True):
    l = Conv2D(filters, kernel_size, use_bias = False, 
               padding='same', name = name, kernel_regularizer=l2(1e-4))(in_layer)
    if bn:
        l = BatchNormalization(axis=3, name = name + '_bn')(l)
    if relu:
        l = Activation('relu', name = name + '_relu')(l)
        
    return l


# Residual Block
def residual_conv(in_layer, idx, filters, kernel_size=(3, 3), bn=True, relu=True):
    name = 'res_' + str(idx)
    # Full conv block of pre-defined shape
    l = conv_block(in_layer, name + '_conv1', filters, kernel_size=(3, 3), bn=True, relu=True)
    # Second block with skip connection
    l = Conv2D(filters, kernel_size, use_bias = False, padding='same', 
               name = name + '_conv2', kernel_regularizer=l2(1e-4))(l)
    if bn:
        l = BatchNormalization(axis=3, name = name + '_conv2_bn')(l)
    
    l = Add()([in_layer, l])  # Skip conn.
    
    if relu:
        l = Activation('relu', name=name + '_relu')(l)
        
    return l


def value_head(in_layer):
    l = conv_block(in_layer, 'value_head', filters=1, kernel_size=(1,1))
    l = Flatten(name = 'value_flatten')(l)

    l = Dense(128, use_bias=False, kernel_regularizer=l2(1e-4), activation='relu', 
              name = 'value_dense')(l)

    l = BatchNormalization(axis=1, name = 'value_bn')(l)
    l = Dense(1, use_bias = False, name = 'value', kernel_regularizer=l2(1e-4),
              activation='tanh')(l) # Value output

    return l


def policy_head(in_layer):
    l = conv_block(in_layer, 'policy_head', filters=2, kernel_size=(1,1),
                   kernel_regularizer=l2(1e-4))

    l = Flatten(name = 'policy_flatten')(l)
    l = Dense(4, name = 'policy', use_bias = False, kernel_regularizer=l2(1e-4),
              activation='softmax')(l) # Policy output
    return l


class ZeroConv(Layer):

    def __init__(self, **kwargs):
        super(ZeroConv, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ZeroConv, self).build(input_shape)

    def call(self, x):
        prev_conv, next_conv = x
        zero_mask = tf.equal(prev_conv, 0)
        one_mask = tf.math.logical_not(zero_mask)
        one_mask = tf.cast(one_mask, next_conv.dtype)

        return next_conv * one_mask + next_conv * 0

    def compute_output_shape(self, input_shape):
        return input_shape


def declare_model(n_channels, n_residual):

    input_layer = Input(INPUT_SIZE)
    l = conv_block(input_layer, 'conv')
    for i in range(n_residual):
        l = residual_conv(l, idx=i + 1, filters=n_channels)

    policy = policy_head(l)
    value = value_head(l)

    alphabot = Model(input_layer, [policy, value])
    return alphabot
