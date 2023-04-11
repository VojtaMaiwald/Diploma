# https://github.com/Haikoitoh/paper-implementation/blob/main/ShuffleNet.ipynb

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Add, AveragePooling2D, Dense, AvgPool2D, BatchNormalization, ReLU, DepthwiseConv2D, Reshape, Permute, Conv2D, MaxPool2D, GlobalAveragePooling2D, concatenate


def channel_shuffle(x, groups):
    _, width, height, channels = x.get_shape().as_list()
    group_ch = channels // groups

    x = Reshape([width, height, group_ch, groups])(x)
    x = Permute([1, 2, 4, 3])(x)
    x = Reshape([width, height, channels])(x)
    return x

def shuffle_unit(x, groups, channels, strides):
    y = x
    x = Conv2D(channels//4, kernel_size=1, strides=(1, 1), padding='same', groups=groups)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = channel_shuffle(x, groups)

    x = DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same')(x)
    x = BatchNormalization()(x)

    if strides == (2, 2):
        channels = channels - y.shape[-1]
    x = Conv2D(channels, kernel_size=1, strides=(1, 1), padding='same', groups=groups)(x)
    x = BatchNormalization()(x)

    if strides == (1, 1):
        x = Add()([x, y])
    if strides == (2, 2):
        y = AvgPool2D((3, 3), strides=(2, 2), padding='same')(y)
        x = concatenate([x, y])
    x = ReLU()(x)

    return x

def ShuffleNet(start_channels, classes = 1000, input_shape=(224, 224, 3), include_top = True):
    groups = 2
    input = Input(input_shape)

    x = Conv2D(24, kernel_size=3, strides=(2, 2), padding='same', use_bias=True)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

    repetitions = [3, 7, 3]

    for i, repetition in enumerate(repetitions):
        channels = start_channels * (2**i)

        x = shuffle_unit(x, groups, channels, strides=(2, 2))

        for i in range(repetition):
            x = shuffle_unit(x, groups, channels, strides=(1, 1))

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes, activation='softmax')(x)

    model = Model(input, x)
    return model
