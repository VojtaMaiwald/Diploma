#https://github.com/CarloLepelaars/ghostnet_tf2

from keras.models import Model
from keras import backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Lambda, Reshape, DepthwiseConv2D, Layer, Concatenate, add

from math import ceil

class GhostModule(Layer):
    """
    The main Ghost module
    """
    def __init__(self, out, ratio, convkernel, dwkernel):
        super(GhostModule, self).__init__()
        self.ratio = ratio
        self.out = out
        self.conv_out_channel = ceil(self.out * 1.0 / ratio)
        self.conv = Conv2D(int(self.conv_out_channel), (convkernel, convkernel), use_bias=False,
                           strides=(1, 1), padding='same', activation=None)
        self.depthconv = DepthwiseConv2D(dwkernel, 1, padding='same', use_bias=False,
                                         depth_multiplier=ratio-1, activation=None)
        self.slice = Lambda(self._return_slices, arguments={'channel': int(self.out - self.conv_out_channel)})
        self.concat = Concatenate()

    @staticmethod
    def _return_slices(x, channel):
        return x[:, :, :, :channel]

    def call(self, inputs):
        x = self.conv(inputs)
        if self.ratio == 1:
            return x
        dw = self.depthconv(x)
        dw = self.slice(dw)
        output = self.concat([x, dw])
        return output

class SEModule(Layer):
    """
    A squeeze and excite module
    """
    def __init__(self, filters, ratio):
        super(SEModule, self).__init__()
        self.pooling = GlobalAveragePooling2D()
        self.reshape = Lambda(self._reshape)
        self.conv1 = Conv2D(int(filters / ratio), (1, 1), strides=(1, 1), padding='same',
                           use_bias=False, activation=None)
        self.conv2 = Conv2D(int(filters), (1, 1), strides=(1, 1), padding='same',
                           use_bias=False, activation=None)
        self.relu = Activation('relu')
        self.hard_sigmoid = Activation('hard_sigmoid')

    @staticmethod
    def _reshape(x):
        return Reshape((1, 1, int(x.shape[1])))(x)

    @staticmethod
    def _excite(x, excitation):
        """
        Multiply by an excitation factor
        :param x: A Tensorflow Tensor
        :param excitation: A float between 0 and 1
        :return:
        """
        return x * excitation

    def call(self, inputs):
        x = self.reshape(self.pooling(inputs))
        x = self.relu(self.conv1(x))
        excitation = self.hard_sigmoid(self.conv2(x))
        x = Lambda(self._excite, arguments={'excitation': excitation})(inputs)
        return x

class GBNeck(Layer):
    """
    The GhostNet Bottleneck
    """
    def __init__(self, dwkernel, strides, exp, out, ratio, use_se):
        super(GBNeck, self).__init__()
        self.strides = strides
        self.use_se = use_se
        self.conv = Conv2D(out, (1, 1), strides=(1, 1), padding='same',
                           activation=None, use_bias=False)
        self.relu = Activation('relu')
        self.depthconv1 = DepthwiseConv2D(dwkernel, strides, padding='same', depth_multiplier=ratio-1,
                                         activation=None, use_bias=False)
        self.depthconv2 = DepthwiseConv2D(dwkernel, strides, padding='same', depth_multiplier=ratio-1,
                                         activation=None, use_bias=False)
        for i in range(5):
            setattr(self, f"batchnorm{i+1}", BatchNormalization())
        self.ghost1 = GhostModule(exp, ratio, 1, 3)
        self.ghost2 = GhostModule(out, ratio, 1, 3)
        self.se = SEModule(exp, ratio)

    def call(self, inputs):
        x = self.batchnorm1(self.depthconv1(inputs))
        x = self.batchnorm2(self.conv(x))

        y = self.relu(self.batchnorm3(self.ghost1(inputs)))
        # Extra depth conv if strides higher than 1
        if self.strides > 1:
            y = self.relu(self.batchnorm4(self.depthconv2(y)))
        # Squeeze and excite
        if self.use_se:
            y = self.se(y)
        y = self.batchnorm5(self.ghost2(y))
        # Skip connection
        return add([x, y])


class GhostNet(Model):
    """
    The main GhostNet architecture as specified in "GhostNet: More Features from Cheap Operations"
    Paper:
    https://arxiv.org/pdf/1911.11907.pdf
    """
    def __init__(self, classes):
        super(GhostNet, self).__init__()
        self.classes = classes
        self.conv1 = Conv2D(16, (3, 3), strides=(2, 2), padding='same',
                            activation=None, use_bias=False)
        self.conv2 = Conv2D(960, (1, 1), strides=(1, 1), padding='same', data_format='channels_last',
                            activation=None, use_bias=False)
        self.conv3 = Conv2D(1280, (1, 1), strides=(1, 1), padding='same',
                            activation=None, use_bias=False)
        self.conv4 = Conv2D(self.classes, (1, 1), strides=(1, 1), padding='same',
                            activation=None, use_bias=False)
        for i in range(3):
            setattr(self, f"batchnorm{i+1}", BatchNormalization())
        self.relu = Activation('relu')
        self.softmax = Activation('softmax')
        self.squeeze = Lambda(self._squeeze)
        self.reshape = Lambda(self._reshape)
        self.pooling = GlobalAveragePooling2D()

        self.dwkernels = [3, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5]
        self.strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
        self.exps = [16, 48, 72, 72, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960, 960, 960]
        self.outs = [16, 24, 24, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 160, 160]
        self.ratios = [2] * 16
        self.use_ses = [False, False, False, True, True, False, False, False,
                        False, True, True, True, False, True, False, True]
        for i, args in enumerate(zip(self.dwkernels, self.strides, self.exps, self.outs, self.ratios, self.use_ses)):
            setattr(self, f"gbneck{i}", GBNeck(*args))

    @staticmethod
    def _squeeze(x):
        """
        Remove all axes with a dimension of 1
        """
        return K.squeeze(x, 1)

    @staticmethod
    def _reshape(x):
        return Reshape((1, 1, int(x.shape[1])))(x)

    def call(self, inputs):
        x = self.relu(self.batchnorm1(self.conv1(inputs)))
        # Iterate through Ghost Bottlenecks
        for i in range(16):
            x = getattr(self, f"gbneck{i}")(x)
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.reshape(self.pooling(x))
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.conv4(x)
        x = self.squeeze(x)
        output = self.softmax(x)
        return output