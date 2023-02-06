#https://github.com/jh88/fbnet
import tensorflow as tf
from keras import Model
from keras.layers import BatchNormalization, Layer
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import Mean, SparseCategoricalAccuracy
from keras.optimizers import Adam, SGD
from keras import backend as K

def gumbel_softmax(logits, tau, axis=-1):
    shape = K.int_shape(logits)
    
    # Gumbel(0, 1)
    if len(shape) == 1:
        gumbels = K.log(tf.random.gamma(shape, 1))
    else:
        gumbels = K.log(
            tf.random.gamma(shape[:-1], [1 for _ in range(shape[-1])])
        )
        
    # Gumbel(logits, tau)
    gumbels = (logits + gumbels) / tau
    
    y_soft = K.softmax(gumbels, axis=axis)
    
    return y_soft


class MixedOperation(Layer):
    def __init__(self, blocks, latency, **kwargs):
        super().__init__(**kwargs)
        self.ops = blocks
        self.theta = self.add_weight(
            '{}/theta'.format(kwargs.get('name', 'mixed_operation')),
            shape=(len(blocks),),
            initializer=tf.ones_initializer
        )

        self.latency = tf.constant(latency, dtype=tf.float32)

    def call(self, inputs, temperature, training=False):
        mask_variables = gumbel_softmax(self.theta, temperature)

        self.add_loss(tf.reduce_sum(mask_variables * self.latency))

        x = sum(
            mask_variables[i] * op(inputs, training=training)
            for i, op in enumerate(self.ops)
        )

        return x

    def sample(self, temperature=None):
        mask_variables = gumbel_softmax(self.theta, temperature)
        mask = tf.argmax(mask_variables)
        op = self.ops[mask]

        return op


class FBNet(Model):
    def __init__(
        self,
        super_net,
        lookup_table=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ops = []
        for i, layer in enumerate(super_net):
            if isinstance(layer, Layer):
                self.ops.append(layer)
            elif isinstance(layer, list):
                latency = lookup_table[i] if lookup_table else None
                self.ops.append(
                    MixedOperation(layer, latency, name='tbs{}'.format(i))
                )

    def call(self, inputs, temperature=5, training=False):
        x = inputs
        for op in self.ops:
            if isinstance(op, MixedOperation):
                x = op(x, temperature, training)
            elif isinstance(op, BatchNormalization):
                x = op(x, training=training)
            else:
                x = op(x)

        return x