import tensorflow as tf

def shufflenet_v2_block(inputs, filters, stride, name):
    with tf.variable_scope(name):
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if stride == 2:
            inputs = tf.keras.layers.AveragePooling2D(pool_size=3, strides=2, padding='same')(inputs)

        x = tf.keras.layers.Concatenate()([x, inputs])
        x = tf.keras.layers.ReLU()(x)

    return x

def shufflenet_v2(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(filters=24, kernel_size=3, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = shufflenet_v2_block(x, filters=144, stride=2, name='block_1')
    x = shufflenet_v2_block(x, filters=288, stride=2, name='block_2')
    x = shufflenet_v2_block(x, filters=576, stride=2, name='block_3')

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model
