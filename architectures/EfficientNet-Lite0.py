import tensorflow as tf
import numpy as np

class EfficientNetLite0:
    def __init__(self, input_shape, n_classes):
        self.input_shape = input_shape
        self.n_classes = n_classes

        self.model = tf.keras.Sequential()

        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, X, y, batch_size, epochs):
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs)