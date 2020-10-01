import tensorflow as tf
import tensorflow.keras as ks
import gin
import numpy as np


@gin.configurable('ExampleModel')
class Architecture(tf.keras.Model):
    def __init__(self,
                 n_filters_in=16,
                 kernel_size=3,
                 dim_h=10):
        super().__init__(name='ExampleModel')
        # Define layers here
        self.conv_list = []
        self.num_convs = 3
        for idx_conv in range(self.num_convs):
            self.conv_list.append(ks.layers.Conv2D(filters=n_filters_in*2**idx_conv, kernel_size=kernel_size, strides=(2,2), padding='same', activation='relu'))
        self.pool = ks.layers.GlobalAveragePooling2D()
        self.flatten = ks.layers.Flatten()
        self.fc1 = ks.layers.Dense(dim_h)

    @tf.function
    def call(self, inputs, training=False):
        # connect layers here
        features = inputs
        for idx_conv in range(self.num_convs):
            features = self.conv_list[idx_conv](features)
        features = self.pool(features)
        features = self.flatten(features)
        h = self.fc1(features)
        return features, h
