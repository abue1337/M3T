import tensorflow as tf
import tensorflow.keras as ks
import gin
import numpy as np
from model.architectures import Resnet
from model.architectures import resnet18


# MLP class for projector and predictor


class MLP(tf.keras.Model):
    def __init__(self, hidden_size, projection_size, momentum, weight_decay,unsup =False):
        super().__init__()
        self.unsup=unsup
        self._kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
        if unsup == True:
            self.dense = tf.keras.layers.Dense(hidden_size, activation='relu', kernel_regularizer=self._kernel_regularizer)
            # self.batch_norm = tf.keras.layers.BatchNormalization(momentum=momentum)
            # self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(projection_size, kernel_regularizer=self._kernel_regularizer)

    def call(self, x, training=False):
        output = x
        if self.unsup == True:
            output = self.dense(x)
            #output = self.batch_norm(output, training=training)
        output = self.dense1(output)
        return output


@gin.configurable('OnlineNetwork')
class Architecture(tf.keras.Model):
    def __init__(self, shape, hidden_size, projection_size, num_classes=None,
                 base_model='resnet50', num_initial_filters=16, num_layers=20, weight_decay=2e-4,group_norm_groups=16):
        super().__init__(name='OnlineNetwork')
        self.shape = shape
        if base_model == 'resnet50':
            self.encoder = tf.keras.applications.ResNet50(input_shape=shape, include_top=False, weights=None)
        elif base_model == 'resnet18':
            self.encoder = resnet18.resnet_18()
        elif base_model == 'resnet20':
            self.encoder = Resnet.Architecture(num_classes, num_initial_filters=num_initial_filters,
                                               num_layers=num_layers, weight_decay=weight_decay,group_norm_groups=group_norm_groups)
        #self.projector = MLP(hidden_size=hidden_size, projection_size=projection_size, momentum=0.9,
        #                      weight_decay=weight_decay,unsup=True)
        self.predictor = MLP(hidden_size=hidden_size, projection_size=projection_size, momentum=0.9,
                             weight_decay=weight_decay,unsup=True)
        self.classifier = MLP(hidden_size=hidden_size, projection_size=10, momentum=0.9,
                              weight_decay=weight_decay)

    def call(self, inputs, unsupervised_training=False, online=False, training=False):
        # connect layers here
        features = inputs

        if unsupervised_training:
            features = self.encoder(features, training=training)
            #features = self.projector(features, training=training)
            if online:
                features = self.predictor(features, training=training)
        else:
            features = self.encoder(features, training=training)
            #features = self.projector(features, training=training)
            features = tf.nn.softmax(self.classifier(features))

        return features

    # @tf.function
    def call_representation(self, inputs, training=False):
        # connect layers here
        features = inputs
        features = self.encoder(features, training=training)

        return features
