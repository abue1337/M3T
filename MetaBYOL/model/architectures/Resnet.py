import gin
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import models
import tensorflow_addons as tfa


@gin.configurable('Resnet')
class Architecture(models.Model):
    """ResNet for CIFAR10 dataset."""
    "Adapted from: https://github.com/chao-ji/tf-resnet-cifar10/blob/master/v2/model.py"

    def __init__(self,
                 n_classes,
                 num_layers=20,
                 num_initial_filters=16,
                 shortcut_connection=True,
                 weight_decay=2e-4,
                 batch_norm_momentum=0.99,
                 batch_norm_epsilon=1e-3,
                 batch_norm_center=True,
                 batch_norm_scale=True,
                 group_norm_groups=16):
        """Constructor.
        Args:
          num_layers: int scalar, num of layers.
          shortcut_connection: bool scalar, whether to add shortcut connection in
            each Resnet unit. If False, degenerates to a 'Plain network'.
          weight_decay: float scalar, weight for l2 regularization.
          batch_norm_momentum: float scalar, the moving avearge decay.
          batch_norm_epsilon: float scalar, small value to avoid divide by zero.
          batch_norm_center: bool scalar, whether to center in the batch norm.
          batch_norm_scale: bool scalar, whether to scale in the batch norm.
        """
        super(Architecture, self).__init__(name='ResNetFelix')
        if num_layers not in (20, 32, 44, 56, 110):
            raise ValueError('num_layers must be one of 20, 32, 44, 56 or 110.')

        self._num_layers = num_layers
        self._num_initial_filters = num_initial_filters
        self._shortcut_connection = shortcut_connection
        self._weight_decay = weight_decay
        self._batch_norm_momentum = batch_norm_momentum
        self._batch_norm_epsilon = batch_norm_epsilon
        self._batch_norm_center = batch_norm_center
        self._batch_norm_scale = batch_norm_scale
        self._group_norm_groups = group_norm_groups

        self._num_units = (num_layers - 2) // 6

        self._kernel_regularizer = regularizers.l2(weight_decay)

        self._init_conv = layers.Conv2D(self._num_initial_filters, 3, 1, 'same', use_bias=False,
                                        kernel_regularizer=self._kernel_regularizer, name='init_conv')

        self._block1 = models.Sequential([ResNetUnit(
            self._num_initial_filters,
            1,
            shortcut_connection,
            True if i == 0 else False,
            weight_decay,
            batch_norm_momentum,
            batch_norm_epsilon,
            batch_norm_center,
            batch_norm_scale,
            group_norm_groups,
            'res_net_unit_%d' % (i + 1)) for i in range(self._num_units)],
            name='block1')
        self._block2 = models.Sequential([ResNetUnit(
            self._num_initial_filters * 2,
            2 if i == 0 else 1,
            shortcut_connection,
            False if i == 0 else False,
            weight_decay,
            batch_norm_momentum,
            batch_norm_epsilon,
            batch_norm_center,
            batch_norm_scale,
            group_norm_groups,
            'res_net_unit_%d' % (i + 1)) for i in range(self._num_units)],
            name='block2')
        self._block3 = models.Sequential([ResNetUnit(
            self._num_initial_filters * 4,
            2 if i == 0 else 1,
            shortcut_connection,
            False if i == 0 else False,
            weight_decay,
            batch_norm_momentum,
            batch_norm_epsilon,
            batch_norm_center,
            batch_norm_scale,
            group_norm_groups,
            'res_net_unit_%d' % (i + 1)) for i in range(self._num_units)],
            name='block3')

        self._global_avg = tf.keras.layers.GlobalAveragePooling2D(name="GlobalAvgPooling")
        self._last_bn = tfa.layers.GroupNormalization(groups=group_norm_groups,
                                                             axis=-1,
                                                             epsilon=batch_norm_epsilon,
                                                             center=batch_norm_center,
                                                             scale=batch_norm_scale)
        # self._last_conv = layers.Conv2D(self._num_initial_filters*4*4, 1, 1, 'same', use_bias=False,
        #                                 kernel_regularizer=self._kernel_regularizer, name='init_conv') # Last conv has 4 times the filters of layer before
        # self._last_bn2 = layers.BatchNormalization(-1,
        #                                           batch_norm_momentum,
        #                                           batch_norm_epsilon,
        #                                           batch_norm_center,
        #                                           batch_norm_scale,
        #                                           name='batchnorm_20')
        # self._mlp_dense_out = tf.keras.layers.Dense(n_classes, use_bias=False, name="Head_Dense",activation='softmax')

    def call(self, inputs, training=False):
        """Execute the forward pass.
        Args:
          inputs: float tensor of shape [batch_size, 32, 32, 3], the preprocessed,
            data-augmented, and batched CIFAR10 images.
        Returns:
          logits: float tensor of shape [batch_size, 10], the unnormalized logits.
        """
        h = inputs
        h = self._init_conv(h)

        h = self._block1(h, training=training)
        h = self._block2(h, training=training)
        h = self._block3(h, training=training)
        h = tf.nn.relu(self._last_bn(h))
        # h = tf.nn.relu(self._last_bn2(self._last_conv(h),training=training))
        h = self._global_avg(h)

        # For simclr v2:
        # g = self._mlp_dense_out(h)

        return h  # , g


class ResNetUnit(models.Model):
    """A ResNet Unit contains two conv2d layers interleaved with Batch
    Normalization and ReLU.
    """

    def __init__(self,
                 depth,
                 stride,
                 shortcut_connection,
                 shortcut_from_preact,
                 weight_decay,
                 batch_norm_momentum,
                 batch_norm_epsilon,
                 batch_norm_center,
                 batch_norm_scale,
                 group_norm_groups,
                 name):
        """Constructor.
        Args:
          depth: int scalar, the depth of the two conv ops in each Resnet unit.
          stride: int scalar, the stride of the first conv op in each Resnet unit.
          shortcut_connection: bool scalar, whether to add shortcut connection in
            each Resnet unit. If False, degenerates to a 'Plain network'.
          shortcut_from_preact: bool scalar, whether the shortcut connection starts
            from the preactivation or the input feature map.
          weight_decay: float scalar, weight for l2 regularization.
          batch_norm_momentum: float scalar, the moving average decay.
          batch_norm_epsilon: float scalar, small value to avoid divide by zero.
          batch_norm_center: bool scalar, whether to center in the batch norm.
          batch_norm_scale: bool scalar, whether to scale in the batch norm.
        """
        super(ResNetUnit, self).__init__(name=name)
        self._depth = depth
        self._stride = stride
        self._shortcut_connection = shortcut_connection
        self._shortcut_from_preact = shortcut_from_preact
        self._weight_decay = weight_decay

        self._kernel_regularizer = regularizers.l2(weight_decay)

        self._bn1 = tfa.layers.GroupNormalization(groups=group_norm_groups,
                                                         axis=-1,
                                                         epsilon=batch_norm_epsilon,
                                                         center=batch_norm_center,
                                                         scale=batch_norm_scale)
        self._conv1 = layers.Conv2D(depth,
                                    3,
                                    stride,
                                    'same',
                                    use_bias=True,
                                    kernel_regularizer=self._kernel_regularizer,
                                    name='conv1')
        self._bn2 = tfa.layers.GroupNormalization(groups=group_norm_groups,
                                                         axis=-1,
                                                         epsilon=batch_norm_epsilon,
                                                         center=batch_norm_center,
                                                         scale=batch_norm_scale)
        self._conv2 = layers.Conv2D(depth,
                                    3,
                                    1,
                                    'same',
                                    use_bias=True,
                                    kernel_regularizer=self._kernel_regularizer,
                                    name='conv2')

    def call(self, inputs, training=False):
        """Execute the forward pass.
        Args:
          inputs: float tensor of shape [batch_size, height, width, depth], the
            input tensor.
        Returns:
          outouts: float tensor of shape [batch_size, out_height, out_width,
            out_depth], the output tensor.
        """
        depth_in = inputs.shape[3]  # depth_in = num_initial_filters
        depth = self._depth
        preact = tf.nn.relu(self._bn1(inputs))

        shortcut = preact if self._shortcut_from_preact else inputs

        if depth != depth_in:
            shortcut = tf.nn.avg_pool2d(
                shortcut, (2, 2), strides=(1, 2, 2, 1), padding='SAME')
            shortcut = tf.pad(
                shortcut, [[0, 0], [0, 0], [0, 0], [(depth - depth_in) // 2] * 2])

        residual = tf.nn.relu(self._bn2(self._conv1(preact)))
        residual = self._conv2(residual)

        outputs = residual + shortcut if self._shortcut_connection else residual

        return outputs
