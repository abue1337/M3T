import os
import logging
import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
import gin
import time
import sys
# from utils import utils_params

import matplotlib.pyplot as plt


# from model import input_fn, model_fn
@gin.configurable(blacklist=['target_model', 'shape', 'test_time'])
class MAML():

    def __init__(self, target_model, shape, num_steps_ml, lr_inner_ml, lr_test_time, num_test_time_steps,
                 test_time=False):

        self.target_model = target_model()
        self.num_steps_ml = num_steps_ml
        self.lr_inner_ml = lr_inner_ml

        self.test_time = test_time
        self.lr_test_time = lr_test_time
        self.num_test_time_steps = num_test_time_steps

        self.input_shape = shape
        self.target_model(tf.zeros(shape=self.input_shape))
        self.target_model(tf.zeros(shape=self.input_shape), unsupervised_training=True, online=True)
        self.updated_models = list()
        self.test_model = target_model()
        for _ in range(self.num_steps_ml + 1):
            updated_model = target_model()
            updated_model(tf.zeros(shape=self.input_shape))
            updated_model(tf.zeros(shape=self.input_shape), unsupervised_training=True, online=True)
            self.updated_models.append(updated_model)
        self.inner_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_inner_ml)

    @gin.configurable(blacklist=['ds_train', 'ds_val', 'run_paths'])
    def train(self,
              ds_train,
              ds_val,
              run_paths,
              n_meta_epochs=10,
              meta_batch_size=4,
              meta_learning_rate=0.001,
              optimizer='Adam',
              save_period=20
              ):

        # Generate summary writer
        meta_train_loss, meta_train_accuracy, meta_val_loss, meta_val_accuracy = define_metrics()
        writer = tf.summary.create_file_writer(os.path.dirname(run_paths['path_logs_train']))
        logging.info(f"Saving log to {os.path.dirname(run_paths['path_logs_train'])}")

        # Define optimizer
        if optimizer == 'Adam':
            meta_optimizer = tf.keras.optimizers.Adam(learning_rate=meta_learning_rate)

        elif optimizer == 'SGD':
            overall_batchsize = ds_train._flat_shapes[0][0] * ds_train._flat_shapes[0][1]
            steps_per_epoch = round(50000 / overall_batchsize)
            learning_rate_schedule = ks.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=[steps_per_epoch, steps_per_epoch * 120, steps_per_epoch * 150],
                values=[meta_learning_rate / 10,
                        meta_learning_rate,
                        meta_learning_rate / 10,
                        meta_learning_rate / 100])
            """learning_rate_schedule = ks.optimizers.schedules.PiecewiseConstantDecay(boundaries=[25, 2000, 3000],
                                                                                    values=[meta_learning_rate / 10,
                                                                                            meta_learning_rate,
                                                                                            meta_learning_rate / 10,
                                                                                            meta_learning_rate / 100])"""
            meta_optimizer = ks.optimizers.SGD(learning_rate=learning_rate_schedule, momentum=0.9)

        # Define checkpoints and checkpoint manager
        # manager automatically handles model reloading if directory contains ckpts
        ckpt = tf.train.Checkpoint(net=self.target_model, opt=meta_optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, directory=run_paths['path_ckpts_train'],
                                                  max_to_keep=2, keep_checkpoint_every_n_hours=1)
        ckpt.restore(ckpt_manager.latest_checkpoint)

        if ckpt_manager.latest_checkpoint:
            logging.info(f"Restored from {ckpt_manager.latest_checkpoint}.")
            epoch_start = int(os.path.basename(ckpt_manager.latest_checkpoint).split('-')[1]) + 1
        else:
            logging.info("Initializing from scratch.")
            epoch_start = 1

        # Define Metrics
        metric_loss_train = ks.metrics.Mean()

        logging.info(f"Training from epoch {epoch_start} to {n_meta_epochs}.")
        # use tf variable for epoch passing - so no new trace is triggered if using normal range (instead of
        # tf.range) assign a epoch_tf tensor, otherwise function gets recreated every turn
        epoch_tf = tf.Variable(1, dtype=tf.int32)

        # Meta-Training
        for epoch in range(epoch_start, int(n_meta_epochs + 1)):

            # assign tf variable, graph build doesn't get triggered again
            epoch_tf.assign(epoch)
            logging.info(f"Epoch {epoch}/{n_meta_epochs}: starting training.")

            tf_meta_batch_size = tf.Variable(1, dtype=tf.int32)
            tf_meta_batch_size.assign(meta_batch_size)

            for train1_ep, train2_ep, test_ep, test_label_ep in ds_train:
                start = time.time()
                self.meta_train_step(train1_ep, train2_ep, test_ep, test_label_ep, meta_optimizer, tf_meta_batch_size,
                                     meta_batch_size, meta_train_accuracy, meta_train_loss)

                stop = time.time()
                # logging.info(f"Time for one iteration {stop - start}")
            for image, label in ds_val:
                self.meta_val_step(image, label, meta_val_loss, meta_val_accuracy)

            # Saving checkpoints
            if (epoch % save_period == 0) | (epoch == n_meta_epochs):
                logging.info(f'Saving checkpoint to {run_paths["path_ckpts_train"]}.')
                ckpt_manager.save(checkpoint_number=epoch)

            logging.info(f"Epoch {epoch}: acc = {meta_train_accuracy.result()}")
            logging.info(f"Epoch {epoch}: val_acc = {meta_val_accuracy.result()}")
            logging.info(f"Epoch {epoch}: loss = {meta_train_loss.result()}")
            with writer.as_default():
                tf.summary.scalar('Average meta test tasks accuracy', meta_train_accuracy.result(),
                                  step=epoch)
                tf.summary.scalar('Average meta test tasks loss', meta_train_loss.result(),
                                  step=epoch)
                tf.summary.scalar('Average meta val accuracy', meta_val_accuracy.result(),
                                  step=epoch)
                tf.summary.scalar('Average meta val loss', meta_val_loss.result(),
                                  step=epoch)
                tf.summary.scalar('Learning rate', learning_rate_schedule(meta_optimizer.iterations),
                                  step=epoch)

                reset_metrics(meta_train_loss, meta_train_accuracy, meta_val_loss, meta_val_accuracy)
        logging.info(f"Finished")

        return self.target_model

    @tf.function
    def func_help(self, train1_ep, train2_ep, test_ep, test_label_ep, meta_batch_size):

        tasks_final_losses, test_predictions = tf.map_fn(
            self.get_losses_of_tasks_batch,
            elems=(
                train1_ep,
                train2_ep,
                test_ep,
                test_label_ep
            ),
            dtype=(tf.float32, tf.float32),
            parallel_iterations=meta_batch_size
        )
        final_loss = tf.reduce_mean(tasks_final_losses)
        return final_loss, test_predictions

    def meta_train_step(self, train1_ep, train2_ep, test_ep, test_label_ep, meta_optimizer, tf_meta_batch_size,
                        meta_batch_size, meta_train_acc, meta_train_loss):
        with tf.GradientTape(persistent=False) as outer_tape:
            final_loss, test_predictions = self.func_help(train1_ep, train2_ep, test_ep, test_label_ep, meta_batch_size)
        outer_gradients = outer_tape.gradient(final_loss, self.target_model.trainable_variables)
        meta_optimizer.apply_gradients(zip(outer_gradients, self.target_model.trainable_variables))
        meta_train_acc(test_label_ep, test_predictions)
        meta_train_loss(final_loss)

    # @tf.function
    def get_losses_of_tasks_batch(self, inputs):
        train1_ep, train2_ep, test_ep, test_label_ep = inputs
        updated_model = self.inner_train_loop(train1_ep, train2_ep)
        test_prediction = updated_model(test_ep, training=True,
                                        unsupervised_training=False)  # TODO: training=True or False??
        test_loss = tf.reduce_mean(self.loss_function(test_label_ep, test_prediction))
        return test_loss, test_prediction

    def create_meta_model(self, updated_model, model, gradients):
        k = 0
        variables = list()
        model_layers = list(flatten(model.layers))
        updated_model_layers = list(flatten(updated_model.layers))
        if not self.test_time:
            lr = self.lr_inner_ml
        else:
            lr = self.lr_test_time
        gradients = [tf.zeros(1) if v is None else v for v in gradients]
        for i in range(len(model_layers)):
            if isinstance(model_layers[i], tf.keras.layers.Conv2D) or \
                    isinstance(model_layers[i], tf.keras.layers.Dense):
                updated_model_layers[i].kernel = model_layers[i].kernel - lr * gradients[k]
                k += 1
                variables.append(updated_model_layers[i].kernel)

                if not updated_model_layers[i].bias is None:
                    updated_model_layers[i].bias = model_layers[i].bias - lr * gradients[k]
                    k += 1
                    variables.append(updated_model_layers[i].bias)

            elif isinstance(model_layers[i], tf.keras.layers.BatchNormalization):
                if hasattr(model_layers[i], 'moving_mean') and model_layers[i].moving_mean is not None:
                    updated_model_layers[i].moving_mean.assign(model_layers[i].moving_mean)
                if hasattr(model_layers[i], 'moving_variance') and model_layers[i].moving_variance is not None:
                    updated_model_layers[i].moving_variance.assign(model_layers[i].moving_variance)
                if hasattr(model_layers[i], 'gamma') and model_layers[i].gamma is not None:
                    updated_model_layers[i].gamma = model_layers[i].gamma - lr * gradients[k]
                    k += 1
                    variables.append(updated_model_layers[i].gamma)
                if hasattr(model_layers[i], 'beta') and model_layers[i].beta is not None:
                    updated_model_layers[i].beta = \
                        model_layers[i].beta - lr * gradients[k]
                    k += 1
                    variables.append(updated_model_layers[i].beta)

            elif isinstance(model_layers[i], tf.keras.layers.LayerNormalization):
                if hasattr(model_layers[i], 'gamma') and model_layers[i].gamma is not None:
                    updated_model_layers[i].gamma = model_layers[i].gamma - lr * gradients[k]
                    k += 1
                    variables.append(updated_model_layers[i].gamma)
                if hasattr(model_layers[i], 'beta') and model_layers[i].beta is not None:
                    updated_model_layers[i].beta = \
                        model_layers[i].beta - lr * gradients[k]
                    k += 1
                    variables.append(updated_model_layers[i].beta)
        setattr(updated_model, 'meta_trainable_variables', variables)

    def inner_train_loop(self, train1_ep, train2_ep):

        gradients = list()
        for variable in self.target_model.trainable_variables:
            gradients.append(tf.zeros_like(variable))

        self.create_meta_model(self.updated_models[0], self.target_model,
                               gradients)

        tar1 = self.target_model(train1_ep, training=True, unsupervised_training=True)
        tar2 = self.target_model(train2_ep, training=True,
                                 unsupervised_training=True)
        for k in range(1, self.num_steps_ml + 1):
            with tf.GradientTape(persistent=False) as train_tape:
                train_tape.watch(self.updated_models[k - 1].trainable_variables)
                prediction1 = self.updated_models[k - 1](train1_ep, training=True, unsupervised_training=True,
                                                         online=True)
                prediction2 = self.updated_models[k - 1](train2_ep, training=True, unsupervised_training=True,
                                                         online=True)
                loss1 = self.byol_loss_fn(prediction1, tf.stop_gradient(tar2))
                loss2 = self.byol_loss_fn(prediction2, tf.stop_gradient(tar1))
                loss = tf.reduce_mean(loss1 + loss2)
            gradients = train_tape.gradient(loss, self.updated_models[k - 1].meta_trainable_variables)

            self.create_meta_model(self.updated_models[k], self.updated_models[k - 1], gradients)

        return self.updated_models[-1]

    def loss_function(self, labels, predictions):
        return tf.keras.losses.categorical_crossentropy(labels, predictions)

    def byol_loss_fn(self, x, y):
        x = tf.math.l2_normalize(x, axis=-1)
        y = tf.math.l2_normalize(y, axis=-1)
        return 2 - 2 * tf.math.reduce_sum(x * y, axis=-1)

    @tf.function
    def meta_val_step(self, images, labels, val_loss, val_acc):
        predictions = self.target_model(images, training=False,
                                        unsupervised_training=False)
        loss = self.loss_function(labels, predictions)
        val_loss(loss)
        val_acc(labels, predictions)

    def test(self,
             ds_test,
             run_paths
             ):
        meta_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        ckpt = tf.train.Checkpoint(net=self.target_model, opt=meta_optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, directory=run_paths['path_ckpts_train'],
                                                  max_to_keep=2, keep_checkpoint_every_n_hours=1)
        ckpt.restore(ckpt_manager.latest_checkpoint)
        if ckpt_manager.latest_checkpoint:
            logging.info(f"Restored from {ckpt_manager.latest_checkpoint}.")
            epoch_start = int(os.path.basename(ckpt_manager.latest_checkpoint).split('-')[1]) + 1
        else:
            logging.info("Initializing from scratch.")
            epoch_start = 1
        test_loss, test_accuracy = define_metrics_test()
        # num_steps = tf.Variable(0, dtype=tf.int32)
        # num_steps.assign(grad_steps)
        for im1, im2, test_im, test_label in ds_test:
            self.help_func_test(im1, im2, test_im, test_label, test_loss, test_accuracy)

        logging.info(f"Test acc after {self.num_test_time_steps} gradient steps: {test_accuracy.result()} Test loss after "
                     f"{self.num_test_time_steps} gradient steps:{test_loss.result()}")
        test_loss.reset_states()
        test_accuracy.reset_states()


    @tf.function
    def help_func_test(self, im1, im2, test_im, test_label, test_loss, test_accuracy):
        updated_model = self.test_time_loop(im1, im2)
        test_prediction = updated_model(test_im, training=False, unsupervised_training=False)
        loss = self.loss_function(test_label, test_prediction)
        test_loss(loss)
        test_accuracy(test_label, test_prediction)

    def test_time_loop(self, train1_ep, train2_ep):

        gradients = list()
        for variable in self.target_model.trainable_variables:
            gradients.append(tf.zeros_like(variable))

        self.create_meta_model(self.updated_models[0], self.target_model,
                               gradients)

        tar1 = self.target_model(train1_ep, training=False, unsupervised_training=True)
        tar2 = self.target_model(train2_ep, training=False,
                                 unsupervised_training=True)
        k=0
        for k in range(1, self.num_test_time_steps+1):
            with tf.GradientTape(persistent=False) as test_time_tape:
                test_time_tape.watch(self.updated_models[k - 1].trainable_variables)
                prediction1 = self.updated_models[k - 1](train1_ep, training=False, unsupervised_training=True,
                                                         online=True)
                prediction2 = self.updated_models[k - 1](train2_ep, training=False, unsupervised_training=True,
                                                         online=True)
                loss1 = self.byol_loss_fn(prediction1, tf.stop_gradient(tar2))
                loss2 = self.byol_loss_fn(prediction2, tf.stop_gradient(tar1))
                loss = tf.reduce_mean(loss1 + loss2)
            gradients = test_time_tape.gradient(loss, self.updated_models[k - 1].meta_trainable_variables)
            self.create_meta_model(self.updated_models[k], self.updated_models[k - 1], gradients)

        return self.updated_models[k]


def define_metrics_test():
    test_loss = tf.keras.metrics.Mean(name='test_loss_')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy_')
    return test_loss, test_accuracy


def flatten(l):
    for el in l:
        if hasattr(el, 'layers'):
            yield from flatten(el.layers)
        else:
            yield el


def define_metrics():
    """
    This function initializes metrics for training.

    Returns:
        train_loss: a tf.keras.losses-object
        train_accuracy: tf.keras.metrics.BinaryAccuracy-object
    """
    train_loss = tf.keras.metrics.Mean(name='meta_train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='meta_train_accuracy')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
    return train_loss, train_accuracy, val_loss, val_accuracy




def reset_metrics(train_loss, train_accuracy, val_loss, val_accuracy):
    """
    This function resets metrics after epoch during training.
    Args:
        train_loss: a tf.keras.losses-object
        train_accuracy: tf.keras.metrics.BinaryAccuracy-object
        val_loss: a tf.keras.losses-object
        val_accuracy: tf.keras.metrics.BinaryAccuracy-object
    Returns: Nothing
    """
    train_loss.reset_states()
    train_accuracy.reset_states()
    val_loss.reset_states()
    val_accuracy.reset_states()
