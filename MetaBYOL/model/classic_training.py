import os
import logging
import tensorflow as tf
import tensorflow.keras as ks
import gin
import sys
from utils import utils_params
import random
from model import input_fn, model_fn


@gin.configurable(blacklist=['model', 'train_ds', 'run_paths'])
def train_classic(model,
                  train_ds,
                  run_paths,
                  learning_rate=0.001,
                  save_period=1,
                  epochs=1):
    model = model()
    loss_fc = tf.keras.losses.MeanSquaredError()

    # Generate summary writer
    writer = tf.summary.create_file_writer(os.path.dirname(run_paths['path_logs_train']))
    logging.info(f"Saving log to {os.path.dirname(run_paths['path_logs_train'])}")

    # Define optimizer
    optimizer = ks.optimizers.Adam(learning_rate=learning_rate)

    # Define checkpoints and checkpoint manager
    # manager automatically handles model reloading if directory contains ckpts
    ckpt = tf.train.Checkpoint(net=model, opt=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=run_paths['path_ckpts_train'],
                                              max_to_keep=2, keep_checkpoint_every_n_hours=1)
    ckpt.restore(ckpt_manager.latest_checkpoint)

    if ckpt_manager.latest_checkpoint:
        logging.info(f"Restored from {ckpt_manager.latest_checkpoint}.")
        epoch_start = int(os.path.basename(ckpt_manager.latest_checkpoint).split('-')[1]) + 1
    else:
        logging.info("Initializing from scratch.")
        epoch_start = 0

    # Define Metrics
    metric_loss_train = ks.metrics.Mean()

    logging.info(f"Classic training from epoch {epoch_start + 1} to {epochs}.")
    # use tf variable for epoch passing - so no new trace is triggered
    # if using normal range (instead of tf.range) assign a epoch_tf tensor, otherwise function gets recreated every turn
    epoch_tf = tf.Variable(1, dtype=tf.int32)

    total_loss = 0
    losses = []
    # Meta-Training
    for epoch in range(epoch_start, int(epochs)):
        # assign tf variable, graph build doesn't get triggered again
        epoch_tf.assign(epoch)
        logging.info(f"Epoch {epoch + 1}/{epochs}: starting training.")

        for i, sine_generator in enumerate(train_ds):
            x, y = sine_generator.batch()
            train_loss = train_step(model, x, y, optimizer, metric_loss_train, loss_fc, epoch)
            total_loss += train_loss
            moving_avg_losses = total_loss / (epoch + 1.0)
            losses.append(moving_avg_losses)
        # Print summary
        if epoch <= 0:
            model.summary()

        # Fetch metrics
        logging.info(f"Epoch {epoch + 1}/{epochs}: fetching metrics.")
        loss_train_avg = metric_loss_train.result()

        # Log to tensorboard
        with writer.as_default():
            tf.summary.scalar('loss_train_average', loss_train_avg, step=epoch)

        # Reset metrics after each epoch
        metric_loss_train.reset_states()

        logging.info(f'Epoch {epoch + 1}/{epochs}: loss_train_average: {loss_train_avg}')

        # Saving checkpoints
        if epoch % save_period == 0:
            logging.info(f'Saving checkpoint to {run_paths["path_ckpts_train"]}.')
            ckpt_manager.save(checkpoint_number=epoch)
        # write config after everything has been established
        """
        if epoch <= 0:
            gin_string = gin.operative_config_str()
            logging.info(f'Fetched config parameters: {gin_string}.')
            utils_params.save_gin(run_paths['path_gin'], gin_string)
        """
    print("Finished Classic Training")
    return model


@tf.function
def train_step(model1, image, label, optimizer, metric_loss_train, loss_function, epoch_tf):
    logging.info(f'Trace indicator - train epoch - eager mode: {tf.executing_eagerly()}.')
    with tf.device('/gpu:*'):
        with tf.GradientTape() as tape:
            features, h = model1(image, training=True)
            loss = loss_function(h, label)
        gradients = tape.gradient(loss, model1.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model1.trainable_variables))
    # Update metrics
    metric_loss_train.update_state(loss)
    # tf.print("Training loss for epoch:", epoch_tf + 1, " and step: ", optimizer.iterations, " - ", loss,
    #         output_stream=sys.stdout)
    return loss
