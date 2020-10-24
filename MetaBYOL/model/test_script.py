import os
import logging
import tensorflow as tf
import gin


@gin.configurable(blacklist=['ds_test', 'target_model', 'run_paths'])
def test(ds_test,
         target_model,
         update_model,
         run_paths,
         num_test_time_steps,
         test_lr
         ):
    losses = list()
    accuracies = list()

    target_model(tf.zeros(shape=ds_test._flat_shapes[0][0:]))
    update_model(tf.zeros(shape=ds_test._flat_shapes[0][0:]))

    ckpt = tf.train.Checkpoint(net=target_model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=run_paths['path_ckpts_train'],
                                              max_to_keep=2, keep_checkpoint_every_n_hours=1)
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    if ckpt_manager.latest_checkpoint:
        logging.info(f"Restored from {ckpt_manager.latest_checkpoint}.")
        epoch_start = int(os.path.basename(ckpt_manager.latest_checkpoint).split('-')[1]) + 1
    else:
        logging.info("Initializing from scratch.")

    test_loss, test_accuracy = define_metrics_test()
    meta_test_optimizer = tf.optimizers.SGD(learning_rate=test_lr)

    grad_steps_tf = tf.Variable(1, dtype=tf.int32)
    for i in range(num_test_time_steps+1):
        grad_steps_tf.assign(i)
        for im1, im2, test_im, test_label in ds_test:
            update_model.set_weights(target_model.get_weights())  # TODO: check if ckpt.restore or
            inner_loop(im1, im2, test_im, test_label, target_model, update_model, meta_test_optimizer, test_loss,
                       test_accuracy, grad_steps_tf)
        logging.info(
            f"Test acc after {i} gradient steps: {test_accuracy.result()} Test loss after "
            f"{i} gradient steps:{test_loss.result()}")
        losses.append(test_loss.result().numpy())
        accuracies.append(test_accuracy.result().numpy())
        test_loss.reset_states()
        test_accuracy.reset_states()
    return accuracies, losses


#@tf.function
def inner_loop(im1, im2, test_im, test_label, target_model, update_model, optimizer, test_loss, test_accuracy, grad_steps):
    tar1 = target_model(im1, training=True, unsupervised_training=True)
    tar2 = target_model(im2, training=True, unsupervised_training=True)
    for k in range(grad_steps + 1):
        with tf.GradientTape(persistent=False) as test_tape:
            test_tape.watch(update_model.trainable_variables)
            prediction1 = update_model(im1, training=True, unsupervised_training=True,
                                        online=True)
            prediction2 = update_model(im2, training=True, unsupervised_training=True,
                                        online=True)
            loss1 = byol_loss_fn(prediction1, tf.stop_gradient(tar2))
            loss2 = byol_loss_fn(prediction2, tf.stop_gradient(tar1))
            loss = tf.reduce_mean(loss1 + loss2)
        gradients = test_tape.gradient(loss, update_model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        optimizer.apply_gradients(zip(gradients, update_model.trainable_variables))
    test_prediction = update_model(test_im, training=False, unsupervised_training=False)
    loss = loss_function(test_label, test_prediction)
    test_loss(loss)
    test_accuracy(test_label, test_prediction)


def loss_function(labels, predictions):
    return tf.keras.losses.categorical_crossentropy(labels, predictions)


def byol_loss_fn(x, y):
    x = tf.math.l2_normalize(x, axis=-1)
    y = tf.math.l2_normalize(y, axis=-1)
    return 2 - 2 * tf.math.reduce_sum(x * y, axis=-1)


def define_metrics_test():
    test_loss = tf.keras.metrics.Mean(name='test_loss_')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy_')
    return test_loss, test_accuracy
