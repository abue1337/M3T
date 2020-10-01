import tensorflow as tf
import os
import logging
import gin


@gin.configurable
def set_devices(idx_gpu, soft_device_placement=True, num_cpu_threads=0):

    # Set jit active for XLA routines - Not functional yet
    # tf.config.optimizer.set_jit(True)

    os.unsetenv('CUDA_VISIBLE_DEVICES')  # to see the actual list, prior restrictions have to be unset
    gpus = tf.config.experimental.list_physical_devices('GPU')
    logging.info(f'Available physical devices: {gpus}.')
    if gpus:
        logging.info(f'Fetching devices: {idx_gpu}.')
        # Restrict TensorFlow to only use one specific GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[int(idx_gpu)], 'GPU')
            #  set memory growth option for all GPU devices (even the ones not used). Otherwise current tf version throws a ValueError
            for idx in range(len(gpus)):
                tf.config.experimental.set_memory_growth(gpus[int(idx)], True)
        except RuntimeError as e:
            # Visible devices and memory growth must be set at program startup
            print(e)

        # set device placement
        tf.config.set_soft_device_placement(soft_device_placement)

        logging.info(f'Visible devices: {tf.config.experimental.get_visible_devices()}.')
        logging.info(f'Memory growth option: {tf.config.experimental.get_memory_growth(gpus[int(idx_gpu)])}')
        logging.info(f'Using soft device placement: {tf.config.get_soft_device_placement()}')

    # Set CPU options
    if num_cpu_threads > 0:
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(num_cpu_threads)
    else:
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(4)

    # set some optimizations
    tf.config.optimizer.set_experimental_options({'layout_optimizer': True,
                                                  'remapping': True,
                                                  'arithmetic_optimization': True,
                                                  'dependency_optimization': True,
                                                  'loop_optimization': True,
                                                  'function_optimization': True,
                                                  'scoped_allocator_optimization': True})
    # report CPU optimizations
    logging.info(f'Using the following optimization options: {tf.config.optimizer.get_experimental_options()}.')