import logging
import tensorflow as tf


def set_loggers(path_log=None, logging_level=0, b_stream=True, b_debug=False):

    # std. logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    # tf logger
    logger_tf = tf.get_logger()
    logger_tf.setLevel(logging_level)

    if path_log:
        file_handler = logging.FileHandler(path_log)
        logger.addHandler(file_handler)
        logger_tf.addHandler(file_handler)

    # plot to console
    if b_stream:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

    if b_debug:
        tf.debugging.set_log_device_placement(False)


# def save_dict_to_yaml(d, path_yaml):
#     with open(path_yaml, 'w') as file:
#         # We need to convert the values to float for yaml (it doesn't accept np.array, np.float, )
#         # TODO: check if this is the case for yaml
#         d = {k: float(v) for k, v in d.items()}
#         yaml.dump(d, file, indent=4)
#
#
# def plot_dataset(ds):
#     import matplotlib
#     matplotlib.use('TkAgg')  # qt may not work on server
#     from matplotlib import pyplot as plt
#     for images, labels in ds:
#         plt.figure()
#         plt.imshow(images[0])
#         plt.show()