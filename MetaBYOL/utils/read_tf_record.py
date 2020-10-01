import numpy as np
import tensorflow as tf




def read_tf_record(path):
    """
    Reads and parses TFRecord files

    Args:
        - path: String containing path to TFRecord file
    Returns:
        Parsed dataset
    """

    tfrecord_dataset = tf.data.TFRecordDataset(path)
    parsed_dataset = tfrecord_dataset.map(unserialize_tf_record)

    return parsed_dataset

def unserialize_tf_record(serialized_example):
    """
    Parses serialized example message w.r.t. to 'data' and 'labels'.

    Args:
        - serialized_example: Byte-string of TFRecords dataset
    Returns:
        - data: 3-D Tensor of shape [batch-size, window-size, sensor_type]
        - labels 3-D Tensor of shape [batch_size, window_size, label]
    """

    feature_description = {

        'data': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'labels': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    data = tf.io.parse_tensor(example['data'], out_type=tf.float32)
    labels = tf.io.parse_tensor(example['labels'], out_type=tf.float32)
    return data, labels
