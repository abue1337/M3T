import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import gin
from model.augmentations.auto_augment import distort_image_with_autoaugment
from model.augmentations.simclr_augment import distort_simclr
from model.augmentations.simclr_augment import resize_only

@gin.configurable
def gen_pipeline_train(ds_name='mnist',
                       tfds_path='~/tensorflow_datasets',
                       size_batch=64,
                       meta_batch_size=4,
                       b_shuffle=True,
                       size_buffer_cpu=5,
                       shuffle_buffer_size=0,
                       dataset_cache=False,
                       augmentation='autoaug',
                       num_parallel_calls=10,
                       split=tfds.Split.TRAIN,
                       finetuning=False,
                       validation_set=False):
    # Load and prepare tensorflow dataset
    data, info = tfds.load(name=ds_name,
                           data_dir=tfds_path,
                           split=split,
                           shuffle_files=False,
                           with_info=True)

    @tf.function
    def _map_data(*args):
        image = args[0]['image']
        label = args[0]['label']
        label = tf.one_hot(label, info.features['label'].num_classes)

        # Cast image type and normalize to 0/1
        if validation_set:
            image = tf.cast(image, tf.float32) / 255.0
        if finetuning:
            image = tf.cast(image, tf.float32) / 255.0
            if augmentation == 'simclr':
                image = resize_only(image)
        return image, label

    @tf.function
    def _map_augment_cifar10(*inputs):
        image, label = inputs

        image = tf.image.random_flip_left_right(image)
        IMG_SIZE = image.shape[0]  # CIFAR10: 32
        # Add 4 pixels of padding
        image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 4, IMG_SIZE + 4)
        # Random crop back to the original size
        image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])

        return image, label

    @tf.function
    def _map_augment_cifar10(*inputs):
        image, label = inputs

        image = tf.image.random_flip_left_right(image)
        IMG_SIZE = image.shape[0]  # CIFAR10: 32
        # Add 4 pixels of padding
        image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 4, IMG_SIZE + 4)
        # Random crop back to the original size
        image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])

        return image, label

    def _simple_augment(image):
        image = tf.cast(image, tf.float32) / 255.
        image = tf.image.random_flip_left_right(image)
        IMG_SIZE = image.shape[0]  # CIFAR10: 32
        # Add 4 pixels of padding
        image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 4, IMG_SIZE + 4)
        # Random crop back to the original size
        image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
        return image

    @tf.function
    @tf.autograph.experimental.do_not_convert
    def _map_augment(*args):
        image = args[0]
        label = args[1]
        if augmentation == 'autoaug':
            image1 = distort_image_with_autoaugment(image, 'cifar10')
            image2 = distort_image_with_autoaugment(image, 'cifar10')
            # image3 = distort_image_with_autoaugment(image, 'cifar10')
            image3 = _simple_augment(image)
            image1 = tf.cast(image1, tf.float32) / 255.0
            image2 = tf.cast(image2, tf.float32) / 255.0
            image3 = tf.cast(image3, tf.float32) / 255.0
        elif augmentation == 'simclr':
            image1 = distort_simclr(image)
            image2 = distort_simclr(image)
            # image3 = distort_simclr(image)
            image3 = _simple_augment(image)

        return image1, image2, image3, label


    #@tf.function
    def _batch_specific_aug(*args):
        #TODO: Add contrast?

        stddev = tf.random.uniform((1,), 0, 0.02, dtype=tf.float32)
        flip_ud = tf.random.uniform([], 0, 1)
        flip_rl = tf.random.uniform([], 0, 1)
        blur = tf.random.uniform([], 0, 1)
        brightness = tf.random.uniform((1,), -0.2, 0.2, dtype=tf.float32)
        img1 = batch_augmentation(args[0], stddev, flip_ud, flip_rl, brightness, blur)
        img2 = batch_augmentation(args[1], stddev, flip_ud, flip_rl, brightness, blur)
        img3 = batch_augmentation(args[2], stddev, flip_ud, flip_rl, brightness, blur)
        label = args[3]
        return img1, img2, img3, label

    # Map data
    dataset = data.map(map_func=_map_data, num_parallel_calls=num_parallel_calls)

    # Cache data
    if dataset_cache:
        dataset = dataset.cache()

    # Shuffle data
    if b_shuffle:
        if shuffle_buffer_size == 0:
            shuffle_buffer_size = info.splits['train'].num_examples
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    # Map Augmentation
    if not validation_set:
        if not finetuning:
            dataset = dataset.map(map_func=_map_augment, num_parallel_calls=num_parallel_calls)
        if finetuning and split == 'train':
            dataset = dataset.map(map_func=_map_augment_cifar10, num_parallel_calls=num_parallel_calls)

    # Batching
    dataset = dataset.batch(batch_size=size_batch,
                            drop_remainder=True)  # > 1.8.0: use drop_remainder=True

    if not validation_set:
        dataset = dataset.map(map_func=_batch_specific_aug, num_parallel_calls=num_parallel_calls)

    if not validation_set:
        dataset = dataset.batch(batch_size=meta_batch_size,
                                drop_remainder=True)  # > 1.8.0: use drop_remainder=True
    if size_buffer_cpu > 0:
        dataset = dataset.prefetch(buffer_size=size_buffer_cpu)

    return dataset, info


@gin.configurable
def gen_pipeline_test_time(ds_name='mnist',
                           tfds_path='~/tensorflow_datasets',
                           size_batch=1,
                           b_shuffle=True,
                           size_buffer_cpu=5,
                           shuffle_buffer_size=0,
                           dataset_cache=False,
                           augmentation='simclr',
                           num_parallel_calls=10,
                           split='test',
                           image_repetition=True
                           ):
    # Load and prepare tensorflow dataset
    data, info = tfds.load(name=ds_name,
                           data_dir=tfds_path,
                           split=split,
                           shuffle_files=False,
                           with_info=True)

    @tf.function
    def _map_data_im_rep(*args):
        image = tf.expand_dims(args[0]['image'], 0)
        image = tf.repeat(image, repeats=size_batch, axis=0)

        label = tf.one_hot(args[0]['label'], info.features['label'].num_classes)
        label = tf.repeat(tf.expand_dims(label, 0), repeats=size_batch, axis=0)
        return image, label

    @tf.function
    def _map_data(*args):
        image = args[0]['image']
        label = args[0]['label']
        label = tf.one_hot(label, info.features['label'].num_classes)

        return image, label

    @tf.function
    @tf.autograph.experimental.do_not_convert
    def _map_test_time_augment(*args):
        image = args[0]
        label = args[1]
        if augmentation == 'autoaug':
            image1 = distort_image_with_autoaugment(image, 'cifar10')
            image2 = distort_image_with_autoaugment(image, 'cifar10')
            image1 = tf.cast(image1, tf.float32) / 255.0
            image2 = tf.cast(image2, tf.float32) / 255.0
            image3 = tf.cast(image, tf.float32) / 255.0

        elif augmentation == 'simclr':
            image1 = distort_simclr(image)
            image2 = distort_simclr(image)
            image3 = tf.cast(image, tf.float32) / 255.0

        return image1, image2, image3, label


    # Map data
    if image_repetition is True:
        dataset = data.map(map_func=_map_data_im_rep, num_parallel_calls=num_parallel_calls)
        dataset = dataset.unbatch()
    else:
        dataset = data.map(map_func=_map_data, num_parallel_calls=num_parallel_calls)

    # Cache data
    if dataset_cache:
        dataset = dataset.cache()
    if image_repetition is False:
        # Shuffle data
        if b_shuffle:
            if shuffle_buffer_size == 0:
                shuffle_buffer_size = info.splits['train'].num_examples
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
    dataset = dataset.map(map_func=_map_test_time_augment, num_parallel_calls=num_parallel_calls)
    # Batching
    dataset = dataset.batch(batch_size=size_batch,
                            drop_remainder=True)  # > 1.8.0: use drop_remainder=True

    return dataset, info

@gin.configurable
def batch_augmentation(image, stddev, flip_ud, flip_rl, brightness, blur, flip_ud_rate, flip_rl_rate, blur_rate):
    # image must be scaled in [0, 1]

    # flip up/down
    if flip_ud < flip_ud_rate:
        image = tf.image.flip_up_down(image)

    # flip right/left
    if flip_rl < flip_rl_rate:
        image = tf.image.flip_left_right(image)

    # gaussian blur
    if blur < blur_rate:
        image = tfa.image.filters.gaussian_filter2d(image)

    # brightness
    image = tf.image.adjust_brightness(image, brightness)

    # gaussian noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=stddev, dtype=tf.float32)  # stddev=0.01
    noise_img = tf.add(image, noise)
    noise_img = tf.clip_by_value(noise_img, 0.0, 1.0)
    return noise_img
