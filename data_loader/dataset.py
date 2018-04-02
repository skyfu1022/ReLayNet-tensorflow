import tensorflow as tf
from utils.utils_2 import read_data
import numpy as np
import os
import functools


def random_flip(images, labels, contours):
    border = tf.get_seed(0)
    seed = np.random.randint(border[1], border[0])
    images = tf.image.random_flip_left_right(images, seed)
    labels = tf.image.random_flip_left_right(labels, seed)
    contours = tf.image.random_flip_left_right(contours, seed)
    return images, labels, contours


def random_crop(images, labels, contours, shape):
    border = tf.get_seed(0)
    seed = np.random.randint(border[1], border[0])
    images = tf.random_crop(images, shape, seed)
    labels = tf.random_crop(labels, shape, seed)
    contours = tf.random_crop(contours, shape, seed)
    return images, labels, contours


def parse_function(images, labels, contours, resize=True, shape=None):
    images_string = tf.read_file(images)
    images_decoded = tf.image.decode_image(images_string)
    if resize:
        images_resized = tf.image.resize_image_with_crop_or_pad(images_decoded, shape[0], shape[1])
    else:
        images_resized = tf.reshape(images_decoded, shape)

    labels_string = tf.read_file(labels)
    labels_decoded = tf.image.decode_image(labels_string)
    if resize:
        labels_resized = tf.image.resize_image_with_crop_or_pad(labels_decoded, shape[0], shape[1])
    else:
        labels_resized = tf.reshape(labels_decoded, shape)

    contours_string = tf.read_file(contours)
    contours_decoded = tf.image.decode_image(contours_string)
    if resize:
        contours_resized = tf.image.resize_image_with_crop_or_pad(contours_decoded, shape[0], shape[1])
    else:
        contours_resized = tf.reshape(contours_decoded, shape)
    # value_decode = tf.reshape(value_decode, shape)
    return images_resized, labels_resized, contours_resized


def unlabeled_parse_function(data):
    image_string = tf.read_file(data)
    image_decoded = tf.image.decode_image(image_string)
    return image_decoded


class RetinalDataset:
    def __init__(self, config):
        self.config = config
        data_path = os.path.join(self.config.data_path, self.config.data_set)
        save_path = os.path.join('data', self.config.data_set)
        self.train_data = read_data(save_path, data_path, 'train')
        self.num_train_samples = len(self.train_data['images'])
        self.val_data = read_data(save_path, data_path, 'val')
        self.num_val_samples = len(self.val_data['images'])
        self.test_data = read_data(save_path, data_path, 'test')
        self.num_test_samples = len(self.test_data['images'])
        self._unlabeled_data = read_data(save_path, data_path, 'unlabeled')

    def get_batch(self, mode):
        if mode is 'train':
            parse = functools.partial(parse_function, resize=True, shape=(512, 80, 1))
            crop = functools.partial(random_crop, shape=(496, 64, 1))
            data = self.train_data
        elif mode is 'val':
            parse = functools.partial(parse_function, resize=False, shape=(496, 64, 1))
            # crop = functools.partial(random_crop, shape=(496, 64, 1))
            data = self.val_data
        elif mode is 'test':
            parse = functools.partial(parse_function, shape=(496, 512, 1))
            data = self.test_data
        else:
            raise ValueError
        images = data['images']
        filenames_image = tf.constant(images)

        labels = data['layers']
        filenames_label = tf.constant(labels)

        contours = data['contours']
        filenames_contours = tf.constant(contours)
        # data_dict = {'image': filenames_image, 'label': filenames_label, 'contour': filenames_contour}

        dataset = tf.data.Dataset.from_tensor_slices((filenames_image, filenames_label, filenames_contours))
        dataset = dataset.map(parse)

        if self.config.augmentation and mode is 'train':
            dataset = dataset.map(random_flip)
            dataset = dataset.map(crop)
        if self.config.shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.config.batch_size).repeat(self.config.num_epochs)
        iterator = dataset.make_one_shot_iterator()
        next_batch = iterator.get_next()
        image_batch = next_batch[0]
        label_batch = next_batch[1]
        contour_batch = next_batch[2]
        return image_batch, label_batch, contour_batch

