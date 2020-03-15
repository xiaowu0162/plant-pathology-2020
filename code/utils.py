import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import tensorflow as tf
import numpy as np
import random
import pathlib
from PIL import Image
import csv

AUTOTUNE = tf.data.experimental.AUTOTUNE


def show_jpg(image_path):
    im = Image.open(image_path)
    im.show()


def get_labels(csv_path):
    labels = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                labels.append([int(row[1]), int(row[2]), int(row[3]), int(row[4]),])
            except:
                continue
    return labels


def load_and_preprocess_image(image_path, image_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image /= 255.0  # normalize to [0,1] range
    return image


def augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if label is None:
        return image
    else:
        return image, label


def get_train_valid_dataset(train_path, image_count, image_size, batch_size=32, valid_split_rate=0.2, seed=42):
    train_dir = pathlib.Path(train_path)
    all_image_paths = list(train_dir.glob('*.jpg'))
    all_image_paths = [str(p) for p in all_image_paths]
    random.shuffle(all_image_paths)
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

    image_ds = path_ds.map((lambda x : load_and_preprocess_image(x, image_size)), num_parallel_calls=AUTOTUNE)
    labels = get_labels(str(train_dir) + '\\train.csv')
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds)).shuffle(buffer_size=image_count)

    train_ds = image_label_ds.skip(int(valid_split_rate * image_count))\
                            .map(augment, num_parallel_calls=AUTOTUNE)\
                            .repeat()\
                            .shuffle(buffer_size=int((1-valid_split_rate) * image_count), reshuffle_each_iteration=True)\
                            .batch(batch_size)\
                            .prefetch(buffer_size=AUTOTUNE)

    valid_ds = image_label_ds.take(int(valid_split_rate * image_count))\
                            .shuffle(buffer_size=int(valid_split_rate * image_count))\
                            .batch(batch_size)\
                            .cache()\
                            .prefetch(buffer_size=AUTOTUNE)

    return train_ds, valid_ds


def get_test_dataset(test_path, image_size, batch_size=None):
    test_dir = pathlib.Path(test_path)
    all_image_paths = list(test_dir.glob('*.jpg'))
    all_image_paths = [str(p) for p in all_image_paths]

    image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)\
                            .map((lambda x : load_and_preprocess_image(x, image_size)), num_parallel_calls=AUTOTUNE)
    if batch_size is not None:
        image_ds = image_ds.batch(batch_size)

    return image_ds
