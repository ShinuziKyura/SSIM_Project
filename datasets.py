import numpy as np
import cv2 as cv
import tensorflow as tf


class ImageMaskDataset:
    def __init__(self, images, labels, image_shape=None, label_shape=None, size=None):
        assert images.shape[0] == labels.shape[0]

        self.images = images
        self.labels = labels

        self.image_shape = (image_shape if image_shape is not None else images.shape[1:])
        self.label_shape = (label_shape if label_shape is not None else labels.shape[1:])

        self.size = (size if size is not None else images.shape[0])


def create_dataset(images, masks, rescale=None):
    assert images.shape[0] > 0
    assert masks.shape[0] > 0

    print('Creating dataset...')

    scale_arr = list(images[0]['data'].shape[0:2])
    rescale_arr = list(rescale) if rescale is not None else scale_arr

    image_shape = rescale + images[0]['data'].shape[2:]
    label_shape = rescale + (1,)

    images_map = {}
    masks_map = {}

    if rescale is not None:
        if np.all(scale_arr < rescale_arr, 0):
            method = cv.INTER_CUBIC
        elif np.all(scale_arr > rescale_arr, 0):
            method = cv.INTER_AREA
        else:
            method = cv.INTER_LINEAR

        for image in images:
            images_map[image['name']] = cv.resize(image['data'], rescale, method)
        for mask in masks:
            masks_map[mask['name']] = cv.resize(mask['data'], rescale, method)
    else:
        for image in images:
            images_map[image['name']] = image['data']
        for mask in masks:
            masks_map[mask['name']] = mask['data']

    images = np.empty(shape=masks.shape[0:1] + image_shape, dtype=np.float32)
    labels = np.empty(shape=masks.shape[0:1] + label_shape, dtype=np.float32)

    index = 0

    for name in masks_map:
        if name in images_map:
            mask = cv.cvtColor(masks_map[name], cv.COLOR_BGRA2GRAY)

            images[index] = images_map[name] / 255.0
            labels[index] = mask.reshape(mask.shape + (1,)) / 255.0

            index += 1

    images.resize((index,) + images.shape[1:], refcheck=False)
    labels.resize((index,) + labels.shape[1:], refcheck=False)

    print('Dataset created!')

    return ImageMaskDataset(images, labels, image_shape, label_shape, index)


def augment_dataset(dataset, new_size=1000):
    assert new_size > dataset.size

    print('Augmenting dataset...')

    old_size = dataset.size

    dataset.images.resize((new_size,) + dataset.image_shape, refcheck=False)
    dataset.labels.resize((new_size,) + dataset.label_shape, refcheck=False)
    dataset.size = new_size

    augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.5,
        zoom_range=0.25,
        channel_shift_range=0.5,
        fill_mode='reflect'
    )

    for index in range(old_size, new_size):
        choice = np.random.randint(old_size)
        seed = np.random.randint(2147483647)

        dataset.images[index] = augmentation.random_transform(dataset.images[choice], seed)
        dataset.labels[index] = augmentation.random_transform(dataset.labels[choice], seed)

        # cv.imshow('image_wnd', dataset.images[index])
        # cv.imshow('label_wnd', dataset.labels[index])
        # cv.waitKey(0)

    for _ in range(np.amax([int(new_size * 0.01), 1])):
        permutation = np.random.permutation(new_size)

        dataset.images = dataset.images[permutation]
        dataset.labels = dataset.labels[permutation]

    print('Dataset augmented!')

    return dataset
