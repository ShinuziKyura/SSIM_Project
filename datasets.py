import os

import cv2 as cv
import numpy as np


class TFImageDataset:
    def __init__(self, images, labels, image_shape=None, label_shape=None, size=None):
        assert images.shape[0] == labels.shape[0]

        self.images = images
        self.labels = labels

        self.image_shape = (image_shape if image_shape is not None else images.shape[1:])
        self.label_shape = (label_shape if label_shape is not None else labels.shape[1:])

        self.size = (size if size is not None else images.shape[0])


def create_muenster_dataset(images, masks, rescale=None):
    assert images.shape[0] > 0
    assert masks.shape[0] > 0

    image_shape = images[0]['data'].shape
    label_shape = (8,)

    images_map = {}
    masks_map = {}

    if rescale is not None:
        image_shape = rescale + images[0]['data'].shape[2:]

        scale_arr = list(images[0]['data'].shape[0:2])
        rescale_arr = list(rescale)

        if np.all(scale_arr < rescale_arr, 0):
            method = cv.INTER_CUBIC
        elif np.all(scale_arr > rescale_arr, 0):
            method = cv.INTER_AREA
        else:
            method = cv.INTER_LINEAR

        for image in images:
            images_map[image['name']] = cv.resize(image['data'], rescale, method)
        for mask in masks:
            masks_map[mask['name']] = cv.resize(mask['data'], rescale)
    else:
        for image in images:
            images_map[image['name']] = image['data']
        for mask in masks:
            masks_map[mask['name']] = mask['data']

    images = np.empty(shape=masks.shape[0:1] + image_shape, dtype=images[0]['data'].dtype)
    labels = np.empty(shape=masks.shape[0:1] + label_shape, dtype=np.float32)

    index = 0

    for name in masks_map:
        if name in images_map:
            mask = cv.cvtColor(masks_map[name], cv.COLOR_BGRA2GRAY)
            mask = cv.cornerHarris(mask, 2, 3, 0.04)
            _, mask = cv.threshold(mask, 0.1 * mask.max(), 255.0, cv.THRESH_BINARY)
            mask = np.uint8(mask)

            _, _, _, label = cv.connectedComponentsWithStats(mask)
            label = np.int32(label[1:])

            if label.shape[0] == 4:
                if rescale is not None:
                    label = label / list(rescale)

#               label = np.array([tuple(elem) for elem in label.tolist()], dtype=[('x', np.float32), ('y', np.float32)])
#               label.sort(0, order=('y', 'x'))
#               label = np.array([list(elem) for elem in label.tolist()])
                label = label.flatten()

                images[index] = images_map[name]
                labels[index] = label
                index += 1

    images.resize((index,) + images.shape[1:], refcheck=False)
    labels.resize((index,) + labels.shape[1:], refcheck=False)

    return TFImageDataset(images, labels, image_shape, label_shape, index)
