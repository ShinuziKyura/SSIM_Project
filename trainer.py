import os
import sys

import cv2 as cv
import numpy as np

import utils
import storage
import datasets
import models


muenster_images_path = '.\\datasets\\Muenster_BarcodeDB\\N95-2592x1944_scaledTo640x480bilinear'
muenster_masks_path = '.\\datasets\\Muenster_BarcodeDB_detection_masks\\Detection'
muenster_models_path = '.\\models\\Muenster_BarcodeDB_N95-2592x1944_scaledTo640x480bilinear'


def execute(images_path, masks_path, models_path):
    # load data
    images = storage.load_data(images_path)
    masks = storage.load_data(masks_path)

    # create dataset
    dataset = datasets.create_muenster_dataset(images, masks, rescale=(256, 256)) # TODO make both function and rescale parameters

    # create / load model
    model = None

    if utils.boolean_input('Should load model?'):
        model = storage.load_model(models_path)

    if model is None:
        model = models.create_model(dataset.image_shape, dataset.label_shape[0])

    # train model

    # save model
    if utils.boolean_input('Should save model?'):
        storage.save_model(model, models_path)

    sys.exit(0)


if __name__ == '__main__':
    if len(sys.argv) > 3:
        muenster_models_path = os.path.join(sys.argv[0], sys.argv[3])
    if len(sys.argv) > 2:
        muenster_masks_path = os.path.join(sys.argv[0], sys.argv[2])
    if len(sys.argv) > 1:
        muenster_images_path = os.path.join(sys.argv[0], sys.argv[1])

    execute(muenster_images_path, muenster_masks_path, muenster_models_path)
