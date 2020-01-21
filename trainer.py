import os
import sys

import cv2 as cv
import numpy as np
import tensorflow as tf

import utils
import storage
import datasets
import models


muenster_images_path = '.\\datasets\\Muenster_BarcodeDB\\N95-2592x1944_scaledTo640x480bilinear'
muenster_masks_path = '.\\datasets\\Muenster_BarcodeDB_detection_masks\\Detection'
muenster_models_path = '.\\models\\Muenster_BarcodeDB_N95-2592x1944_scaledTo640x480bilinear'
number_of_epochs = 10


def execute(images_path, masks_path, models_path):
    # load data
    images = storage.load_data(images_path)
    masks = storage.load_data(masks_path)

    # create dataset
    dataset = datasets.create_muenster_dataset(images, masks, rescale=(512, 512)) # TODO make both function and rescale parameters

    # create / load model
    model = None

    if utils.boolean_input('Should load model?'):
        model = storage.load_model(models_path)

    if model is None:
        model = models.create_model(dataset.image_shape, dataset.label_shape[0])

    # train model
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    for epoch in range(number_of_epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in zip(dataset.images, dataset.labels):
            models.train_step(model, images, labels, train_loss=train_loss, train_accuracy=train_accuracy)

        for images, labels in zip(dataset.images, dataset.labels):
            models.test_step(model, images, labels, test_loss=test_loss, test_accuracy=test_accuracy)

        print('Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'.format(
            epoch + 1,
            train_loss.result(),
            train_accuracy.result() * 100,
            test_loss.result(),
            test_accuracy.result() * 100)
        )

    # save model
    if utils.boolean_input('Should save model?'):
        storage.save_model(model, models_path)

    sys.exit(0)


if __name__ == '__main__':
    base_path = os.path.dirname(sys.argv[0])

    muenster_images_path = os.path.normpath(os.path.join(
        base_path,
        sys.argv[1] if len(sys.argv) > 1 else muenster_images_path
    ))
    muenster_masks_path = os.path.normpath(os.path.join(
        base_path,
        sys.argv[2] if len(sys.argv) > 2 else muenster_masks_path
    ))
    muenster_models_path = os.path.normpath(os.path.join(
        base_path,
        sys.argv[3] if len(sys.argv) > 3 else muenster_models_path
    ))

    execute(muenster_images_path, muenster_masks_path, muenster_models_path)
