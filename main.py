import os
import sys

import numpy as np
import tensorflow as tf
# import wandb

import utils
import storage
import datasets
import models


wwu_muenster_images_path = 'WWU_Muenster_Barcode_DB/images/'
wwu_muenster_masks_path = 'WWU_Muenster_Barcode_DB/masks/'

arte_lab_images_path = 'ArTe-Lab_1D_Medium_Barcode/images/'
arte_lab_masks_path = 'ArTe-Lab_1D_Medium_Barcode/masks/'


def main():
    print('Initializing...')

    base_path = os.path.dirname(sys.argv[0])

    images_paths = [
        os.path.normpath(os.path.join(base_path, 'datasets/', wwu_muenster_images_path)),
        os.path.normpath(os.path.join(base_path, 'datasets/', arte_lab_images_path))
    ]

    masks_paths = [
        os.path.normpath(os.path.join(base_path, 'datasets/', wwu_muenster_masks_path)),
        os.path.normpath(os.path.join(base_path, 'datasets/', arte_lab_masks_path))
    ]

    models_path = os.path.normpath(os.path.join(base_path, 'models/'))

    # Load data
    images = None
    masks = None

    for images_path, masks_path in zip(images_paths, masks_paths):
        images = storage.load_datamap(images_path, datamap=images)
        masks = storage.load_datamap(masks_path, datamap=masks)

    # Create dataset
    dataset = datasets.create_dataset(images, masks, rescale=(256, 256))

    # Augment dataset
    dataset = datasets.augment_dataset(dataset, new_size=2500)

    # Create model
    model = models.create_model()
    model.summary()

    loss_function = {
        'mse': tf.keras.losses.MeanSquaredError(name='mse'),
        'mae': tf.keras.losses.MeanAbsoluteError(name='mae'),
        'bce': tf.keras.losses.BinaryCrossentropy(name='bce')
    }[sys.argv[1]] if len(sys.argv) > 1 else tf.keras.losses.MeanSquaredError(name='mse')

    print(loss_function)

    # Load weights
    if utils.input_boolean('Load weights?'):
        try:
            model.load_weights(os.path.join(models_path, loss_function.name))
        except Exception as ex:
            print('Load error!')
            print(ex)

    # Predict images (pre-training)
    print('Showing pre-training example predictions...')
    for index in range(5):
        predictions = model.predict(np.array([dataset.images[index]]))
        utils.print_prediction(dataset.images[index], dataset.labels[index], predictions[0])

    # Train model
    models.train_model(
        dataset, model,
        loss_function=loss_function,
        epochs=60,
    )

    # Predict images (post-training)
    print('Showing post-training example predictions...')
    for index in range(5):
        predictions = model.predict(np.array([dataset.images[index]]))
        utils.print_prediction(dataset.images[index], dataset.labels[index], predictions[0])

    # Save weights
    if utils.input_boolean('Save weights?'):
        try:
            model.save_weights(os.path.join(models_path, loss_function.name))
        except Exception as ex:
            print('Save error!')
            print(ex)

    print('Terminating...')

    return model


if __name__ == '__main__':
    # wandb.init(project="ssim_project")

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    main()

    sys.exit(0)
