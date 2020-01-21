import os
import sys

import numpy as np
import tensorflow as tf

import utils
import storage
import datasets
import models


wwu_muenster_images_path = 'datasets/WWU_Muenster_Barcode_DB/images/'
wwu_muenster_masks_path = 'datasets/WWU_Muenster_Barcode_DB/masks/'

arte_lab_images_path = 'datasets/ArTe-Lab_1D_Medium_Barcode/images/'
arte_lab_masks_path = 'datasets/ArTe-Lab_1D_Medium_Barcode/masks/'

mse_models_path = 'models/mean_squared_error'
#bc_models_path = 'models/binary_crossentropy'


def execute(image_paths, mask_paths, model_path):
    assert len(image_paths) == len(mask_paths)

    print('Initializing...')

    # Load data
    images = None
    masks = None

    for image_path, mask_path in zip(image_paths, mask_paths):
        images = storage.load_datamap(image_path, datamap=images)
        masks = storage.load_datamap(mask_path, datamap=masks)

    # Create dataset
    dataset = datasets.create_dataset(images, masks, rescale=(256, 256))

    # Augment dataset
    dataset = datasets.augment_dataset(dataset, new_size=5000)

    # Create model
    model = models.create_unet_model()

    # Load weights
    if utils.boolean_input('Load weights?'):
        model.load_weights(model_path)

    # Predict images (pre-training)
    for index in range(5):
        result = model.predict(np.array([dataset.images[index]]))
        utils.display_prediction(dataset.images[index], dataset.labels[index], result)

    # Train model
    models.train_model(dataset, model, batch_size=50, epochs=20)

    # Predict images (post-training)
    for index in range(5):
        result = model.predict(np.array([dataset.images[index]]))
        utils.display_prediction(dataset.images[index], dataset.labels[index], result)

    # Save weights
    if utils.boolean_input('Save weights?'):
        model.save_weights(model_path)

    print('Terminating...')
    return model


if __name__ == '__main__':
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    base_path = os.path.dirname(sys.argv[0])

    images_path_1 = os.path.normpath(os.path.join(
        base_path,
        sys.argv[1] if len(sys.argv) > 1 else wwu_muenster_images_path
    ))
    images_path_2 = os.path.normpath(os.path.join(
        base_path,
        sys.argv[1] if len(sys.argv) > 1 else arte_lab_images_path
    ))

    masks_path_1 = os.path.normpath(os.path.join(
        base_path,
        sys.argv[2] if len(sys.argv) > 2 else wwu_muenster_masks_path
    ))
    masks_path_2 = os.path.normpath(os.path.join(
        base_path,
        sys.argv[2] if len(sys.argv) > 2 else arte_lab_masks_path
    ))

    models_path = os.path.normpath(os.path.join(
        base_path,
        sys.argv[3] if len(sys.argv) > 3 else mse_models_path
    ))

    execute([images_path_1, images_path_2], [masks_path_1, masks_path_2], models_path)

    sys.exit(0)
