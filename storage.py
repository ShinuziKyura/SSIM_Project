import os
import sys

import cv2 as cv
import numpy as np
import tensorflow as tf


def load_data(path, ext=('jpeg', 'jpg', 'png', 'tiff'), load_from_cache=True, save_to_cache=True):
    assert os.path.isdir(path)

    named_data = np.empty(shape=(0,), dtype=[('name', np.str_, 255), ('data', np.ndarray)])
    index = 0

    pathname = os.path.splitdrive(os.path.abspath(path))
    cached_file = os.path.normpath(os.path.join(
        os.path.dirname(sys.argv[0]),
        'cache/',
        pathname[0].replace(':', '').replace(os.path.sep, '__').strip('__')
        + '__'
        + pathname[1].replace(os.path.sep, '__').strip('__')
        + '.npz'
    ))

    if load_from_cache and os.path.exists(cached_file):
        print('Loading cached data...')

        with np.load(cached_file, allow_pickle=True) as archive:
            named_data = archive['named_data']

        print('Data loaded!')
    else:
        print('Loading data...')

        directory = os.fsencode(path)

        for file in os.listdir(directory):
            filename = os.fsdecode(file)

            if filename.endswith(ext):
                filepath = os.path.join(path, filename)

                name = os.path.splitext(filename)[0]
                data = cv.imread(filepath, cv.IMREAD_UNCHANGED)

                if index == named_data.shape[0]:
                    named_data.resize((index + 100,), refcheck=False)

                named_data[index] = (name, data)
                index += 1

        named_data.resize((index,), refcheck=False)

        print('Data loaded!')

        if save_to_cache:
            print('Caching data...')

            np.savez_compressed(cached_file, named_data=named_data)

            print('Data cached!')

    return named_data


def load_model(path):
    model = None
    loaded = False

    try:
        print('Loading model...')
        model = tf.keras.models.load_model(path)
        loaded = True
        print('Model loaded!')
    except (IOError, ImportError) as ex:
        print('Load error!')
        print(ex)

    #return model, loaded
    return model


def save_model(model, path, model_format='tf'):
    saved = False

    try:
        print('Saving model...')
        tf.keras.models.save_model(model, path, save_format=model_format)
        saved = True
        print('Model saved!')
    except ImportError as ex:
        print('Save error!')
        print(ex)

    return saved
