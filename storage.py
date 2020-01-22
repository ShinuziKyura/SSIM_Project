import os
import sys

import numpy as np
import cv2 as cv


def load_datamap(path, ext=('jpeg', 'jpg', 'png', 'tiff'), datamap=None, load_from_cache=True, save_to_cache=True):
    assert os.path.isdir(path)

    if datamap is None:
        datamap = np.empty(shape=(0,), dtype=[('name', np.str_, 255), ('data', np.ndarray)])

    index = len(datamap)

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

        with np.load(cached_file, allow_pickle=True) as cache:
            datamap = np.concatenate((datamap, cache['datamap']))

        print('Loading finished!')
    else:
        print('Loading data...')

        directory = os.fsencode(path)

        for file in os.listdir(directory):
            filename = os.fsdecode(file)

            if filename.endswith(ext):
                filepath = os.path.join(path, filename)

                name = os.path.splitext(filename)[0]
                data = cv.imread(filepath, cv.IMREAD_UNCHANGED)

                if index >= len(datamap):
                    datamap.resize((index + 1000,), refcheck=False)

                datamap[index] = (name, data)
                index += 1

        datamap.resize((index,), refcheck=False)

        print('Loading finished!')

        if save_to_cache:
            print('Caching data...')

            np.savez_compressed(cached_file, datamap=datamap)

            print('Caching finished!')

    return datamap
