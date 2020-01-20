import tensorflow as tf


# TODO check:
#  https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/?utm_source=blog&utm_medium=computer-vision-implementing-mask-r-cnn-image-segmentation
#  https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4
#  https://www.tensorflow.org/tutorials/images/segmentation
#  https://arxiv.org/abs/1505.04597
def create_model(input_shape, output_shape):
    print('Creating model...')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.MaxPool2D(pool_size=2),

        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.MaxPool2D(pool_size=2),

        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.MaxPool2D(pool_size=2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ], name='')
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print('Creation successful!')

    return model
