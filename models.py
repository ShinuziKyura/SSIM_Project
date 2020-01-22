import os
import sys
import datetime

import tensorflow as tf


# TODO check:
#  https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/?utm_source=blog&utm_medium=computer-vision-implementing-mask-r-cnn-image-segmentation
#  https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4
#  https://www.tensorflow.org/tutorials/images/segmentation
#  https://arxiv.org/abs/1505.04597


def downsampling_layer(filters, size, apply_normalization=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size,
                               strides=2,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False)
    )

    if apply_normalization:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsampling_layer(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size,
                                        strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False)
    )

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def create_unet_model():  # TODO possibly pass input/output shape and adjust
    print('Creating model...')

    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsampling_layer(64, 4, apply_normalization=False),
        downsampling_layer(128, 4),
        downsampling_layer(256, 4),
        downsampling_layer(512, 4),
        downsampling_layer(512, 4),
        downsampling_layer(512, 4),
        downsampling_layer(512, 4),
        downsampling_layer(512, 4),
    ]

    up_stack = [
        upsampling_layer(512, 4, apply_dropout=True),
        upsampling_layer(512, 4, apply_dropout=True),
        upsampling_layer(512, 4, apply_dropout=True),
        upsampling_layer(512, 4),
        upsampling_layer(256, 4),
        upsampling_layer(128, 4),
        upsampling_layer(64, 4),
    ]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(filters=1,
                                           kernel_size=3,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='sigmoid')

    x = inputs

    # Downsampling and creating the skip connections
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    # Upsampling and establishing the skip connections
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    outputs = last(x)

    print('Model created!')

    return tf.keras.Model(inputs=inputs, outputs=outputs)


@tf.function
def train_step(model, images, labels, optimizer, loss_function):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_function(predictions, labels)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


@tf.function
def test_step(model, images, labels, loss_function):
    predictions = model(images, training=False)
    loss = loss_function(predictions, labels)

    return loss


def train_model(dataset, model, loss_function, dataset_split=0.1, batch_size=50, learning_rate=0.001, epochs=20):
    print('Training model...')

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    summary = tf.summary.create_file_writer(
        os.path.normpath(os.path.join(
            os.path.dirname(sys.argv[0]),
            'logs/{}_{}'.format(loss_function.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        ))
    )

    split_index = int(dataset.size * dataset_split)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (dataset.images[split_index:], dataset.labels[split_index:])
    ).shuffle(dataset.size, reshuffle_each_iteration=True).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (dataset.images[:split_index], dataset.labels[:split_index])
    ).batch(batch_size)

    for epoch in range(epochs):  # TODO add way to log and report loss
        print('\tEpoch: {}'.format(epoch + 1))

        total_loss = 0.0
        total_steps = 0.0

        for images, labels in train_dataset:
            loss = train_step(model, images, labels, optimizer, loss_function)

            with summary.as_default():
                tf.summary.scalar('Train Loss', loss, step=epoch + 1)

            total_loss += loss
            total_steps += 1.0

            print('\t\tTrain loss: {}\t\t\t\t\t'.format(loss), end='\r')

        print('\t\tTrain average loss: {}\t\t\t\t\t'.format(total_loss / total_steps))

        total_loss = 0.0
        total_steps = 0.0

        for images, labels in test_dataset:
            loss = test_step(model, images, labels, loss_function)

            with summary.as_default():
                tf.summary.scalar('Test Loss', loss, step=epoch + 1)

            total_loss += loss
            total_steps += 1.0

            print('\t\tTest loss: {}\t\t\t\t\t'.format(loss), end='\r')

        print('\t\tTest average loss: {}\t\t\t\t\t'.format(total_loss / total_steps))

    print('Model trained!')
