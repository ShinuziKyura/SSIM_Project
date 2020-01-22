import tensorflow as tf
# import wandb


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


def create_model():
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
                                           kernel_size=4,
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

    # wandb.config.epochs = epochs
    # wandb.config.dataset_size = dataset.size
    # wandb.config.batch_size = batch_size
    # wandb.config.dataset_split = 1.0 - dataset_split

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    split_index = int(dataset.size * dataset_split)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (dataset.images[split_index:], dataset.labels[split_index:])
    ).shuffle(dataset.size, reshuffle_each_iteration=True).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (dataset.images[:split_index], dataset.labels[:split_index])
    ).batch(batch_size)

    for epoch in range(epochs):
        print('\tEpoch: {}'.format(epoch + 1))

        total_loss = 0.0
        total_steps = 0.0

        for images, labels in train_dataset:
            loss = train_step(model, images, labels, optimizer, loss_function)

            total_loss += loss
            total_steps += 1.0

            # wandb.log({'train_loss': loss.numpy()})
            print('\t\tTrain loss: {}\t\t\t\t\t'.format(loss), end='\r')

        print('\t\tTrain average loss: {}\t\t\t\t\t'.format(total_loss / total_steps))

        total_loss = 0.0
        total_steps = 0.0

        for images, labels in test_dataset:
            loss = test_step(model, images, labels, loss_function)

            total_loss += loss
            total_steps += 1.0

            # wandb.log({'test_loss': loss.numpy()})
            print('\t\tTest loss: {}\t\t\t\t\t'.format(loss), end='\r')

        print('\t\tTest average loss: {}\t\t\t\t\t'.format(total_loss / total_steps))


print('Training finished!')
