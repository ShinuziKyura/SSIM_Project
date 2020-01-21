import tensorflow as tf


# TODO check:
#  https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/?utm_source=blog&utm_medium=computer-vision-implementing-mask-r-cnn-image-segmentation
#  https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4
#  https://www.tensorflow.org/tutorials/images/segmentation
#  https://arxiv.org/abs/1505.04597
def create_model(input_shape, output_shape):
    print('Creating model...')

    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    inputs_1 = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu')(inputs_1)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
    inputs_2 = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu')(inputs_2)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
    inputs_3 = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu')(inputs_3)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu')(x)
    inputs_4 = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, activation='relu')(inputs_4)
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, activation='relu')(x)

    # Decoder
    x = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, activation='relu')(x)

    outputs = tf.keras.layers.Conv2DTranspose(filters=output_shape, kernel_size=1, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    print('Creation successful!')

    return model


@tf.function
def train_step(model, images, labels, train_loss, train_accuracy):
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_function(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(model, images, labels, test_loss, test_accuracy):
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

    predictions = model(images, training=False)
    loss = loss_function(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)
