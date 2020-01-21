import matplotlib.pyplot as plt
import tensorflow as tf


def boolean_input(prompt):
    while True:
        print(prompt)
        retval = input('> ')

        switch = {
            'y': True,
            'yes': True,
            'n': False,
            'no': False
        }

        if retval in switch:
            return switch[retval]


def display_prediction(input_image, ground_truth, predicted_image):
    display_list = [input_image, ground_truth, predicted_image[0]]
    title_list = ['Input image', 'Ground truth', 'Predicted image']

    plt.figure(figsize=(15, 15))

    for idx in range(3):
        plt.subplot(1, 3, idx + 1)
        plt.title(title_list[idx])
        plt.axis('off')
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[idx]))

    plt.show()
