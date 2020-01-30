import cv2 as cv


def input_boolean(prompt):
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


def print_prediction(input_image, ground_truth, predicted_image):
    cv.namedWindow('Input image', cv.WINDOW_FREERATIO)
    cv.namedWindow('Ground truth', cv.WINDOW_FREERATIO)
    cv.namedWindow('Predicted image', cv.WINDOW_FREERATIO)
    cv.imshow('Input image', input_image)
    cv.imshow('Ground truth', ground_truth)
    cv.imshow('Predicted image', predicted_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
