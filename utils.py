import os
import sys

import numpy as np
import cv2 as cv
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


