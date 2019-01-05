

import cv2
import numpy as np

import torch
import torchvision

import matplotlib.pyplot as plt


def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)


def black_out(image, width=3):
    '''
    Blacks out everything in the image except for the top left quadrant
    :param image: Numpy array
    :return:
    '''

    h = image.shape[1]
    w = image.shape[2] // 2

    mask = np.zeros(image.shape, np.float32)

    mask[0, 0: h, w - width: w + width] = image[0, 0: h, w - width: w + width]

    return mask
