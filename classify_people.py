import numpy
import sklearn
import cv2 as cv
import numpy as np
import os


def get_features(im):
    h, w = im.shape[0], im.shape[1]

    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im = cv.resize(im, (10, 10))
    ftrs = np.zeros(102)
    ftrs[:100] = im.flatten()
    ftrs[100] = h
    ftrs[101] = w

    return ftrs


def load_images(path):
    im_list = []
    file_list = os.listdir(path)

    for file in file_list:
        if file[-4:] == '.jpg':
            im_list.append(cv.imread(file))

    return im_list






