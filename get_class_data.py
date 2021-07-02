from main import Clip

import torch
import torchvision
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import cv2 as cv
import numpy as np
import urllib
import requests
import seaborn
import datetime
import copy
import yaml
import random

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


def get_images(vid_ind):
    clip = Clip(vid_ind, model=model)
    clip.step_size = 10
    clip.compute_boxes_for_vid()
    clip.prune_small_boxes()

    non_none = [i for i in range(len(clip.box_list_list)) if clip.box_list_list[i] is not None]
    i = random.choice(non_none)

    box_list = clip.box_list_list[i]

    im_list = []

    for box in box_list:
        x_min, y_min, x_max, y_max = [int(a.item()) for a in box]
        im_list.append(clip.vid[i][y_min: y_max, x_min: x_max])

    return im_list


def classify(im):
    cv.imshow('is_fencer?', im)
    cv.waitKey(1)
    print('y/n:')
    while True:
        res = input()
        if res == 'y':
            return 1
        elif res == 'n':
            return 0


def get_and_save_ims(vid_ind):
    im_list = get_images(vid_ind)
    for i, im in enumerate(im_list):
        res = classify(im)
        filename = str(vid_ind) + 'a' + str(i) + '.jpg'
        path = '/Users/sebastianmonnet/PycharmProjects/yolov5_fencing/class_data/' + str(res) + '/' + filename
        cv.imwrite(path, im)

def save_without_classification(vid_ind):
    im_list = get_images(vid_ind)
    for i, im in enumerate(im_list):
        filename = str(vid_ind) + 'a' + str(i) + '.jpg'
        path = '/Users/sebastianmonnet/PycharmProjects/yolov5_fencing/class_data/unclassified/' + filename
        cv.imwrite(path, im)


for i in range(10100, 12500, 10):
    save_without_classification(i)