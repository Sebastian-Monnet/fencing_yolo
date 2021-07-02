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
import sklearn
import pickle


class Clip:
    def __init__(self, ind, svm, step_size=5, model=None):
        self.step_size = step_size
        self.brightness_thresh = 50
        self.dist_thresh = 10
        self.comp_time_lag = 40
        self.top_crop = 60 / 360
        self.bottom_crop = 310 / 360
        self.width_thresh = 10
        self.linreg_tol = 3

        self.svm = svm

        self.vid = Clip.load_vid(ind)
        self.vid = self.vid[:, int(self.top_crop *  self.vid.shape[1]): int(self.bottom_crop * self.vid.shape[1])]
        self.vid_ind = ind
        self.box_list_list = [None for i in self.vid]

        self.left_list = [None for i in self.vid]
        self.right_list = [None for i in self.vid]

        self.ref_depth = self.vid.shape[1] - 5

        if model is None:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        self.model = model

    # ---------------------------------------------- Load and play video

    @staticmethod
    def load_vid(ind):
        path = 'clips/' + str(ind) + '.mp4'
        if not os.path.isfile(path):
            Clip.download_clip('clips/', ind)
        cap = cv.VideoCapture(path)
        frame_list = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_list.append(frame)
            else:
                break
        return np.array(frame_list)

    @staticmethod
    def download_clip(path, ind):
        url = 'https://actions.quarte-riposte.com/api/actions/' + str(ind)
        file = urllib.request.urlopen(url)
        line = next(file)
        video_url = json.loads(line.decode())['video_url']
        clip = requests.get('https://actions.quarte-riposte.com' + video_url)

        open(path + str(ind) + '.mp4', 'wb').write(clip.content)

    def play_vid(self, wait=30, start=0):
        vid = self.vid.astype('uint8')
        for frame in vid[start:]:
            cv.imshow(str(self.vid_ind), frame)
            cv.waitKey(wait)

    def show_frame(self, frame_ind):
        cv.imshow(str(self.vid_ind), self.vid[frame_ind])
        cv.waitKey(10000)

    # -------------------------------------------------- Draw on video

    def draw_box(self, frame_ind, box, colour_channel=0):
        frame = self.vid[frame_ind]
        x_min, y_min, x_max, y_max = [int(a) for a in box[:4]]
        y_min = max(y_min, 1)
        y_max = min(y_max, frame.shape[0] - 2)
        x_min = max(x_min, 1)
        x_max = min(x_max, frame.shape[1] - 2)

        for i in range(y_min, y_max):
            frame[i, x_min, colour_channel] = frame[i, x_max, colour_channel] = 255
            frame[i, x_min - 1, colour_channel] = frame[i, x_max - 1, colour_channel] = 255
            frame[i, x_min + 1, colour_channel] = frame[i, x_max + 1, colour_channel] = 255

        for j in range(x_min, x_max):
            frame[y_min, j, colour_channel] = frame[y_max, j, colour_channel] = 255
            frame[y_min - 1, j, colour_channel] = frame[y_max - 1, j, colour_channel] = 255
            frame[y_min + 1, j, colour_channel] = frame[y_max + 1, j, colour_channel] = 255

    def draw_all_boxes_on_frame(self, frame_ind):
        for box in self.box_list_list[frame_ind]:
            self.draw_box(frame_ind, box)

    def draw_original_boxes_on_vid(self):
        for i, box_list in enumerate(self.box_list_list):
            if box_list is not None:
                for box in box_list:
                    self.draw_box(i, box, colour_channel=0)


    def draw_all_boxes_on_vid(self):
        for i, box_list in enumerate(self.box_list_list):
            if self.left_list[i] is not None:
                self.draw_left_on_frame(i)
            if self.right_list[i] is not None:
                self.draw_right_on_frame(i)

    def draw_left_on_frame(self, frame_ind):
        self.draw_box(frame_ind, self.left_list[frame_ind], colour_channel=2)

    def draw_right_on_frame(self, frame_ind):
        self.draw_box(frame_ind, self.right_list[frame_ind], colour_channel=1)



    # ------------------------------------------------ Compute boxes

    def compute_boxes_for_frame(self, frame_ind):
        result = self.model(self.vid[frame_ind])
        box_tensor = result.xyxy[0]
        box_list = []
        for box in box_tensor:
            if box[5] == 0:
                box_list.append(box[:4])
        # we only take the first four boxes, because they are the biggest, so we automatically discard a lot of
        # stuff in the background
        self.box_list_list[frame_ind] = box_list[:4]

    def compute_boxes_for_vid(self):
        step_size = self.step_size
        for frame_ind in range(0, len(self.vid), step_size):
            self.compute_boxes_for_frame(frame_ind)
        self.compute_boxes_for_frame(len(self.box_list_list) - 1)

    # ----------------------------------------------- Prune boxes

    def get_box_subframe(self, box, frame_ind):
        x_min, y_min, x_max, y_max = [int(a.item()) for a in box]
        return self.vid[frame_ind][y_min: y_max, x_min: x_max]


    @staticmethod
    def get_box_perimeter(box):
        return 2 * (box[2] - box[0] + box[3] - box[1])



    @staticmethod
    def get_features(im):
        re_size = 3
        h, w = im.shape[0], im.shape[1]

        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        im = cv.resize(im, (re_size, re_size))
        ftrs = np.zeros(re_size * re_size + 2)
        ftrs[:re_size * re_size] = im.flatten()
        ftrs[re_size * re_size] = h
        ftrs[re_size * re_size + 1] = w

        return ftrs

    def prune_non_fencers(self):
        for i, box_list in enumerate(self.box_list_list):
            if box_list is None:
                continue
            new_list = []
            for box in box_list:
                subim = self.get_box_subframe(box, i)
                ftrs = Clip.get_features(subim)
                pred = self.svm.predict([ftrs])
                if pred[0] == 1:
                    new_list.append(box)
            self.box_list_list[i] = new_list

    def prune_low_boxes(self):
        for i, box_list in enumerate(self.box_list_list):
            if box_list is None:
                continue
            new_list = []
            for box in box_list:
                if box[3] <= self.ref_depth or box[1] <= self.vid.shape[1] / 2:
                    new_list.append(box)
            self.box_list_list[i] = new_list



    def prune_small_boxes(self):
        small_list = []
        for box_list in self.box_list_list:
            if box_list is None or len(box_list) <= 1:
                continue
            else:
                perim_list = sorted([Clip.get_box_perimeter(box) for box in box_list])
                small_list.append(perim_list[-2])
        if small_list == []:
            return
        mean = np.mean(small_list)
        std = np.std(small_list)

        for i, box_list in enumerate(self.box_list_list):
            if box_list is None:
                continue
            self.box_list_list[i] = [box for box in box_list if Clip.get_box_perimeter(box) >= mean - 1.5 * std]

    def prune_high_boxes(self):
        for i, box_list in enumerate(self.box_list_list):
            if box_list is None or len(box_list) < 2:
                continue
            y_max_list = sorted([box[3] for box in box_list])
            second_lowest = y_max_list[-2]
            new_list = [box for box in box_list if box[3] >= second_lowest]
            self.box_list_list[i] = new_list

    def prune_with_linreg(self):
        not_none = [i for i in range(len(self.box_list_list)) if self.box_list_list[i] is not None]
        X_list = []
        y_list = []

        for i in not_none:
            for box in self.box_list_list[i]:
                cent = Clip.get_box_centre(box)
                X_list.append([cent[0]])
                y_list.append([cent[1]])
                
        X_arr = np.array(X_list)
        y_arr = np.array(y_list)

        linreg = sklearn.linear_model.LinearRegression()
        linreg.fit(X_arr, y_arr)
        
        res_arr = linreg.predict(X_arr) - y_arr
        
        std = np.std(res_arr)

        for i in not_none:
            new_list = []
            for box in self.box_list_list[i]:
                cent = Clip.get_box_centre(box)
                x, y= cent[0], cent[1]
                y_pred = linreg.predict([[x]])[0][0]
                if abs(y_pred - y) <= self.linreg_tol * std:
                    new_list.append(box)
            self.box_list_list[i] = new_list







    # ------------------------------------ Work with boxes
    @staticmethod
    def get_box_centre(box):
        # returns as (x, y)
        return (box[0] + box[2]) / 2, (box[1] + box[3])/2

    @staticmethod
    def get_box_distance(box_1, box_2):
        # returns L1 norm between centres
        x1, y1 = Clip.get_box_centre(box_1)
        x2, y2 = Clip.get_box_centre(box_2)

        return abs(x2 - x1) + abs(y2 - y1)



    # ----------------------------------- Reduce to two fencers

    def extract_left_right(self):
        for i, box_list in enumerate(self.box_list_list):
            if box_list is None:
                continue
            if len(box_list) == 2:
                left, right = box_list[0], box_list[1]
                if left[0] > right[0]:
                    left, right = right, left
                self.left_list[i] = left
                self.right_list[i] = right




    def add_more_left_right(self):
        last_left_dic = {}
        last_right_dic = {}
        last_left = None
        last_right = None
        for i in range(len(self.box_list_list)):
            if self.left_list[i] is not None:
                last_left = i
            if self.right_list[i] is not None:
                last_right = i
            last_left_dic[i] = last_left
            last_right_dic[i] = last_right

        for i, box_list in enumerate(self.box_list_list):
            if box_list is None:
                continue
            for box in box_list:
                if last_left_dic[i] is None:
                    pass
                elif Clip.get_box_distance(box, self.left_list[last_left_dic[i]]) < self.dist_thresh * self.step_size:
                    self.left_list[i] = box
                if last_right_dic[i] is None:
                    pass
                elif Clip.get_box_distance(box, self.right_list[last_right_dic[i]]) < self.dist_thresh * self.step_size:
                    self.right_list[i] = box

    # ------------------------------ get consistent subgraph

    def are_connected(self, box_list, ind1, ind2):
        return Clip.get_box_distance(box_list[ind1], box_list[ind2]) < self.dist_thresh * abs(ind2 - ind1)

    def get_comps(self, box_list):
        non_none = [i for i in range(len(box_list)) if box_list[i] is not None]
        comp_list = []
        for i in non_none:
            new_comp = True
            for comp in comp_list:
                valid_comps = []
                if self.are_connected(box_list, comp[-1], i) and i - comp[-1] < self.comp_time_lag:
                    valid_comps.append(comp)
                    new_comp = False
                    break
            if new_comp:
                comp_list.append([i])
            else:
                most_recent_comp = valid_comps[0]
                for comp in valid_comps:
                    if comp[-1] > most_recent_comp[-1]:
                        most_recent_comp = comp
                most_recent_comp.append(i)
        return comp_list

    def prune_inconsistencies_single_list(self, box_list):
        comp_list = self.get_comps(box_list)
        max_ind = 0
        for i, elem in enumerate(comp_list):
            if len(elem) > len(comp_list[max_ind]):
                max_ind = i
        for i in range(len(box_list)):
            if i not in comp_list[max_ind]:
                box_list[i] = None

    def prune_inconsistencies_vid(self):
        self.prune_inconsistencies_single_list(self.left_list)
        self.prune_inconsistencies_single_list(self.right_list)




    # ------------------------------------- Interpolate boxes

    @staticmethod
    def interpolate_pair(box1, box2, ind1, ind2, cur_ind):
        disp = box2 - box1
        proportion = (cur_ind - ind1) / (ind2 - ind1)
        return box1 + proportion * disp

    def interpolate_frames_left(self, ind1, ind2, cur_ind):
        left_1 = self.left_list[ind1]

        left_2 = self.left_list[ind2]

        self.left_list[cur_ind] = Clip.interpolate_pair(left_1, left_2, ind1, ind2, cur_ind)

    def interpolate_frames_right(self, ind1, ind2, cur_ind):
        right_1 = self.right_list[ind1]

        right_2 = self.right_list[ind2]

        self.right_list[cur_ind] = Clip.interpolate_pair(right_1, right_2, ind1, ind2, cur_ind)

    def interpolate_vid_left(self):
        not_none = [i for i in range(len(self.box_list_list)) if self.left_list[i] is not None]
        for i, ind1 in enumerate(not_none[:-1]):
            ind2 = not_none[i + 1]
            for j in range(ind1 + 1, ind2):
                self.interpolate_frames_left(ind1, ind2, j)

    def interpolate_vid_right(self):
        not_none = [i for i in range(len(self.box_list_list)) if self.right_list[i] is not None]
        for i, ind1 in enumerate(not_none[:-1]):
            ind2 = not_none[i + 1]
            for j in range(ind1 + 1, ind2):
                self.interpolate_frames_right(ind1, ind2, j)


    def interpolate_vid(self):
        self.interpolate_vid_left()
        self.interpolate_vid_right()

    # --------------------------------------------------- main method

    def main(self, step_size=5):
        self.step_size = step_size
        self.compute_boxes_for_vid()
        self.prune_non_fencers()
        self.prune_small_boxes()
        self.prune_low_boxes()
        self.prune_high_boxes()
        self.prune_with_linreg()

        self.extract_left_right()
        self.prune_inconsistencies_vid()
        self.add_more_left_right()
        self.interpolate_vid_left()
        self.interpolate_vid_right()

        self.draw_all_boxes_on_vid()
        self.draw_original_boxes_on_vid()




model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
svm = pickle.load(open('/Users/sebastianmonnet/PycharmProjects/yolov5_fencing/svm.pt', 'rb'))

start = datetime.datetime.now()
clip_ind = random.randint(1, 12300)
print(clip_ind)
a = Clip(12205, svm, model=model)
a.main(5)

print(datetime.datetime.now() - start)
a.play_vid(120, start=30)


# troublesome inds: 8015 (ref), 4312 (missing fencer), 723 (cross), 7848, 8135, 11315, 12205



