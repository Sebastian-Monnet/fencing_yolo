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
import datetime
import copy
import yaml
import sklearn
import pickle
import csv

class Clip:
    def __init__(self, loc, svm, step_size=5, model=None):
        self.step_size = step_size
        self.brightness_thresh = 50
        self.x_dist_thresh = 8
        self.y_dist_thresh = 5

        self.small_dist_thresh = 10
        self.big_dist_thresh = 10
        self.comp_time_lag = 40
        self.top_crop = 0 / 360
        self.bottom_crop = 341 / 360
        self.width_thresh = 10
        self.linreg_tol = 3
        self.vert_IOU_thresh = 0.7
        self.grid_size = 10


        self.svm = svm

        self.vid = Clip.load_vid(loc)
        self.vid = self.vid[:, int(self.top_crop *  self.vid.shape[1]): int(self.bottom_crop * self.vid.shape[1])]
        self.disp_arr = [None for frame in self.vid]
        self.vid_ind = loc
        self.box_list_list = [None for i in self.vid]

        self.left_list = [None for i in self.vid]
        self.right_list = [None for i in self.vid]
        self.true_left_list = [None for i in self.vid]
        self.true_right_list = [None for i in self.vid]

        self.ref_depth = self.vid.shape[1] - 5

        if model is None:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        self.model = model

    # ---------------------------------------------- Load and play video

    @staticmethod
    def load_vid(loc):
        if type(loc) == str:
            path = loc
        else:
            path = 'clips/' + str(loc) + '.mp4'
            if not os.path.isfile(path):
                Clip.download_clip(path)
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
        for i, frame in enumerate(vid[start:]):
            if self.disp_arr[i + start] is not None:
                print('disp', self.disp_arr[i+start])
                print('True positions:', self.true_left_list[i+start], self.true_right_list[i+start])
            cv.imshow(str(self.vid_ind), frame)
            cv.waitKey(wait)

    def play_vid_with_disp(self, wait=30, start=0):
        vid = self.vid.astype('uint8')
        min_disp, max_disp = min(self.disp_arr), max(self.disp_arr)

        for i, frame in enumerate(vid[start:]):
            new_frame = np.zeros((frame.shape[0], int(max_disp - min_disp + frame.shape[1]) + 1, 3), dtype='uint8')
            disp = self.disp_arr[i + start]
            #new_frame[:, int(disp - min_disp) : int(disp - min_disp + frame.shape[1])] = frame
            new_frame[:, int(max_disp - disp) : int(frame.shape[1] + max_disp - disp)] = frame

            cv.imshow(str(self.vid_ind), new_frame)
            cv.waitKey(wait)

    def show_frame(self, frame_ind):
        cv.imshow(str(self.vid_ind), self.vid[frame_ind])
        cv.waitKey(10000)

    # ------------------------------------------------- Look for lights
    def is_left_red(self, frame_ind):
        pix = self.vid[frame_ind, 330, 90]
        return pix[2] > 250

    def is_right_white(self, frame_ind):
        pix = self.vid[frame_ind, 337, 392]
        return pix[0] == 255 and pix[1] == 255 and pix[2] == 255

    def is_right_green(self, frame_ind):
        pix = self.vid[frame_ind, 330, 410]
        return pix[1] > 245

    def is_left_white(self, frame_ind):
        pix = self.vid[frame_ind, 340, 240]
        return pix[0] == 255 and pix[1] == 254 and pix[2] == 255

    def get_left_light_time(self):
        num_frames = len(self.box_list_list)
        for frame_ind in range(num_frames - 80, num_frames):
            if self.is_left_white(frame_ind) or self.is_left_red(frame_ind):
                return frame_ind

    def get_right_light_time(self):
        num_frames = len(self.box_list_list)
        for frame_ind in range(num_frames - 80, num_frames):
            if self.is_right_white(frame_ind) or self.is_right_green(frame_ind):
                return frame_ind


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
            if self.true_left_list[i] is not None:
                self.draw_left_on_frame(i)
            if self.true_right_list[i] is not None:
                self.draw_right_on_frame(i)

    def draw_left_on_frame(self, frame_ind):
        disp = self.disp_arr[frame_ind]
        disp_arr = np.array([disp, 0, disp, 0])
        self.draw_box(frame_ind, self.true_left_list[frame_ind] + disp_arr, colour_channel=2)

    def draw_right_on_frame(self, frame_ind):
        disp = self.disp_arr[frame_ind]
        disp_arr = np.array([disp, 0, disp, 0])
        self.draw_box(frame_ind, self.true_right_list[frame_ind] + disp_arr, colour_channel=1)



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
        self.raw_box_list_list = copy.deepcopy(self.box_list_list)

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
        biggest_perim = 0
        for box_list in self.box_list_list:
            if box_list is None or len(box_list) <= 1:
                continue
            else:
                perim_list = sorted([Clip.get_box_perimeter(box) for box in box_list])
                small_list.append(perim_list[-2])
                biggest_perim = max(biggest_perim, perim_list[-1])
        if small_list == []:
            return
        mean = np.mean(small_list)
        std = np.std(small_list)

        for i, box_list in enumerate(self.box_list_list):
            if box_list is None:
                continue
            self.box_list_list[i] = [box for box in box_list if Clip.get_box_perimeter(box)
                                     >= max(mean - 1.5 * std, biggest_perim / 2)]

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
        if box is None:
            return None
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
        x_thresh = self.x_dist_thresh * self.step_size
        y_thresh = self.y_dist_thresh * self.step_size
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
                cent = Clip.get_box_centre(box)
                if last_left_dic[i] is not None:
                    left_cent = Clip.get_box_centre(self.left_list[last_left_dic[last_left_dic[i]]])
                if last_right_dic[i] is not None:
                    right_cent = Clip.get_box_centre(self.right_list[last_right_dic[last_right_dic[i]]])

                if last_left_dic[i] is None:
                    pass
                elif abs(cent[0] - left_cent[0]) < x_thresh and abs(cent[1] - left_cent[1]) < y_thresh:
                    self.left_list[i] = box
                if last_right_dic[i] is None:
                    pass
                elif abs(cent[0] - right_cent[0]) < x_thresh and abs(cent[1] - right_cent[1]) < y_thresh:
                    self.right_list[i] = box

    # ------------------------------ get consistent subgraph

    '''def are_connected(self, box_list, ind1, ind2):
        cent_1 = Clip.get_box_centre(box_list[ind1])
        cent_2 = Clip.get_box_centre(box_list[ind2])

        x_thresh = abs(ind2 - ind1) * self.x_dist_thresh
        y_thresh = abs(ind2 - ind1) * self.y_dist_thresh
        
        return abs(cent_1[0] - cent_2[0]) < x_thresh and abs(cent_1[1] - cent_2[1]) < y_thresh'''

    #def are_connected(self, box_list, ind1, ind2):
    #    return Clip.get_box_distance(box_list[ind1], box_list[ind2]) < 10 * abs(ind2 - ind1)

    def are_connected(self, box_list, ind1, ind2):
        box1 = box_list[ind1]
        box2 = box_list[ind2]
        IOU = Clip.vert_IOU(box1, box2)

        cent_1 = Clip.get_box_centre(box1)
        cent_2 = Clip.get_box_centre(box2)

        x_thresh = abs(ind2 - ind1) * self.x_dist_thresh

        return IOU > self.vert_IOU_thresh and cent_2[0] - cent_1[0] < x_thresh

    @staticmethod
    def vert_IOU(box1, box2):
        if box1[1] > box2[1]:
            box1, box2 = box2, box1
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        if y1_max < y2_min:
            return 0

        intersection = y1_max - y2_min
        union = (y2_max - y2_min) + (y1_max - y1_min) - intersection

        return intersection / union

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
        non_none = [i for i in range(len(box_list)) if box_list[i] is not None]
        new_box_list = copy.deepcopy(box_list)
        for ind in range(1, len(non_none) - 1):
            i = non_none[ind - 1]
            j = non_none[ind]
            k = non_none[ind + 1]
            dist_ij = Clip.get_box_distance(box_list[i], box_list[j])
            dist_jk = Clip.get_box_distance(box_list[j], box_list[k])
            dist_ik = Clip.get_box_distance(box_list[i], box_list[k])
            if dist_ik < self.small_dist_thresh * (k - i) \
                    and dist_jk > self.big_dist_thresh * (k - j)\
                    and dist_ij > self.big_dist_thresh * (j - i):
                new_box_list[j] = None
        for i in range(len(box_list)):
            box_list[i] = new_box_list[i]


    def prune_inconsistencies_vid(self):
        self.prune_inconsistencies_single_list(self.true_left_list)
        self.prune_inconsistencies_single_list(self.true_right_list)

    # ------------------------------------- Interpolate boxes

    @staticmethod
    def interpolate_pair(box1, box2, ind1, ind2, cur_ind):
        disp = box2 - box1
        proportion = (cur_ind - ind1) / (ind2 - ind1)
        return box1 + proportion * disp

    def interpolate_frames_left(self, ind1, ind2, cur_ind):
        left_1 = self.true_left_list[ind1]

        left_2 = self.true_left_list[ind2]

        self.true_left_list[cur_ind] = Clip.interpolate_pair(left_1, left_2, ind1, ind2, cur_ind)

    def interpolate_frames_right(self, ind1, ind2, cur_ind):
        right_1 = self.true_right_list[ind1]

        right_2 = self.true_right_list[ind2]

        self.true_right_list[cur_ind] = Clip.interpolate_pair(right_1, right_2, ind1, ind2, cur_ind)

    def interpolate_vid_left(self):
        not_none = [i for i in range(len(self.box_list_list)) if self.true_left_list[i] is not None]
        for i, ind1 in enumerate(not_none[:-1]):
            ind2 = not_none[i + 1]
            for j in range(ind1 + 1, ind2):
                self.interpolate_frames_left(ind1, ind2, j)

    def interpolate_vid_right(self):
        not_none = [i for i in range(len(self.box_list_list)) if self.true_right_list[i] is not None]
        for i, ind1 in enumerate(not_none[:-1]):
            ind2 = not_none[i + 1]
            for j in range(ind1 + 1, ind2):
                self.interpolate_frames_right(ind1, ind2, j)


    def interpolate_vid(self):
        self.interpolate_vid_left()
        self.interpolate_vid_right()

    # --------------------------------------------------- Frame displacement

    def get_vel_arr(self, old, new):
        '''
        input: two consecutive frames of video
        output: 2-dim numpy array
        Calculates optical flow in a grid of pixels, and returns the x-componenets as a 2-dim numpy array
        '''
        old = cv.GaussianBlur(cv.cvtColor(old, cv.COLOR_BGR2GRAY), (3, 3), 0, 0)
        new = cv.GaussianBlur(cv.cvtColor(new, cv.COLOR_BGR2GRAY), (3, 3), 0, 0)
        old, new = old[40:100], new[40:100]

        feature_params = dict(maxCorners=100,
                              qualityLevel=0.1,
                              minDistance=30,
                              blockSize=7)
        p0 = cv.goodFeaturesToTrack(old, mask=None, **feature_params)

        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                                   10, 0.03))

        p1, st, err = cv.calcOpticalFlowPyrLK(old, new, p0,
                                              None, **lk_params)

        vels = p1 - p0

        x_vel_list = []

        for i in range(vels.shape[0]):
            vel = vels[i]
            x_vel = vel[0, 0]
            x_vel_list.append(x_vel)

        return x_vel_list

    def get_frame_disp(self, frame1, frame2):
        x_vel_arr = self.get_vel_arr(frame1, frame2)
        return np.median(x_vel_arr)

    def compute_vid_disps(self, step_size=1):
        self.disp_arr[0] = 0
        for i in range(step_size, len(self.disp_arr), step_size):
            disp = self.get_frame_disp(self.vid[i - step_size], self.vid[i])
            for j in range(i - step_size + 1, i + 1):
                self.disp_arr[j] = self.disp_arr[i - step_size] + disp * ((j - (i - step_size)) / step_size)
        for i in range(len(self.disp_arr)):
            disp = np.array([self.disp_arr[i], 0, self.disp_arr[i], 0])
            if self.left_list[i] is not None:
                self.true_left_list[i] = self.left_list[i] - disp
            if self.right_list[i] is not None:
                self.true_right_list[i] = self.right_list[i] - disp

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

        self.compute_vid_disps(step_size=1)


        self.prune_inconsistencies_vid()
        #self.add_more_left_right()
        self.interpolate_vid_left()
        self.interpolate_vid_right()


        self.draw_all_boxes_on_vid()
        self.draw_original_boxes_on_vid()

    def get_data(self):
        return self.raw_box_list_list, self.box_list_list, self.left_list, self.right_list

    def save_data(self, dir, filename):
        pickle.dump(self.raw_box_list_list, open(os.path.join(dir, 'raw_box_list_lists', filename), 'wb'))
        pickle.dump(self.box_list_list, open(os.path.join(dir, 'box_list_lists', filename), 'wb'))
        pickle.dump(self.left_list, open(os.path.join(dir, 'left_lists', filename), 'wb'))
        pickle.dump(self.right_list, open(os.path.join(dir, 'right_lists', filename), 'wb'))
        pickle.dump(self.true_left_list, open(os.path.join(dir, 'true_left_lists', filename), 'wb'))
        pickle.dump(self.true_right_list, open(os.path.join(dir, 'true_right_lists', filename), 'wb'))
        pickle.dump(self.disp_arr, open(os.path.join(dir, 'disp_arrs', filename), 'wb'))

    def save_true_left_right_csv(self, dir, filename):
        x_left_list = [Clip.get_box_centre(box) for box in self.true_left_list]
        x_right_list = [Clip.get_box_centre(box) for box in self.true_right_list]

        for i in range(len(x_left_list)):
            if x_left_list[i] is not None:
                x_left_list[i] = x_left_list[i][0].item()
            if x_right_list[i] is not None:
                x_right_list[i] = x_right_list[i][0].item()

        left_light = self.get_left_light_time()
        right_light = self.get_right_light_time()

        if left_light is None:
            left_light = len(self.vid)
        if right_light is None:
            right_light = len(self.vid)

        light_time = min(left_light, right_light) + 3

        x_left_list = x_left_list[:light_time]
        x_right_list = x_right_list[:light_time]

        x_left_list = ['Left'] + x_left_list
        x_right_list = ['Right'] + x_right_list

        np.savetxt(os.path.join(dir, filename), [p for p in zip(x_left_list, x_right_list)], delimiter=',', fmt='%s')







model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
svm = pickle.load(open('/Users/sebastianmonnet/PycharmProjects/yolov5_fencing/svm.pt', 'rb'))

'''start = datetime.datetime.now()
a = Clip(1110, svm, model=model)
a.main()
print(datetime.datetime.now() - start)

a.play_vid_with_disp(60, start=0)'''

clip_inds = [i for i in range(1100, 1400, 10)]

for ind in clip_inds:
    Clip.download_clip('clips/', ind)

for ind in clip_inds:
    print('ind:', ind)
    start = datetime.datetime.now()
    path = 'clips/' + str(ind) + '.mp4'
    print(path)
    a = Clip(path, svm, model=model)
    a.main()
    a.save_data('extracted_data', str(ind) + '.pt')
    a.save_true_left_right_csv('extracted_data/true_left_right_csvs', str(ind) + '.csv')
    print(datetime.datetime.now() - start)




# troublesome inds: 8015 (ref), 4312 (missing fencer), 723 (cross), 7848, 8135, 11315, 12205, 12010



