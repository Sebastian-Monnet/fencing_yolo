import cv2 as cv
import numpy as np
import torch
import pickle

class Viewer:
    def __init__(self, vid_path, box_list_list, left_list, right_list, disp_arr):
        self.vid = Viewer.load_vid(vid_path)
        self.box_list_list = box_list_list
        self.left_list = left_list
        self.right_list = right_list
        self.vid_ind = vid_path
        self.disp_arr = disp_arr

    @staticmethod
    def load_vid(loc):
        print('loc', loc)
        if type(loc) == str:
            path = loc
        else:
            path = 'clips/' + str(loc) + '.mp4'
        cap = cv.VideoCapture(path)
        frame_list = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_list.append(frame)
            else:
                break
        return np.array(frame_list)

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
        disp = self.disp_arr[frame_ind]
        disp_arr = np.array([disp, 0, disp, 0])
        self.draw_box(frame_ind, self.left_list[frame_ind] + disp_arr, colour_channel=2)

    def draw_right_on_frame(self, frame_ind):
        disp = self.disp_arr[frame_ind]
        disp_arr = np.array([disp, 0, disp, 0])
        self.draw_box(frame_ind, self.right_list[frame_ind] + disp_arr, colour_channel=1)

    def play_vid(self, wait=30, start=0):
        vid = self.vid.astype('uint8')
        for frame in vid[start:]:
            cv.imshow('a', frame)
            cv.waitKey(wait)

    def play_vid_with_disp(self, wait=30, start=0):
        vid = self.vid.astype('uint8')
        min_disp, max_disp = min(self.disp_arr), max(self.disp_arr)

        for i, frame in enumerate(vid[start:]):
            new_frame = np.zeros((frame.shape[0], int(max_disp - min_disp + frame.shape[1]) + 1, 3), dtype='uint8')
            disp = self.disp_arr[i + start]
            new_frame[:, int(max_disp - disp): int(frame.shape[1] + max_disp - disp)] = frame

            cv.imshow(str(self.vid_ind), new_frame)
            cv.waitKey(wait)

    def show_frame(self, frame_ind):
        cv.imshow(str(self.vid_ind), self.vid[frame_ind])
        cv.waitKey(10000)



clip_inds = [i for i in range(1110, 1120, 10)]

#clip_inds = [590]
for ind in clip_inds:
    modes = ['box_list_lists/', 'true_left_lists/', 'true_right_lists/', 'disp_arrs/']
    vid_path = 'clips/' + str(ind) + '.mp4'
    filenames = ['extracted_data/' + mode + str(ind) + '.pt' for mode in modes]
    box_list_list, left_list, right_list, disp_arr = [pickle.load(open(filenames[i], 'rb')) for i in range(4)]
    b = Viewer(vid_path, box_list_list, left_list, right_list, disp_arr)

    b.draw_all_boxes_on_vid()
    b.draw_original_boxes_on_vid()


    b.play_vid_with_disp(20)



