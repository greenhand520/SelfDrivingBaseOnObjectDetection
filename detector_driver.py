# -*- coding: utf-8 -*-
# @author: mdmbct
# @date:   4/18/2019 6:59 PM
# @last modified by: 
# @last modified time: 4/18/2019 6:59 PM
from util import *
from server import Server
from client import Client
from uploader import Uploader
from object_detector import Detector, ObjectInfo
import pygame
import cv2
import numpy as np
import time
import sys
from pythonds.basic.stack import Stack
import threading
from os import popen
import datetime
import socket
import threadpool
import queue


SIGN_L, SIGN_R, SIGN_F, SIGN_S, PATH_L, PATH_R, SIGN_TL = 1, 2, 3, 4, 5, 6, 7
CENTER_LR = Constant.IMG_WIDTH // 2
CENTER_TB = Constant.IMG_HEIGHT // 2

MAX_IOU = 0.8
LR_RECTANGLE_BORDER = CENTER_LR
RECTANGLE_L = (0, 0, LR_RECTANGLE_BORDER, Constant.IMG_HEIGHT)
RECTANGLE_R = (LR_RECTANGLE_BORDER, 0, Constant.IMG_WIDTH - LR_RECTANGLE_BORDER, Constant.IMG_HEIGHT)
# the okay pixel between image top and bottom center and sign of stop's y_min
PIXEL_BETWEEN_CENTER_TB_SIGN_S_TOP = 10
# the max dis between image left and right center and path's side
MAX_PIXEL_BETWEEN_CENTER_LR_PATH_INSIDE = 150
# the min dis between image left and right center and path's side
MIN_PIXEL_BETWEEN_CENTER_LR_PATH_INSIDE = 60
PIXEL_BETWEEN_CENTER_TB_SIGN_LR_TOP = 10
MAX_PIXEL_BETWEEN_CENTER_TB_SIGN_TL_BOTTOM = 70
MIN_PIXEL_BETWEEN_CENTER_TB_SIGN_TL_BOTTOM = -30
# MAX_PIXEL_BETWEEN_CENTER_LR_PATH_INSIDE = 25
BEST_ANGLE_PATH_WITH_IMAGE_CENTER = 30

MIN_ANGLE_Y = 40
MIN_PIXEL_B = 55
MIN_ANGLE_TURN = 0

MAX_ANGLE_Y = 80
MAX_PIXEL_B = 230
MAX_ANGLE_TURN = 30


class Path(ObjectInfo):

    def __init__(self, rect, class_name):
        self.PATH_L = 5
        self.PATH_R = 6
        self.class_name = class_name
        self.path_rect = rect

        # self.image_array = image_array

    def get_linear_equation(self):
        x1, y1, x2, y2 = self.path_rect
        a = 0
        b = 0
        if self.class_name == self.PATH_L:
            a = (y2 - y1) / (x1 - x2)
            b = y1 - a * x2
        elif self.class_name == self.PATH_R:
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
        return a, b

    def cal_pixel(self):
        """
        calculate the pixel from the liner of CENTER_LR to the point that value of Y coordinate is CENTER_TB on the linear equation
        corresponding to the path.
        the pixel can be + or -.
        """
        a, b = self.get_linear_equation()
        dis = CENTER_LR - (CENTER_TB - b) / a
        return dis

    def pixel_center_to_img_bottom(self):
        return Constant.IMG_HEIGHT - self.get_center()[1]

    def get_point_with_XY(self):
        a, b = self.get_linear_equation()
        return (0, b), (-b / a, 0)

    def cal_rectangles_iou(self, rectangle):
        """
        两个检测框框是否有交叉，如果有交集则返回重叠度（交叉面积 / 路径矩形框的面积） 如果没有交集则返回 0
        IOU here is different from the IOU on the internet.
        It has my definition, to determine the position of the main part of the rectangle on the image.
        说明：每个矩形，从左往右是 x 轴（0~无穷大），从上往下是 y 轴（0~无穷大），从左往右是宽度 w ，从上往下是高度 h
        :param rectangle the first rectangle
        :return: 两个如果有交集则返回重叠度 IOU, 如果没有交集则返回 0
        """
        x1, y1, w1, h1 = rectangle[0], rectangle[1], rectangle[2] - rectangle[0], rectangle[3] - rectangle[1]
        x2, y2, w2, h2 = self.x_min, self.y_min, self.x_max - self.x_min, self.y_max - self.y_min
        if x1 > x2 + w2:
            return 0
        if y1 > y2 + h2:
            return 0
        if x1 + w1 < x2:
            return 0
        if y1 + h1 < y2:
            return 0
        col_int = abs(min(x1 + w1, x2 + w2) - max(x1, x2))
        row_int = abs(min(y1 + h1, y2 + h2) - max(y1, y2))
        overlap_area = col_int * row_int
        area1 = w1 * h1
        area2 = w2 * h2
        if area2 > area1:
            area1, area2 = area2, area1
        # return overlap_area / (area1 + area2 - overlap_area)
        return overlap_area / area2


class Driver(object):

    def __init__(self):
        # self.server = server
        # self.detector = detector
        # self.is_received = True
        self.client = None
        self.objects_info = None
        self.image_array = None
        self.objects_num = {SIGN_L: 0, SIGN_R: 0, SIGN_F: 0, SIGN_S: 0, PATH_L: 0, PATH_R: 0, SIGN_TL: 0}
        self.excellent_prediction = 0
        self.last_cmd = "0/0/0|"  # record the last commond sent to pi
        # self.sign_lr_detected_num = 0
        self.is_detect_sigh_lr = False
        self.path_l_obj = None
        self.path_r_obj = None
        # self.pixels_path_b = 0
        self.sign_s_detected_num = 0
        self.detected_img_num = 0

    def drive(self, image_path=""):
        print("current img number: ", self.detected_img_num)
        print("is_detect_sigh_lr: ", self.is_detect_sigh_lr)
        is_stop = False
        sign_l_num = self.objects_num[SIGN_L]
        sign_r_num = self.objects_num[SIGN_R]
        sign_f_num = self.objects_num[SIGN_R]
        sign_s_num = self.objects_num[SIGN_S]
        path_l_num = self.objects_num[PATH_L]
        path_r_num = self.objects_num[PATH_R]
        sign_tl_num = self.objects_num[SIGN_TL]
        path_num = path_l_num + path_r_num
        print(self.objects_num)
        cmd = "1/0/0|"
        if sign_s_num != 0:
            print("has sign_s")
            for o in [k for k, v in self.objects_info.items() if v == SIGN_S]:
                if o.x_min > CENTER_TB:
                    cmd = "0/0/0|"
                    is_stop = True
                    self.sign_s_detected_num += 1
                break
        elif sign_tl_num != 0:  # detect traffic light
            sign_tl = [k for k, v in self.objects_info.items() if v == SIGN_TL][0]
            if CENTER_TB - sign_tl.y_max <= MIN_PIXEL_BETWEEN_CENTER_TB_SIGN_TL_BOTTOM:
                state = self.detect_sign_tl()
                print("can_run", state)
                if state == 0:
                    is_stop = True
                    cmd = "0/0/0|"
                else:
                    is_stop = False
        if not is_stop:
            if sign_l_num != 0:
                print("sign_l num: %d" % sign_l_num)
                sign_l = [k for k, v in self.objects_info.items() if v == SIGN_L][0]
                # if sign_l.y_max - CENTER_TB >= PIXEL_BETWEEN_CENTER_TB_SIGN_LR_TOP:
                self.is_detect_sigh_lr = True
            elif sign_r_num != 0:
                print("sign_r num: %d" % sign_r_num)
                sign_r = [k for k, v in self.objects_info.items() if v == SIGN_R][0]
                # if sign_r.y_max - CENTER_TB >= PIXEL_BETWEEN_CENTER_TB_SIGN_LR_TOP:
                self.is_detect_sigh_lr = True
            # elif path_r_num + path_l_num != 0:
            #     self.is_detect_sigh_lr = False
            cmd = self.handle_path()
            self.detected_img_num += 1
        return cmd

    def detect_sign_tl(self):
        self.client.send_msg("tl_state")
        time.sleep(0.05)
        # tl_state = 2
        print("traffic light state: ", ("RED", "YELLOW", "GREEN")[tl_state])
        if client.tl_state == self.client.TL_RED or tl_state == self.client.TL_YELLOW:
            return 0
        else:
            return 1
        # return 1

    def handle_path(self):
        """
        detect the path on the image
        :return: the commond will send to pi
        """
        # get all the path obj
        self.path_l_obj = [k for k, v in self.objects_info.items() if v == PATH_L]
        self.path_r_obj = [k for k, v in self.objects_info.items() if v == PATH_R]
        # correct some path obj detected wrong.
        if not self.is_detect_sigh_lr:
            wrong_path_l, wrong_path_r = 0, 0
            for path_l in self.path_l_obj:
                if path_l.x_max > CENTER_LR:  # it is path_r, remove it from path_l list, add to path_r list
                    wrong_path_l += 1
                    # self.path_r_obj.append(path_l)
                    self.path_l_obj.remove(path_l)
            for path_r in self.path_r_obj:
                if path_r.x_min < CENTER_LR:
                    wrong_path_r += 1
                    # self.path_l_obj.append(path_r)
                    self.path_r_obj.remove(path_r)
            print("wrong path_l: %d, wrong path_r: %d" % (wrong_path_l, wrong_path_r))
        path_l_num = len(self.path_l_obj)
        path_r_num = len(self.path_r_obj)

        cmd = "1/0/0|"
        if len(self.path_l_obj) == 0 and len(self.path_r_obj) != 0:  # detected path_r
            cmd = self.detect_single_path(self.path_r_obj, path_r_num, PATH_R)
        elif len(self.path_r_obj) == 0 and len(self.path_l_obj) != 0:  # detected path_l
            print("detected path_l, number is %d" % path_l_num)
            cmd = self.detect_single_path(self.path_l_obj, path_l_num, PATH_L)
        elif len(self.path_l_obj) != 0 and len(self.path_r_obj) != 0:  # detected path_l and path_r
            print("detected path_l: %.2f, path_r: %.2f" % (path_l_num, path_r_num))
            cmd = self.detect_all_path(path_l_num, path_r_num)
        return cmd

    def detect_single_path(self, path_obj, path_num, path_class):
        print(str(("", "", "", "", "", "path_l", "path_r")[path_class]) + " ==>", end=" ")
        pixels_center = 0
        pixels_bottom = 0
        angle_y = 0
        for info in path_obj:
            print(info.rect)
            pixels_center += info.pixels_center_to_img_center()
            pixels_bottom += info.pixels_center_to_img_bottom()
            angle_y += info.get_angle_with_y()
        pixels_center = pixels_center / path_num
        pixels_bottom = pixels_bottom / path_num
        angle_y = angle_y / path_num
        # turn angle, depends on the path line angle with y and the pixel number of line center to img bottom
        # if pixels_bottom
        if angle_y >= MIN_ANGLE_Y:
            angle_turn = self.cal_angle_turn2(pixels_bottom, angle_y)
            if path_class == PATH_L:
                cmd = "1/2/" + angle_turn + "|"
            else:
                cmd = "1/1/" + angle_turn + "|"
        # keep forward
        elif MIN_PIXEL_BETWEEN_CENTER_LR_PATH_INSIDE <= pixels_center <= MAX_PIXEL_BETWEEN_CENTER_LR_PATH_INSIDE:
            # angle_turn = 0
            cmd = "1/1/0|"
        elif pixels_center < MIN_PIXEL_BETWEEN_CENTER_LR_PATH_INSIDE:
            angle_turn = self.cal_angle_turn1(MIN_PIXEL_BETWEEN_CENTER_LR_PATH_INSIDE, 0, abs(pixels_center))
            if path_class == PATH_L:
                cmd = "1/2/" + angle_turn + "|"
            else:
                cmd = "1/1/" + angle_turn + "|"
        elif pixels_center > MAX_PIXEL_BETWEEN_CENTER_LR_PATH_INSIDE:
            angle_turn = self.cal_angle_turn1(200, MAX_PIXEL_BETWEEN_CENTER_LR_PATH_INSIDE, abs(pixels_center))
            if path_class == PATH_L:
                cmd = "1/1/" + angle_turn + "|"
            else:
                cmd = "1/2/" + angle_turn + "|"
        # else:
        #     angle_turn = self.cal_angle_turn1(MAX_PIXEL_BETWEEN_CENTER_LR_PATH_INSIDE,
        #                                       MIN_PIXEL_BETWEEN_CENTER_LR_PATH_INSIDE, abs(pixels_center))
        print("pixels_center: %.2f, pixels_bottom: %.2f, angle with y: %.2f" % (pixels_center,
                                                                                pixels_bottom, angle_y))
        return cmd

    def detect_all_path(self, path_l_num, path_r_num):
        pixels_l, pixels_r = 0, 0
        cmd = "1/0/0|"
        for info in self.path_l_obj:
            pixels_l += info.pixels_center_to_img_center()
        pixels_l = abs(pixels_l / path_l_num)
        for info in self.path_r_obj:
            pixels_r += info.pixels_center_to_img_center()
        pixels_r = abs(pixels_r / path_r_num)
        center = (CENTER_LR + pixels_r - (CENTER_LR - pixels_l)) // 2
        pixels = CENTER_LR - center

        print("pixels from center: ", pixels)
        angle_turn = self.cal_angle_turn1(50, 80, abs(pixels), max_angle_turn=20)
        if pixels > 0:  # the car is on the left, turn right
            cmd = "1/1/" + angle_turn + "|"
        elif pixels < 0:
            cmd = "1/2/" + angle_turn + "|"
        cmd = "1/0/0|"
        return cmd

    def cal_angle_turn1(self, max_pixel, min_pixel, pixel_from_path=-1, max_angle_turn=30):
        """
        calculate the angle car will turn, when the path line angle with y < MIN_ANGLE_Y, angle_turn = a * pixels_from + b
        :param max_pixel: the max pixel
        :param min_pixel: the min pixel
        :param pixel_from_path: the pixel between path  and center of car
        :return:
        """
        angle, angle1, angle2 = 0, 0, 0
        if pixel_from_path != -1:
            # point1 = (min_pixel, MAX_ANGLE_TURN)
            a = (max_angle_turn - 0) / (
                    min_pixel - max_pixel)
            b = max_angle_turn - min_pixel * a
            print("cal of path: ", a, b)
            angle1 = a * pixel_from_path + b
        return str(int(angle1))
        # angle = angle1
        # if pixel_from_sign != -1:
        #     point2 = (MAX_PIXEL_BETWEEN_CENTER_TB_SIGN_LR_BOTTOM, MAX_ANGLE_TURN)
        #     c = (MAX_ANGLE_TURN - 0) / (
        #             MAX_PIXEL_BETWEEN_CENTER_TB_SIGN_LR_BOTTOM - MIN_PIXEL_BETWEEN_CENTER_TB_SIGN_LR_BOTTOM)
        #     d = point2[1] - point2[0] * c
        #     print("cal of sign: ", c, d)
        #     angle2 = c * pixel_from_sign + d
        #     angle = angle2
        # if pixel_from_path != -1 and pixel_from_sign != -1:
        #     return str(round((angle1 + angle2) / 2))
        # else:
        #     return str(round(angle))

    def cal_angle_turn2(self, pixels_to_b, angle_y):
        """
         calculate the angle car will turn, when the path line angle with y >= MIN_ANGLE_Y
        :param pixels_to_b:
        :param angle_y:
        :return:
        """
        # l = MAX_ANGLE_Y - MIN_ANGLE_Y
        # m = MAX_PIXEL_B - MIN_PIXEL_B
        # n = MAX_ANGLE_TURN - MIN_ANGLE_TURN
        # point1 = (MAX_PIXEL_B, MIN_ANGLE_TURN)
        # point2 = (MIN_PIXEL_B, MAX_ANGLE_TURN)
        sum_angle_turn = MAX_ANGLE_TURN - MIN_ANGLE_TURN
        a = sum_angle_turn / (MIN_PIXEL_B - MAX_PIXEL_B)
        b = MAX_ANGLE_TURN - a * MIN_PIXEL_B
        y1 = a * pixels_to_b + b
        print(y1)
        # point3 = (MAX_ANGLE_Y, MIN_ANGLE_TURN)
        # point4 = (MIN_ANGLE_Y, MAX_ANGLE_TURN)
        c = sum_angle_turn / (MIN_ANGLE_Y - MAX_ANGLE_Y)
        d = MAX_ANGLE_TURN - c * MIN_ANGLE_Y
        y2 = c * angle_y + d
        print(y2)
        # angle_turn = (1 - pixels_to_b / (MAX_PIXEL_B - MIN_PIXEL_B)) * y1 + angle_y / (MAX_ANGLE_Y - MIN_ANGLE_Y) * y2
        angle_turn = (MAX_PIXEL_B - pixels_to_b) / (MAX_PIXEL_B - MIN_PIXEL_B) * y1 + (angle_y - MIN_ANGLE_Y) / (
                MAX_ANGLE_Y - MIN_ANGLE_Y) * y2
        return str(int(angle_turn))


class ObjInfoKey(object):
    def __init__(self, image_array):
        self.image_array = image_array


class ThreadPool(object):

    def __init__(self, thread_num=4):
        self.queue = queue.Queue(thread_num)
        for i in range(thread_num):
            self.queue.put(threading.Thread)

    def get_thread(self):
        return self.queue.get()

    def add_thread(self):
        self.queue.put(threading.Thread)


def get_tl_state(client, is_received):
    global tl_state
    while is_received:
        tl_state = eval(client.rec_msg())
        client.tl_state = tl_state
        print("tl_state:", tl_state)


if __name__ == '__main__':

    hostname = socket.gethostname()
    run_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    folder_name = "set_" + hostname + "_" + run_time
    set_path = Constant.DATA_SET_PATH + folder_name
    os.makedirs(set_path)
    sys.stdout = Logger(set_path + "/log.txt", sys.stdout)
    global image_stack, tl_state, is_received
    is_received = True
    tl_state = 2
    # global image_list
    de = Detector()
    d = Driver()
    server = Server()
    client = Client()
    d.client = client
    is_upload = False
    image_stack = server.image_stack
    # pool = ThreadPool(4)
    # while server.is_received:
    #     thread = pool.get_thread()
    #     t = thread(target=server.get_video_stream)
    #     t.start()
    # t.join()
    # video_stream_thread = threading.Thread(target=server.get_video_stream)
    # video_stream_thread.start()
    # video_stream_thread.join()
    video_stream_thread = threading.Thread(target=server.get_video_stream)
    video_stream_thread.setDaemon(True)
    video_stream_thread.start()
    # video_stream_thread.join()
    image_list = server.image_list
    tl_state_thread = threading.Thread(target=get_tl_state, args=(client, is_received))
    tl_state_thread.setDaemon(True)
    tl_state_thread.start()
    objects_info_dict = {}
    pygame.init()
    window = pygame.display.set_mode((320, 100))
    window.fill((246, 246, 246))
    try:
        start = time.time()
        while server.is_received:
            if not image_stack.isEmpty():
                # if len(image_list) != 0:
                objects_info, objects_num, image_array = de.detect(image_stack.pop())
                # objects_info, objects_num, image_array = de.detect(image_list[0])
                d.objects_info = objects_info
                d.objects_num = objects_num
                d.image_array = image_array
                cmd = d.drive()
                server.send_msg(cmd)
                objects_info_dict[ObjInfoKey(image_array)] = objects_info
                if d.sign_s_detected_num == 2:
                    server.send_msg('0/0/0|q')
                    server.is_received = False
                    break
                print("commond sent to pi: ", cmd)
                print("* " * 50)
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        key_input = pygame.key.get_pressed()
                        if key_input[pygame.K_q]:
                            server.send_msg('0/0/0|q')
                            server.is_received = False
                cv2.waitKey(1)
            is_received = server.is_received
        end = time.time()
        print(end - start)
    finally:
        local_path = object_dict_to_csv(objects_info_dict, folder_name)
        if is_upload:
            # pass
            print(local_path)
            Uploader("server_conf.conf", local_path, Constant.SERVER_DATA_PATH).upload()
