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
# Left half rectangle
RECTANGLE_L = (0, 0, CENTER_LR, Constant.IMG_HEIGHT)
# Right half rectangle
RECTANGLE_R = (CENTER_LR, 0, Constant.IMG_WIDTH, Constant.IMG_HEIGHT)
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

# detect two path
MIN_PATH_ANGLE_Y = 20
MAX_PATH_ANGLE_Y = 42
MIN_PATH_LR_ANGLE_SUB = 15
MAX_PATH_LR_ANGLE_SUB = 25

# detect single path
MIN_ANGLE_Y = 40
MIN_PIXEL_B = 55
MIN_ANGLE_TURN = 0

MAX_ANGLE_Y = 80
MAX_PIXEL_B = 260
MAX_ANGLE_TURN = 30


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
        self.sign_s_obj = None
        # self.pixels_path_b = 0
        self.sign_s_detected_num = 0
        self.detected_img_num = 0

    def handel_cmd(self):
        return self.last_cmd.split('/')

    def drive(self, image_path=""):
        # reset
        self.path_l_obj = None
        self.path_r_obj = None

        print("current img number: ", self.detected_img_num)
        self.detected_img_num += 1
        # print("is_detect_sigh_lr: ", self.is_detect_sigh_lr)
        is_stop = False
        sign_l_num = self.objects_num[SIGN_L]
        sign_r_num = self.objects_num[SIGN_R]
        # sign_f_num = self.objects_num[SIGN_R]
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
                if o.y_max - o.y_min > 120 and o.x_min > CENTER_TB:
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
                if sign_l.y_max - CENTER_TB >= PIXEL_BETWEEN_CENTER_TB_SIGN_LR_TOP:
                    self.is_detect_sigh_lr = True
            elif sign_r_num != 0:
                print("sign_r num: %d" % sign_r_num)
                sign_r = [k for k, v in self.objects_info.items() if v == SIGN_R][0]
                if sign_r.y_max - CENTER_TB >= PIXEL_BETWEEN_CENTER_TB_SIGN_LR_TOP:
                    self.is_detect_sigh_lr = True
            if path_num == 0:
                if self.handel_cmd()[1] == '1' or '2':
                    cmd = "2/0/0|"
            else:
                cmd = self.handle_path()
        if self.handel_cmd()[0] != '2':
            self.last_cmd = cmd
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

    def handle_path(self):
        """
        detect the path on the image
        :return: the commond will send to pi
        """
        # get all the path obj
        self.path_l_obj = [k for k, v in self.objects_info.items() if v == PATH_L]
        self.path_r_obj = [k for k, v in self.objects_info.items() if v == PATH_R]

        # correct some path obj detected wrong by IOU I define.
        if not self.is_detect_sigh_lr:
            wrong_path_l, wrong_path_r = 0, 0
            for path_l in self.path_l_obj:
                if path_l.cal_rectangles_iou(
                        RECTANGLE_R) >= MAX_IOU:  # it is path_r, remove it from path_l list, add to path_r list
                    wrong_path_l += 1
                    self.path_r_obj.append(path_l)
                    self.path_l_obj.remove(path_l)
            for path_r in self.path_r_obj:
                if path_r.cal_rectangles_iou(RECTANGLE_L) >= MAX_IOU:
                    wrong_path_r += 1
                    self.path_l_obj.append(path_r)
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
        # pixels_center = 0
        pixels_bottom = 0
        angle_y = 0
        for info in path_obj:
            print(info.rect)
            # pixels_center += info.pixels_center_to_img_center()
            # pixels_bottom += info.pixels_center_to_img_bottom()
            pixels_bottom += info.pixels_liner_center_to_image_bottom()
            angle_y += info.get_angle_with_y()
        # pixels_center = pixels_center / path_num
        pixels_bottom = pixels_bottom / path_num
        print("pixels_bottom: ", pixels_bottom)
        angle_y = angle_y / path_num
        print("angle with y: ", angle_y)
        # turn angle, depends on the path line angle with y and the pixel number of line center to img bottom
        # if pixels_bottom
        if angle_y >= MIN_ANGLE_Y:
            angle_turn = str(self.cal_angel_turn_by_pixels_angle(pixels_bottom, angle_y))
            if path_class == PATH_L:
                cmd = "1/2/" + angle_turn + "|"
            else:
                cmd = "1/1/" + angle_turn + "|"
        # keep forward
        else:
            cmd = self.handel_single_path_with_small_angle_y(path_obj, path_class)
        return cmd

    def handel_single_path_with_small_angle_y(self, path_obj, path_class):
        angle_y = 0
        for info in path_obj:
            angle_y += info.get_angle_with_y()
        angle_y = angle_y / len(path_obj)
        if path_class == PATH_L:
            # print("path_l_angle_y: ", angle_y)
            if angle_y <= MIN_PATH_ANGLE_Y:
                print("current car is on the road left, turn right....")
                cmd = "1/2/30|"
            else:
                # MIN_PATH_ANGLE_Y(20) < angle_y < MIN_ANGLE_Y(40)
                angle_turn = str(self.cal_angle_turn_by_path_angle_y(MIN_ANGLE_Y, MIN_PATH_ANGLE_Y, angle_y))
                cmd = "1/2/" + angle_turn + "|"
        else:
            # print("path_r_angle_y: ", angle_y)
            if angle_y <= MIN_PATH_ANGLE_Y:
                print("current car is on the road right, turn left....")
                cmd = "1/1/30|"
            else:
                # MIN_PATH_ANGLE_Y(20) < angle_y < MIN_ANGLE_Y(40)
                angle_turn = str(self.cal_angle_turn_by_path_angle_y(MIN_ANGLE_Y, MIN_PATH_ANGLE_Y, angle_y))
                cmd = "1/1/" + angle_turn + "|"
        return cmd

    def detect_all_path(self, path_l_num, path_r_num):
        cmd = "1/0/0|"
        path_l_angle_y, path_r_angle_y = 0, 0
        angle_turn = 0
        for info in self.path_l_obj:
            path_l_angle_y += info.get_angle_with_y()
        path_l_angle_y = path_l_angle_y / path_l_num
        print("path_l_angle_y: ", path_l_angle_y)
        for info in self.path_r_obj:
            path_r_angle_y += info.get_angle_with_y()
        path_r_angle_y = path_r_angle_y / path_r_num
        print("path_r_angle_y: ", path_r_angle_y)

        sub_angle = path_r_angle_y - path_l_angle_y
        print("sub angle: ", sub_angle)
        if path_l_angle_y <= MIN_PATH_ANGLE_Y or path_r_angle_y >= MAX_PATH_ANGLE_Y:
            print("current car is on the road left, turn right....")
            cmd = "1/2/30|"
        elif path_r_angle_y <= MIN_PATH_ANGLE_Y or path_l_angle_y >= MAX_PATH_ANGLE_Y:
            print("current car is on the road right, turn left....")
            cmd = "1/1/30|"
        else:
            # only depend on the angle with y
            angle_turn = str(
                self.cal_angle_turn_by_path_angle_y(MAX_PATH_LR_ANGLE_SUB, MIN_PATH_LR_ANGLE_SUB, sub_angle))
            if sub_angle > 0 and abs(sub_angle) >= MIN_PATH_LR_ANGLE_SUB:
                print("current car is on the road left, turn right..")
                cmd = "1/2/" + angle_turn + "|"
            elif sub_angle < 0 and abs(sub_angle) >= MIN_PATH_LR_ANGLE_SUB:
                print("current car is on the road right, turn left..")
                cmd = "1/1/" + angle_turn + "|"
        return cmd

    def cal_angle_turn_by_path_angle_y(self, max_angle, min_angle, angle_y):
        """
        cal the angle turn by the line angle with y
        """
        angle = 0
        a = (MAX_ANGLE_TURN - 0) / (
                max_angle - min_angle)
        b = MAX_ANGLE_TURN - max_angle * a
        angle = a * abs(angle_y) + b
        return int(angle)

    def cal_angle_turn_by_pixels_to_bottom(self, max_pixels, min_pixels, pixels_to_b):
        """
        cal the angle turn by the line center pixels to image bottom
        """
        if pixels_to_b <= MIN_PIXEL_B:
            angle_turn = MAX_ANGLE_TURN
        else:
            sum_angle_turn = MAX_ANGLE_TURN - MIN_ANGLE_TURN
            a = sum_angle_turn / (min_pixels - max_pixels)
            b = MAX_ANGLE_TURN - a * min_pixels
            angle_turn = a * pixels_to_b + b
        return int(angle_turn)

    def cal_angel_turn_by_pixels_angle(self, pixels_to_b, angle_y, max_pixel=MAX_PIXEL_B, min_pixel=MIN_PIXEL_B,
                                       max_angle=MAX_ANGLE_Y, min_angle=MIN_ANGLE_Y):
        """
         calculate the angle car will turn by the line angle with y and the line center pixels to image bottom
        """
        y1 = self.cal_angle_turn_by_pixels_to_bottom(max_pixel, min_pixel, pixels_to_b)
        print("angle by pixels: ", y1)
        y2 = self.cal_angle_turn_by_path_angle_y(max_angle, min_angle, angle_y)
        print("angle by angle: ", y2)
        # angle_turn = (1 - pixels_to_b / (MAX_PIXEL_B - MIN_PIXEL_B)) * y1 + angle_y / (MAX_ANGLE_Y - MIN_ANGLE_Y) * y2
        # cal the angle by weights
        angle_turn = (max_pixel - pixels_to_b) / (max_pixel - min_pixel) * y1 + (angle_y - min_angle) / (
                max_angle - min_angle) * y2
        return int(angle_turn)


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
    is_upload = True
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
    except Exception as e:
        print(e)
        server.send_msg('0/0/0|q')
        server.is_received = False
    finally:
        local_path = object_dict_to_csv(objects_info_dict, folder_name)
        if is_upload:
            # pass
            print(local_path)
            Uploader("server_conf.conf", local_path, Constant.SERVER_DATA_PATH).upload()