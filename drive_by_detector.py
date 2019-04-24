# -*- coding: utf-8 -*-
# @author: mdmbct
# @date:   4/18/2019 6:59 PM
# @last modified by: 
# @last modified time: 4/18/2019 6:59 PM
from util import *
from object_detector import ObjectInfo, Detector
import pygame
import cv2
import numpy as np
import sys

SIGN_L, SIGN_R, SIGN_F, SIGN_S, PATH = 1, 2, 3, 4, 5
CENTER_LR = Constant.IMG_WIDTH // 2
CENTER_TB = Constant.IMG_HEIGHT // 2
DISTANCE_BETWEEN_CENTER_TB_SIGN_S_Y_MIN = 10  # the okay distance between image top and bottom center and sign of stop's y_min
MAX_DISTANCE_BETWEEN_CENTER_LR_PATH_SIDE = 100  # the max dis between image left and right center and path's side
MIN_DISTANCE_BETWEEN_CENTER_LR_PATH_SIDE = 60  # # the min dis between image left and right center and path's side
MAX_DISTANCE_BETWEEN_CENTER_TB_SIGN_LR_Y_MAX = 70
MIN_DISTANCE_BETWEEN_CENTER_TB_SIGN_LR_Y_MAX = 50
MAX_DISTANCE_BETWEEN_CENTER_LR_PATH_INSIDE = 25


class Driver(object):

    def __init__(self, detector):
        # self.server = server
        self.detector = detector
        self.is_received = True
        self.objects_info = None
        self.signs_num = {SIGN_L: 0, SIGN_R: 0, SIGN_F: 0, SIGN_S: 0, PATH: 0}
        self.excellent_prediction = 0
        pygame.init()
        window = pygame.display.set_mode((320, 100))
        window.fill((246, 246, 246))
        # server.send_info(b'ss')

    def get_video_stream(self, server):
        try:
            print("Getting stream from pi...")
            print("Press 'q' to exit")
            # need bytes here
            stream_bytes = b' '
            while self.is_received:
                stream_bytes += server.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    objects_info, sign_num = self.detector.detect(image)
                    self.drive(objects_info, sign_num)
        except Exception as e:
            print(e)
        finally:
            server.close_server()
            cv2.destroyAllWindows()
            sys.exit(0)

    def drive(self, objects_info, signs_num, image_path=""):

        is_stop = False
        self.objects_info = objects_info
        self.signs_num = signs_num
        direction_predicted = 2
        sign_l_num = signs_num[SIGN_L]
        sign_r_num = signs_num[SIGN_R]
        sign_f_num = signs_num[SIGN_R]
        sign_s_num = signs_num[SIGN_S]
        path_num = signs_num[PATH]
        print(signs_num)
        if sign_s_num != 0:
            for o in [k for k, v in objects_info.items() if v == SIGN_S]:
                if o.y_min - CENTER_TB >= DISTANCE_BETWEEN_CENTER_TB_SIGN_S_Y_MIN:
                    print(o.y_min)
                    direction_predicted = 4
                    is_stop = True
                break
        if not is_stop:
            if path_num != 0 and (sign_r_num + sign_l_num == 0):
                print("have path sign, no direction sign")
                direction_predicted = self.detect_path()
            elif sign_r_num != 0:
                print("sign_r_num = %d" % sign_r_num)
                sign_r = [k for k, v in objects_info.items() if v == SIGN_R][0]
                direction_predicted = self.handel_sign_lr(sign_r)
            elif sign_l_num != 0:
                print("sign_l_num = %d" % sign_l_num)
                sign_l = [k for k, v in objects_info.items() if v == SIGN_L][0]
                direction_predicted = self.handel_sign_lr(sign_l)
        self.print_drive_direction(direction_predicted, eval(image_path[0]))

    def handel_two_path(self, path0, path1):
        """
        when the image has twp paths, call this function, return the predicted direction
        """
        if (path0.x_min - CENTER_LR >= 0) and (path1.x_min - CENTER_LR >= 0):
            print("two paths are on the image right")
            if ((path0.x_min + path1.x_min) // 2 - CENTER_LR) >= MAX_DISTANCE_BETWEEN_CENTER_LR_PATH_SIDE:
                # print("the car is on the left")
                dire_pre = 1
            elif ((path0.x_min + path1.x_min) // 2 - CENTER_LR) <= MIN_DISTANCE_BETWEEN_CENTER_LR_PATH_SIDE:
                # print("the car is on the right")
                dire_pre = 0
            else:
                dire_pre = 2
        elif (CENTER_LR - path0.x_max >= 0) and (CENTER_LR - path1.x_max >= 0):
            print("two paths are on the image left")
            if (CENTER_LR - (path0.x_max + path1.x_max) // 2) >= MAX_DISTANCE_BETWEEN_CENTER_LR_PATH_SIDE:
                dire_pre = 0
            elif (CENTER_LR - (path0.x_max + path1.x_max) // 2) <= MIN_DISTANCE_BETWEEN_CENTER_LR_PATH_SIDE:
                dire_pre = 1
            else:
                dire_pre = 2
        else:
            print("one path is on the image left, the other one is on the right")
            path_inside_center = abs(path1.x_min - path0.x_max) // 2
            if CENTER_LR - path_inside_center >= MAX_DISTANCE_BETWEEN_CENTER_LR_PATH_INSIDE:  # the car is on the left
                dire_pre = 0
            elif path_inside_center - CENTER_LR >= MAX_DISTANCE_BETWEEN_CENTER_LR_PATH_INSIDE:  # the car is on the right
                dire_pre = 1
            else:
                dire_pre = 2
        return dire_pre

    def detect_path(self):
        """
        detect the path on the image depends on the number of path.
        :return: direction predicted
        """
        dire_pre = 2
        path_num = self.signs_num[PATH]
        if path_num == 1:
            print("path_num = 1")
            path1 = [k for k, v in self.objects_info.items() if v == PATH][0]
            dire_pre = self.handel_two_path(path1, path1)
        elif path_num == 2:
            print("path_num = 2")
            paths = [k for k, v in self.objects_info.items() if v == PATH]
            path0 = paths[0]
            path1 = paths[1]
            dire_pre = self.handel_two_path(path0, path1)
        elif path_num == 3:
            print("path_num = 3")
            paths = [k for k, v in self.objects_info.items() if v == PATH]
            path0, path1, path2 = None, None, None
            for i in range(3):
                path = paths[i]
                print(path)
                # path0 is default on the top left or top right
                if (CENTER_LR - path.x_min > 0 and CENTER_TB - path.y_max > 0) or \
                        (CENTER_LR - path.x_max > 0 and CENTER_TB - path.y_max > 0):
                    path0 = path
                    print("path0 is determined")
                # path1 is on the left
                elif CENTER_LR - path.x_max > 0 and CENTER_TB - path.y_max < 0:
                    path1 = path
                    print("path1 is determined")
                # path2 is on the right
                elif CENTER_LR - path.x_min < 0 and CENTER_TB - path.y_max < 0:
                    path2 = path
                    print("path2 is determined")
            dire_pre = self.handel_two_path(path1, path2)
        return dire_pre

    def handel_sign_lr(self, sign_lr):
        """
         when the image has a sign_l or sign_r, call this function, return the predicted direction
        :param sign_lr: sign_r or sign_l
        """
        if sign_lr.class_name == SIGN_L:
            dire = 0
        else:
            dire = 1
        dis1 = CENTER_TB - sign_lr.y_max
        if dis1 >= MAX_DISTANCE_BETWEEN_CENTER_TB_SIGN_LR_Y_MAX:
            # direction_predicted = 2
            dire_pre = self.detect_path()
        elif dis1 <= MIN_DISTANCE_BETWEEN_CENTER_TB_SIGN_LR_Y_MAX:
            dire_pre = 3
        else:
            dire_pre = dire
        return dire_pre

    def print_drive_direction(self, dire_pre, dire_on_img):
        dire_str = ["left", "right", "forward", "back", "stop"]
        print(dire_str[dire_pre], " ", dire_str[dire_on_img], end="=>")
        if dire_pre == dire_on_img:
            self.excellent_prediction += 1
            print("True")
        else:
            print("False")
        print("* " * 30)

    def send_drive_cmd(self, dire_pre):
        dire_str = ["left", "right", "forward", "back", "stop"]
        print(dire_str[dire_pre])
        # self.server.car_control(dire_pre)


if __name__ == '__main__':
    d = Detector()
    # s = Server()
    driver = Driver(d)
    # driver.get_video_stream(s)
