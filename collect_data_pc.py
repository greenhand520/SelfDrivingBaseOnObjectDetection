# -*- coding: utf-8 -*-
# @author: mdmbct
# @date:   3/20/2019 19:28 PM
# @last modified by:
# @last modified time:
import numpy as np
import cv2

import time
import pygame
from pygame.locals import *
import datetime
import os
import sys
from utils import Server, Constant


class VideoStreaming(object):
    def __init__(self):
        self.is_received = True
        pygame.init()
        window = pygame.display.set_mode((320, 100))
        window.fill((246, 246, 246))

    def collect(self, server):

        # firstly, create the img folder
        if not os.path.exists(Constant.BGR_IMG_PATH):
            os.mkdir(Constant.BGR_IMG_PATH)
        if not os.path.exists(Constant.GRAY_IMG_PATH):
            os.mkdir(Constant.GRAY_IMG_PATH)
        if not os.path.exists(Constant.EDGE_IMG_PATH):
            os.mkdir(Constant.EDGE_IMG_PATH)
        path = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "/"
        os.mkdir(path)
        try:
            print("Getting stream from pi...")
            print("Press 'q' to exit")
            # need bytes here
            stream_bytes = b' '
            key_pressed = False
            dir = 4
            frame_num = 0
            start = time.time()
            while self.is_received:
                stream_bytes += server.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    # print(image.shape)  #(width, height, 3)
                    cv2.imshow('image', image)
                    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # img_gb = cv2.GaussianBlur(gray, (3, 3), 0)
                    # edge = cv2.Canny(img_gb, 50, 50)
                    # cv2.imshow("edge", edge)
                    # default direction is 4 (Stop) in every cycle
                    # dir = 4
                    for event in pygame.event.get():
                        # 判断事件是不是按键按下的事件
                        if event.type == pygame.KEYDOWN:
                            key_input = pygame.key.get_pressed()  # 可以同时检测多个按键
                            # 按下前进，保存图片以2开头
                            if key_input[pygame.K_w] and not key_input[pygame.K_LEFT] and not key_input[pygame.K_RIGHT]:
                                # print("Forward")
                                key_pressed = True
                                dir = 2
                            # 按下左键，保存图片以0开头
                            elif key_input[pygame.K_LEFT]:
                                # print("Left")
                                dir = 0
                            # 按下右键，保存图片以1开头
                            elif key_input[pygame.K_RIGHT]:
                                # print("Right")
                                dir = 1
                            # 按下s后退键，保存图片为3开头
                            elif key_input[pygame.K_s]:
                                # print("Backward")
                                dir = 3
                            elif key_input[pygame.K_q]:
                                dir = 4
                                server.send_info(b'4')
                                self.is_received = False
                                end = time.time()
                                print("stop receiving stream...")
                                print("store %d frames in %.2f seconds, %.2ffps" % (
                                    frame_num, end - start, frame_num / (end - start)))
                                time.sleep(0.1)
                                break
                            server.car_control(dir)
                        # 检测按键是不是抬起
                        elif event.type == pygame.KEYUP:
                            key_input = pygame.key.get_pressed()
                            # w键抬起，轮子回正
                            if key_input[pygame.K_w] and not key_input[pygame.K_LEFT] and not key_input[pygame.K_RIGHT]:
                                # print("Forward")
                                dir = 2
                            # s键抬起
                            elif key_input[pygame.K_s] and not key_input[pygame.K_LEFT] and not key_input[
                                pygame.K_RIGHT]:
                                # print("Backward")
                                dir = 3
                            else:
                                dir = 4
                                # print("Stop")
                            server.car_control(dir)
                    if key_pressed:
                        bgr_saved_name = path + str(dir) + "_image" + str(time.time()) + ".jpg"
                        cv2.imwrite(bgr_saved_name, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        frame_num += 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.is_received = False
                        print("stop receiving stream...")
                        break
        finally:
            server.close_server()
            sys.exit(0)


if __name__ == '__main__':
    server = Server(Constant.HOST, Constant.PORT)
    print("Host: ", server.host_name + ' ' + server.host_ip)
    print("Connection from: ", server.client_address)
    vs = VideoStreaming()
    vs.collect(server)
