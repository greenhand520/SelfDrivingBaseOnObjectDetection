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
from util import Server, Constant


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
        path = Constant.BGR_IMG_PATH + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "/"
        os.makedirs(path)
        try:
            print("Getting stream from pi...")
            print("Press 'q' to exit")
            # need bytes here
            stream_bytes = b''
            key_pressed = False
            dire = server.DIRE_STOP
            cmd = b'4/90/'
            frame_num = 0
            start = time.time()
            while self.is_received:
                stream_bytes += server.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                # print(first, " ", last)
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    cv2.imshow('image', image)
                    # return0, return1 = server.receive_info(stream_bytes)
                    # if str(type(return1)) == "<class 'str'>":  # return is not image
                    #     rec = return1
                    #     if rec == "break":
                    #         server.close_server()
                    #         print("Client break ")
                    #         sys.exit(0)
                    # elif str(type(return1)) == "<class 'numpy.ndarray'>":   # return is image
                    #     image = return1
                    #     cv2.imshow('image', image)
                    # stream_bytes = return0
                    for event in pygame.event.get():
                        # 判断事件是不是按键按下的事件
                        if event.type == pygame.KEYDOWN:
                            key_input = pygame.key.get_pressed()  # 可以同时检测多个按键
                            # 按下前进，保存图片以2开头
                            if key_input[pygame.K_w] and not key_input[pygame.K_LEFT] and not key_input[pygame.K_RIGHT]:
                                # print("Forward")
                                key_pressed = True
                                dire = server.DIRE_FORWARD
                                cmd = b'2/90/'  # dire/angle/other
                            # 按下左键，保存图片以0开头
                            elif key_input[pygame.K_LEFT]:
                                # print("Left")
                                dire = server.DIRE_LEFT
                                cmd = b'0/30/'
                            # 按下右键，保存图片以1开头
                            elif key_input[pygame.K_RIGHT]:
                                # print("Right")
                                dire = server.DIRE_RIGHT
                                cmd = b'1/30'
                            # 按下s后退键，保存图片为3开头
                            elif key_input[pygame.K_s]:
                                # print("Backward")
                                dire = server.DIRE_BACK
                                cmd = b'3/90'
                            elif key_input[pygame.K_q]:
                                cmd = b'4/90/q'
                                self.is_received = False
                                end = time.time()
                                print("stop receiving stream...")
                                print("store %d frames in %.2f seconds, %.2ffps" % (
                                    frame_num, end - start, frame_num / (end - start)))
                                time.sleep(0.1)
                                break
                        # 检测按键是不是抬起
                        elif event.type == pygame.KEYUP:
                            key_input = pygame.key.get_pressed()
                            # w键抬起，轮子回正
                            if key_input[pygame.K_w] and not key_input[pygame.K_LEFT] and not key_input[pygame.K_RIGHT]:
                                # print("Forward")
                                dire = server.DIRE_FORWARD
                                cmd = b'2/90/'
                            # s键抬起
                            elif key_input[pygame.K_s] and not key_input[pygame.K_LEFT] and not key_input[
                                pygame.K_RIGHT]:
                                # print("Backward")
                                dire = server.DIRE_BACK
                                cmd = b'3/90'
                            else:
                                cmd = b'4/90'
                                # print("Stop")
                                dire = server.DIRE_STOP
                        server.send_info(cmd)
                    if key_pressed:
                        bgr_saved_name = path + str(dire) + "_image" + str(time.time()) + ".jpg"
                        cv2.imwrite(bgr_saved_name, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        frame_num += 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.is_received = False
                        print("stop receiving stream...")
                        break
        except Exception as e:
            print(e)
        finally:
            server.close_server()
            sys.exit(0)


if __name__ == '__main__':
    server = Server()
    print("Host: ", server.host_name + ' ' + server.host_ip)
    print("Connection from: ", server.client_address)
    vs = VideoStreaming()
    vs.collect(server)
