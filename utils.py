# -*- coding: utf-8 -*-
# @author: mdmbct
# @date:   4/5/2019 8:28 PM
# @last modified by: 
# @last modified time: 4/5/2019 8:28 PM
import socket
import glob
import cv2
import os
import numpy as np
from PIL import Image


class Constant(object):
    HOST = "0.0.0.0"
    PORT = 8000
    IMG_WIDTH = 160
    IMG_HEIGHT = 120
    IMG_CHANNELS = 3
    BGR_IMG_PATH = "bgr_data/"
    EDGE_IMG_PATH = "edge_img/"
    GRAY_IMG_PATH = "gray_img/"
    NPZ_PATH = "old_npz/"
    MODEL_PATH = "model/"
    TRAIN_LOG_PATH = "logs/"


class Server(object):
    def __init__(self, host, port):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(0)
        self.connection, self.client_address = self.server_socket.accept()
        self.connector = self.connection
        self.connection = self.connection.makefile('rwb')
        self.host_name = socket.gethostname()
        self.host_ip = socket.gethostbyname(self.host_name)

    def send_info(self, info):
        self.connector.send(info)

    def receive_info(self):
        return str(self.server_socket.recv(1024), encoding="utf-8")

    def close_server(self):
        self.connection.close()
        self.server_socket.close()

    def car_control(self, direction):

        if direction == 0:  # left
            self.send_info(b'0')
        elif direction == 1:  # right
            self.send_info(b'1')
        elif direction == 3:  # back
            self.send_info(b'3')
        elif direction == 4:  # stop
            self.send_info(b'4')
        else:  # front
            self.send_info(b'2')


def convert_to_edge_img(rgb_img_path):
    if not os.path.exists(Constant.EDGE_IMG_PATH):
        os.mkdir(Constant.EDGE_IMG_PATH)
    rgb_imgs = glob.glob(rgb_img_path + "*.jpg")
    for rgb_img in rgb_imgs:
        image = np.array(Image.open(rgb_img))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_gb = cv2.GaussianBlur(gray, (3, 3), 0)
        edge = cv2.Canny(img_gb, 50, 50)
        edge_img_path = rgb_img.split('\\')[1]
        cv2.imwrite(Constant.EDGE_IMG_PATH + edge_img_path, edge, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imshow("edge", edge)
        cv2.waitKey(1)

    print("all rgb images have converted to edge image")


if __name__ == '__main__':
    convert_to_edge_img(Constant.BGR_IMG_PATH)