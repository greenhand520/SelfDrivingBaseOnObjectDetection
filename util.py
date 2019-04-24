# -*- coding: utf-8 -*-
# @author: mdmbct
# @date:   4/5/2019 8:28 PM
# @last modified by: 
# @last modified time: 4/5/2019 8:28 PM
import socket
import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
import sys
import os
import pandas as pd
from PIL import Image
import shutil

HOST = "0.0.0.0"
PORT = 8000


class Constant(object):
    IMG_WIDTH = 300
    IMG_HEIGHT = 300
    IMG_CHANNELS = 3
    BGR_IMG_PATH = "bgr_data/"
    NPZ_PATH = "E:/CommonFiles/Projects/PycharmProjects/self_drive/old_npz/"
    TRAIN_LOG_PATH = "logs/"
    LABEL_CSV_PATH = "label_csv/"
    TFRECORD_PATH = "tfrecord/"
    MODEL_PATH = "model/"


class Server(object):

    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((HOST, PORT))
        self.server_socket.listen(0)
        self.connection, self.client_address = self.server_socket.accept()
        self.connector = self.connection
        self.connection = self.connection.makefile('rwb')
        self.host_name = socket.gethostname()
        self.host_ip = socket.gethostbyname(self.host_name)
        self.DIRE_LEFT = 0
        self.DIRE_RIGHT = 1
        self.DIRE_FORWARD = 2
        self.DIRE_BACK = 3
        self.DIRE_STOP = 4
        self.CAN_GET_STREAM = b'ss'

    def send_info(self, info):
        self.connector.send(info)

    def receive_info(self, stream_bytes):

        current_rec = self.connection.read(1024)
        stream_bytes += current_rec
        first = stream_bytes.find(b'\xff\xd8')
        last = stream_bytes.find(b'\xff\xd9')
        print(first, " ", last)

        if first is None and last is None:  # is not image
            return stream_bytes, str(current_rec, encoding="utf-8")
        elif first != -1 and last != -1:  # is image
            jpg = stream_bytes[first:last + 2]
            stream_bytes2 = stream_bytes[last + 2:]
            image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            return stream_bytes2, image
        else:
            return stream_bytes, None

    def close_server(self):
        self.connection.close()
        self.server_socket.close()

    def car_control(self, direction):
        # only used in old version used CNN to drive car, new version used object detection does not use it
        if direction == self.DIRE_LEFT:  # left
            self.send_info(b'0')
        elif direction == self.DIRE_RIGHT:  # right
            self.send_info(b'1')
        elif direction == self.DIRE_BACK:  # back
            self.send_info(b'3')
        elif direction == self.DIRE_STOP:  # stop
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


def copy_image(old_folders, old_folders_feature, new_folder):
    folders = glob.glob(old_folders + old_folders_feature)
    images = []
    for folder in folders:
        images += glob.glob(folder + "/*.jpg")
    images_num = len(images)
    if images_num == 0:
        print("no images exit...")
        sys.exit(0)
    else:
        print("all have %d images in %s" % (images_num, old_folders + old_folders_feature))
    for image in images:
        shutil.copy(image, new_folder)
    print("all images have copied to " + new_folder)


def xml_to_csv(label_name, xml_path, csv_path):
    xml_list = []
    xml_files = glob.glob(xml_path + '*.xml')
    if len(xml_files) == 0:
        print("No xml file in " + xml_path)
        sys.exit(0)
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv((csv_path + label_name + ".csv"), index=None)
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    pass
    xml_to_csv("sign_all_300x300", "E:/CommonFiles/git/models/research/object_detection/label_img/",
               "E:/CommonFiles/git/models/research/object_detection/label_csv/")
    # copy_image(Constant.BGR_IMG_PATH, "2019-*", "E:/CommonFiles/git/models/research/object_detection/label_img/")
