# -*- coding: utf-8 -*-
# @author: mdmbct
# @date:   4/5/2019 8:28 PM
# @last modified by: 
# @last modified time: 4/5/2019 8:28 PM
import random
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
import math
from pythonds.basic.stack import Stack
import time


class Constant(object):
    IMG_WIDTH = 400
    IMG_HEIGHT = 300
    IMG_CHANNELS = 3
    BGR_IMG_PATH = "bgr_data/"
    NPZ_PATH = "E:/CommonFiles/Projects/PycharmProjects/self_drive/old_npz/"
    TRAIN_LOG_PATH = "logs/"
    LABEL_CSV_PATH = "label_csv/"
    TFRECORD_PATH = "tfrecord/"
    MODEL_PATH = "model/"
    DATA_SET_PATH = "upload_to_server/data_set/"
    MODEL_UPLOADED_PATH = "upload_to_server/model/"
    SERVER_DATA_PATH = "/root/self_driving_data_set/static/source/data_set/"


class Logger(object):

    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def convert_to_edge_img(rgb_img_path):
    """
    useless, only do not want to remove
    :param rgb_img_path:
    :return:
    """
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


def move_image(old_folders, old_folders_feature, new_folder):
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
        shutil.move(image, new_folder)
    print("all images have moved to " + new_folder)


def random_copyorcut_file(old_folders, new_folder, rate, op='cut'):
    imgs = glob.glob(old_folders + "*.jpg")
    xmls = glob.glob(old_folders + "*.xml")
    cuted_len = round(len(xmls) * rate)
    cuted_num = []
    print(len(imgs), " ", len(xmls))
    while len(cuted_num) <= cuted_len:
        number = random.randint(0, len(xmls) - 1)
        if number not in cuted_num:
            cuted_num.append(number)
    print(cuted_num)
    for i in cuted_num:
        print(i)
        xml_file = xmls[i]
        try:
            if op == 'cut':
                shutil.move(xml_file, new_folder)
                img_file = xml_file.split('/')[-1].split('x')[0] + "jpg"
                shutil.move(img_file, new_folder)
            else:
                shutil.copy(xml_file, new_folder)
                img_file = xml_file.split('/')[-1].split('x')[0] + "jpg"
                shutil.copy(img_file, new_folder)
        finally:
            pass


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


def cal_color_distance(rgb1, rgb2):
    r1, g1, b1 = rgb1
    r2, g2, b2 = rgb2
    rmean = (r1 + r2) / 2
    r = r1 - r2
    g = g1 - g2
    b = b1 - b2
    return math.sqrt((2 + rmean / 256) * (r ** 2) + 4 * (g ** 2) + (2 + (255 - rmean) / 256) * (b ** 2))


def object_dict_to_csv(obj_dict, folder_name):
    # SIGN_L, SIGN_R, SIGN_F, SIGN_S, PATH_L, PATH_R = 1, 2, 3, 4, 5, 6
    sign = ("bgi", "sign_l", "sign_r", "sign_f", "sign_s", "path_t", "path_r", "sign_tl")
    folder_path = Constant.DATA_SET_PATH + folder_name
    upload_img_path = folder_path + "/images/"
    os.makedirs(upload_img_path)
    data = pd.DataFrame()
    # filename	width	height	class	xmin	ymin	xmax	ymax
    i = 0
    for item in obj_dict.items():
        img_file_name = "image_" + str(i) + ".jpg"
        cv2.imwrite(upload_img_path + img_file_name, item[0].image_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
        for key in item[1].keys():
            df = pd.DataFrame(
                {"filename": img_file_name, "width": key.x_max - key.x_min,
                 "height": key.y_max - key.y_min,
                 "class": sign[key.class_name], "xmin": key.x_min, "ymin": key.y_min, "xmax": key.x_max,
                 "ymax": key.y_max}, index=[0])
            data = data.append(df)
        i += 1
    file_name = folder_name
    shutil.copy("E:/upload_to_server/pet_label_map.pbtxt", folder_path)
    data.to_csv(folder_path + "/" + file_name + ".csv", index=False)
    return folder_path


if __name__ == '__main__':
    pass
    xml_to_csv("sign_test", "E:/CommonFiles/git/models/research/object_detection/test_img_20190516/",
               "E:/CommonFiles/git/models/research/object_detection/")
    xml_to_csv("sign_train", "E:/CommonFiles/git/models/research/object_detection/train_img_20190516/",
               "E:/CommonFiles/git/models/research/object_detection/")
    # move_image(Constant.BGR_IMG_PATH, "2019-05-15_*", "E:/CommonFiles/git/models/research/object_detection/train_img_20190515_2/")
    # random_copyorcut_file("E:\CommonFiles\git\models\\research\object_detection\\train_img_20190516\\",
    #                 "E:\CommonFiles\git\models\\research\object_detection\\test_img_20190516\\", 0.3, op='copy')
