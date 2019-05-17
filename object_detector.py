# -*- coding: utf-8 -*-
# @author: mdmbct
# @date:   4/18/2019 10:13 AM
# @last modified by: 
# @last modified time: 4/18/2019 10:13 AM
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import glob
import math
import time
from util import *

MODEL_PATH = "D:\object detector\ob20190515\model\inference_graph\\"
# MODEL_PATH = "E:\CommonFiles\git\models\\research\object_detection\inference_graph_ssdlite\\"
TRAINING_PATH = "D:\object detector\ob20190515\model\\training\\"
# TRAINING_PATH = "E:\CommonFiles\git\models\\research\object_detection\\training_ssdlite"
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(MODEL_PATH, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(TRAINING_PATH, 'pet_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 7

# Y-axis unit vector
VECTOR_Y = (0, 1)

SIGN_L, SIGN_R, SIGN_F, SIGN_S, PATH_L, PATH_R, SIGN_TL = 1, 2, 3, 4, 5, 6, 7


class Detector(object):
    def __init__(self):
        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        # Load the TensorFlow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=detection_graph)
        # Input tensor is the image
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # Number of objects detected
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def detect(self, img_array):
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image_expanded = np.expand_dims(img_array, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})
        # Draw the results of the detection
        img_array2 = np.array(img_array)
        vis_util.visualize_boxes_and_labels_on_image_array(
            img_array,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=1,
            min_score_thresh=0.5)

        s_boxes = boxes[scores > 0.5]
        s_classes = classes[scores > 0.5]
        s_scores = scores[scores > 0.5]
        objects = {}  # dict of labels, key is label object, value is its class_name
        sign_s_num, sign_l_num, sign_r_num, sign_f_num, path_l_num, path_r_num, sign_tl_num = 0, 0, 0, 0, 0, 0, 0
        for i in range(len(s_classes)):
            info = ObjectInfo(i,
                              s_classes[i],
                              (round(s_boxes[i][1] * Constant.IMG_WIDTH), round(s_boxes[i][0] * Constant.IMG_HEIGHT),
                               round(s_boxes[i][3] * Constant.IMG_WIDTH), round(s_boxes[i][2] * Constant.IMG_HEIGHT)),
                              s_scores[i])
            objects[info] = info.class_name
            if info.class_name == SIGN_F:
                sign_f_num += 1
            elif info.class_name == SIGN_S:
                sign_s_num += 1
            elif info.class_name == SIGN_L:
                sign_l_num += 1
            elif info.class_name == SIGN_R:
                sign_r_num += 1
            elif info.class_name == PATH_L:
                path_l_num += 1
            elif info.class_name == PATH_R:
                path_r_num += 1
            elif info.class_name == SIGN_TL:
                sign_tl_num += 1
        cv2.imshow('Object detector', img_array)
        # cv2.waitKey(1)
        return objects, {SIGN_L: sign_l_num, SIGN_R: sign_r_num, SIGN_F: sign_f_num, SIGN_S: sign_s_num,
                         PATH_L: path_l_num, PATH_R: path_r_num, SIGN_TL: sign_tl_num}, img_array


class ObjectInfo(object):
    def __init__(self, id, class_name, rect, score):
        self.id = id
        self.class_name = int(class_name)
        self.x_min = int(rect[0])
        self.y_min = int(rect[1])
        self.x_max = int(rect[2])
        self.y_max = int(rect[3])
        self.rect = rect
        self.score = score

    def set_rect(self, rect):
        self.x_min = int(rect[0])
        self.y_min = int(rect[1])
        self.x_max = int(rect[2])
        self.y_max = int(rect[3])

    def get_top_left(self):
        return self.x_min, self.y_min

    def get_top_right(self):
        return self.x_max, self.y_min

    def get_bottom_left(self):
        return self.x_min, self.y_max

    def get_bottom_right(self):
        return self.x_max, self.y_max

    def get_center(self):
        return (self.x_max + self.x_min) / 2, (self.y_max + self.y_min) / 2

    def pixels_center_to_img_bottom(self):
        return Constant.IMG_HEIGHT - self.get_center()[1]

    def pixels_center_to_img_center(self):
        return Constant.IMG_WIDTH / 2 - self.get_center()[0]

    def get_linear_equation(self):
        x_min, y_min, x_max, y_max = self.rect
        a = 0
        b = 0
        if self.class_name == PATH_L:
            a = (y_max - y_min) / (x_min - x_max)
            b = y_min - a * x_max
        elif self.class_name == PATH_R:
            a = (y_max - y_min) / (x_max - x_min)
            b = y_min - a * x_min
        return a, b

    def pixels_liner_center_to_image_bottom(self):
        a, b = self.get_linear_equation()
        return Constant.IMG_HEIGHT - Constant.IMG_WIDTH / 2 * a + b

    def get_point_with_xy(self):
        a, b = self.get_linear_equation()
        return (0, b), (-b / a, 0)

    def pixels_line_to_img_center(self):
        a, b = self.get_linear_equation()
        dis = Constant.IMG_WIDTH / 2 - (Constant.IMG_HEIGHT / 2 - b) / a
        return dis

    def get_size(self):
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def get_vector(self):
        """
        get the two vectors of two diagonals, if the path is path_l, the shapes likes this /,
        if the path is path_r, the shape likes this \
        :return:
        """
        if self.class_name == PATH_L:
            return self.x_min - self.x_max, self.y_max - self.y_min
        elif self.class_name == PATH_R:
            return self.x_max - self.x_min, self.y_max - self.y_min

    def get_vector_length(self):
        return math.sqrt(pow((self.x_max - self.x_min), 2) + pow((self.y_max - self.y_min), 2))

    def get_angle_with_y(self):
        return round(
            (np.arccos(np.dot(self.get_vector(), VECTOR_Y) / (self.get_vector_length() * 1)) * 180) / np.pi,
            1)

    def cal_rectangles_iou(self, rectangle):
        """
        两个检测框框是否有交叉，如果有交集则返回重叠度（交叉面积 / 路径矩形框的面积） 如果没有交集则返回 0
        IOU here is different from the IOU on the internet.
        It's just my definition, to determine the position of the main part of the rectangle on the image.
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