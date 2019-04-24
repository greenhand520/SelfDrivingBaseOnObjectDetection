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
import drive_by_detector as driver
from util import *

MODEL_PATH = "E:/CommonFiles/git/models/research/object_detection/inference_graph/"
TRAINING_PATH = "E:/CommonFiles/git/models/research/object_detection/training/"
# TRAINING_PATH = "H:\\training"
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
NUM_CLASSES = 5

# Y-axis unit vector
VECTOR_Y = (0, 1)


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
        vis_util.visualize_boxes_and_labels_on_image_array(
            img_array,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=0.80)

        s_boxes = boxes[scores > 0.5]
        s_classes = classes[scores > 0.5]
        s_scores = scores[scores > 0.5]
        objects = {}  # dict of labels, key is label object, value is its class_name
        sign_s_num, sign_l_num, sign_r_num, sign_f_num, path_num = 0, 0, 0, 0, 0
        for i in range(len(s_classes)):
            info = ObjectInfo(i, s_classes[i], s_boxes[i][1] * Constant.IMG_WIDTH, s_boxes[i][3] * Constant.IMG_WIDTH,
                              s_boxes[i][0] * Constant.IMG_HEIGHT, s_boxes[i][2] * Constant.IMG_HEIGHT, s_scores[i])
            objects[info] = info.class_name
            if info.class_name == SIGN_F:
                sign_f_num += 1
            elif info.class_name == SIGN_S:
                sign_s_num += 1
            elif info.class_name == SIGN_L:
                sign_l_num += 1
            elif info.class_name == SIGN_R:
                sign_r_num += 1
            elif info.class_name == PATH:
                path_num += 1
        cv2.imshow('Object detector', img_array)
        cv2.waitKey(1)
        return objects, {SIGN_L: sign_l_num, SIGN_R: sign_r_num, SIGN_F: sign_f_num, SIGN_S: sign_s_num, PATH: path_num}


class ObjectInfo(object):
    def __init__(self, id, class_name, x_min, x_max, y_min, y_max, score):
        self.id = id
        self.class_name = int(class_name)
        self.x_min = int(round(x_min))
        self.x_max = int(round(x_max))
        self.y_min = int(round(y_min))
        self.y_max = int(round(y_max))
        self.score = score

    def get_top_left(self):
        return self.x_min, self.y_min

    def get_top_right(self):
        return self.x_max, self.y_min

    def get_bottom_left(self):
        return self.x_min, self.y_max

    def get_bottom_right(self):
        return self.x_max, self.y_max

    def get_size(self):
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def get_vector(self):
        """
        get the two vectors of two diagonals
        :return:
        """
        return (self.x_max - self.x_min, self.y_max - self.y_min), (self.x_min - self.x_max, self.y_max - self.y_min)

    def get_vector_length(self):
        return math.sqrt(pow((self.x_max - self.x_min), 2) + pow((self.y_max - self.y_min), 2))

    def get_angle_with_y(self):
        return np.dot(self.get_vector()[0], VECTOR_Y) / (self.get_vector_length() * 1), \
               np.dot(self.get_vector()[1], VECTOR_Y) / (self.get_vector_length() * 1)


if __name__ == '__main__':
    images = []
    d = Detector()
    detector_driver = driver.Driver(d)
    image_paths = glob.glob(Constant.BGR_IMG_PATH + "2019-*")
    for image_path in image_paths:
        images += glob.glob(image_path + "/*.jpg")
    image_num = len(images)
    print(image_num)
    start = time.time()
    SIGN_L, SIGN_R, SIGN_F, SIGN_S, PATH = 1.0, 2.0, 3.0, 4.0, 5.0
    i = 0
    for image in images:
        i += 1
        objects_info, sign_num = d.detect(cv2.imread(image))
        # print(image)
        # print(image.split('\\')[2][0])
        detector_driver.drive(objects_info, sign_num, image.split('\\')[2])
    end = time.time()
    print(detector_driver.excellent_prediction / image_num)
    cv2.destroyAllWindows()
    print(end - start)
