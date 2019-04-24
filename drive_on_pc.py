# -*- coding: utf-8 -*-
# @author: mdmbct
# @date:   4/12/2019 3:35 PM
# @last modified by: 
# @last modified time: 4/12/2019 3:35 PM

import glob
import time
import threading
import random
from PIL import Image
import numpy as np
from keras.models import load_model
import tensorflow as tf
from util import *
import pygame
import datetime
import os
import cv2
import sys


def get_max_prob_num(predictions_array):
    """to get the integer of predition, instead of digit number"""

    prediction_edit = np.zeros([1, 5])
    for i in range(0, 5):
        if predictions_array[0][i] == predictions_array.max():
            prediction_edit[0][i] = 1
            return i
    return 2


class Driver(object):

    def __init__(self, sever, model, graph):
        self.server = sever
        self.is_received = True
        self.model = model
        self.graph = graph
        self.img_array = []
        pygame.init()
        window = pygame.display.set_mode((320, 100))
        window.fill((246, 246, 246))

    def get_video_stream(self):

        try:
            # path = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "/"
            # os.mkdir(path)
            print("Getting stream from pi...")
            print("Press 'q' to exit")
            # need bytes here
            stream_bytes = b' '
            frame_num = 0
            start = time.time()
            while self.is_received:
                stream_bytes += self.server.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    self.img_array.append(image)
                    cv2.imshow('image', image)
                    self.predict_direction(image)
        except Exception as e:
            print(e)
        finally:
            self.server.close_server()
            sys.exit(0)

    def predict_direction(self, img):
        # image = Image.open(img)
        image_np = np.array(img)
        camera_data_array = np.expand_dims(image_np, axis=0)
        with self.graph.as_default():
            predictions_array = self.model.predict(camera_data_array, batch_size=20, verbose=1)
        print(predictions_array)
        action_num = get_max_prob_num(predictions_array)
        print("direction: ", action_num)
        self.server.car_control(action_num)


def load_models(model_path):
    global model, graph
    model_loaded = glob.glob(model_path + '*.h5')
    for single_mod in model_loaded:
        model = load_model(single_mod)
    graph = tf.get_default_graph()


def main():
    load_models(Constant.MODEL_PATH)
    server = Server()
    driver = Driver(server, model, graph)
    driver.get_video_stream()


if __name__ == '__main__':
    main()
