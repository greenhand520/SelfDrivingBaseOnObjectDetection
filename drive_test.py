import glob
import time
import threading
import random
from PIL import Image
import numpy as np
# import car_control
from keras.models import load_model
import tensorflow as tf
from utils import Constant
global correct_pre


def get_max_prob_num(predictions_array):
    """to get the integer of predition, instead of digit number"""

    prediction_edit = np.zeros([1, 5])
    for i in range(0, 5):
        if predictions_array[0][i] == predictions_array.max():
            prediction_edit[0][i] = 1
            return i
    return 2


def control_car_simulation(action_num):
    if action_num == 0:
        print("Left")
        # time.sleep(0.25)
    elif action_num == 1:
        print("Right")
        # time.sleep(0.25)
    elif action_num == 2:

        print('Forward')
    elif action_num == 3:
        print('Backward')
    else:
        print('Stop')


class ImageProcessor():
    def __init__(self,img, img_dir):
        super(ImageProcessor, self).__init__()
        self.img = img
        # self.event = threading.Event()
        self.terminated = False
        # self.owner = owner
        self.img_dir = img_dir
        # self.start()

    def run(self):
        global latest_time, model, graph
        image = Image.open(self.img)
        image_np = np.array(image)
        camera_data_array = np.expand_dims(image_np, axis=0)
        current_time = time.time()
        if current_time > latest_time:
            if current_time - latest_time > 1:
                print("*" * 30)
                print(current_time - latest_time)
                print("*" * 30)
            latest_time = current_time
            with graph.as_default():
                predictions_array = model.predict(camera_data_array, batch_size=20, verbose=1)
            print(predictions_array)
            action_num = get_max_prob_num(predictions_array)
            control_car_simulation(action_num)
            print("img_dir: ", self.img_dir, " ", "pre_dir: ", action_num)
            if action_num == int(self.img_dir):
                return True
            else:
                return False


def main():
    """get data, then predict the data, edited data, then control the car"""
    global model, graph

    model_loaded = glob.glob(Constant.MODEL_PATH + '*.h5')
    for single_mod in model_loaded:
        model = load_model(single_mod)
    graph = tf.get_default_graph()

    imgs = glob.glob(Constant.BGR_IMG_PATH + "*")
    count = 5683
    print(len(imgs))
    correct_pre = 0
    while count > 0:
        count = count - 1
        img = imgs[random.randint(0, len(imgs) - 1)]
        img_dir = img[9]
        if ImageProcessor(img, img_dir).run():
            correct_pre += 1
    print("correct prediction times is %d" % correct_pre)
    print("accuracy is %.3f" % (correct_pre / 5683))


if __name__ == '__main__':
    global latest_time
    latest_time = time.time()
    main()
