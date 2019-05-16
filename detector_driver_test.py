# -*- coding: utf-8 -*-
# @author: mdmbct
# @date:   4/28/2019 8:05 PM
# @last modified by: 
# @last modified time: 4/28/2019 8:05 PM
from object_detector import Detector
from detector_driver import *
import glob
import time
import cv2
from PIL import Image
import numpy as np
from uploader import Uploader


def test_drive():
    images = glob.glob(
        "bgr_data/2019-05-09_04-48-50/" + "*.jpg")
    image_num = len(images)
    print(image_num)
    hostname = socket.gethostname()
    run_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    folder_name = "set_" + hostname + "_" + run_time
    set_path = Constant.DATA_SET_PATH + folder_name
    os.makedirs(set_path)
    sys.stdout = Logger(set_path + "/log.txt", sys.stdout)
    de = Detector()
    d = Driver()
    # server = Server()
    # client = Client()
    # d.client = client
    is_upload = True
    # video_stream_thread = threading.Thread(target=server.get_video_stream)
    # video_stream_thread.setDaemon(True)
    # video_stream_thread.start()
    # tl_state_thread = threading.Thread(target=client.get_tl_state)
    # tl_state_thread.setDaemon(True)
    # tl_state_thread.start()
    objects_info_dict = {}
    start = time.time()
    # i = 0
    for image in images:
        print(image.split('/')[-1])
        objects_info, objects_num, image_array = de.detect(cv2.imread(image))
        d.objects_info = objects_info
        d.objects_num = objects_num
        d.image_array = image_array
        cmd = d.drive()
        # server.send_msg(cmd)
        print("commond sent to pi: ", cmd)
        # server.send_msg(cmd.encode(encoding="utf-8"))
        objects_info_dict[ObjInfoKey(image_array)] = objects_info
        print("* " * 50)
        cv2.waitKey(1)
    end = time.time()
    local_path = object_dict_to_csv(objects_info_dict, folder_name)
    print("local_path: ", local_path)
    print(end - start)
    Uploader("server_conf.conf", local_path, Constant.SERVER_DATA_PATH).upload()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_drive()
