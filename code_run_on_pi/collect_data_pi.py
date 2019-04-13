# -*- coding: utf-8 -*-
# @author: mdmbct
# @date:   3/20/2019 19:24 PM
# @last modified by:
# @last modified time:
# 收集数据，赛道照片和对应的前、后、左、右、4停
import io
import socket
import struct
import car_control
import os
import sys
os.environ['SDL_VIDEODRIVE'] = 'x11'
from time import ctime, sleep, time
import threading
import picamera
import picamera.array
import datetime
global is_capture_running, cmd
IMG_WIDTH, IMG_HEIGHT = 160, 120
global direction

class Client(object):

    def __init__(self, host, port):
        # create socket and bind host
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))
        self.connection = self.client_socket.makefile('rwb')

class SplitFrames(object):

    def __init__(self, client):
        self.frame_num = 0
        # self.output = None
        self.connection = client.connection
        self.socket = client.client_socket
        self.stream = io.BytesIO()
        # self.path = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "/"
        # os.mkdir(self.path)

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # Start of new frame; send the old one's length
            # then the data
            size = self.stream.tell()
            if size > 0:
                self.connection.write(struct.pack('<L', size))
                self.connection.flush()
                self.stream.seek(0)
                self.connection.write(self.stream.read(size))
                # if self.output:
                #     self.output.close()
                # self.stream = io.open(self.path + '%s_image%s.jpg' % (direction, time()), 'wb')  # 改变格式为jpg
                self.frame_num += 1
                self.stream.seek(0)
        # self.output.write(buf)
        self.stream.write(buf)

def pi_capture(client):
    # init the train_label array
    print("Start capture")
    try:
        with picamera.PiCamera(resolution=(IMG_WIDTH, IMG_HEIGHT), framerate=30) as camera:
            # 根据摄像头实际情况判断是否要加这句上下翻转
            # camera.vflip = True
            # Give the camera some warm-up time
            sleep(2)
            output = SplitFrames(client)
            start = time()
            camera.start_recording(output, format='mjpeg')
            camera.wait_recording(120)
            camera.stop_recording()
            client.connection.write(struct.pack('<L', 0))
    finally:
        finish = time()
        global is_capture_running
        is_capture_running = False
        print('Captured %d frames at %.2ffps' % (
            output.frame_num,
            output.frame_num / (finish - start)))
        print("quit pi capture")
        client.connection.close()
        client.client_socket.close()
        sys.exit(0)
        # main()


def my_car_control():
    global is_capture_running, cmd
    is_capture_running = True
    car_control.car_stop()
    sleep(0.1)
    print("Start control!")
    while is_capture_running:
        car_control.control_by_cmd(cmd, sleep_time=0.1)
        if cmd == 'q':
            is_capture_running = False

def receive_cmd(client):
    global cmd, is_capture_running, direction
    while is_capture_running:
        cmd = str(client.client_socket.recv(1024), encoding='utf-8')
        if cmd != 'q':
            direction = cmd
        else:
            direction = '4'
        # print(cmd)


if __name__ == '__main__':
    h = "169.254.178.216"
    p = 8000
    c = Client(h, p)
    global cmd, is_capture_running
    cmd = 's'

    is_capture_running = True
    print("capture thread")
    print('-' * 50)


    capture_thread = threading.Thread(target=pi_capture, args=(c,))
    capture_thread.setDaemon(True)
    # print(capture_thread.name)
    capture_thread.start()
    receive_cmd_thread = threading.Thread(target=receive_cmd, args=(c,))
    receive_cmd_thread.setDaemon(True)
    receive_cmd_thread.start()
    my_car_control()
    while is_capture_running:
        pass
    # when the main thread ended, stop the car, clean GPIO, and child thread also have to end
    car_control.car_stop()
    car_control.clean_GPIO()
    print("all have done!")
