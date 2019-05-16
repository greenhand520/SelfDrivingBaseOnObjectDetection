# -*- coding: utf-8 -*-
# @author: mdmbct
# @date:   5/4/2019 9:28 AM
# @last modified by: 
# @last modified time: 5/4/2019 9:28 AM
import cv2
import socket
import numpy as np
from pythonds.basic.stack import Stack

HOST = "0.0.0.0"
PORT = 8000


class Server(object):

    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((HOST, PORT))
        self.server_socket.listen(0)
        self.connector, self.client_address = self.server_socket.accept()
        self.connection = self.connector.makefile('rwb')
        self.host_name = socket.gethostname()
        self.host_ip = socket.gethostbyname(self.host_name)
        self.is_received = True
        self.image_stack = Stack()
        self.image_list = []
        self.DIRE_LEFT = 0
        self.DIRE_RIGHT = 1
        self.DIRE_FORWARD = 2
        self.DIRE_BACK = 3
        self.DIRE_STOP = 4

    def send_msg(self, msg):
        self.connector.send(msg.encode(encoding="utf-8"))
        # self.connection.write(info)

    def get_video_stream(self):
        try:
            print("Getting stream from pi...")
            print("Press 'q' to exit")
            # need bytes here
            stream_bytes = b' '
            while self.is_received:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if not self.image_stack.isEmpty():
                        self.image_stack.pop()  # if the stack is not empty, pop a image, the push a image to it.
                    self.image_stack.push(image)
                    # if len(self.image_list) != 0:
                    #     self.image_list[0] = image
                    # else:
                    #     self.image_list.append(image)
        except Exception as e:
            print(e)
        finally:
            self.close_server()
            cv2.destroyAllWindows()
            self.is_received = False

    def receive_info(self, stream_bytes):
        # a useless function, only do not want remove it.
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
        # only used in old version used CNN to drive car, new version used object detection does not use it.
        # do not want remove it.
        if direction == self.DIRE_LEFT:  # left
            self.send_msg('0')
        elif direction == self.DIRE_RIGHT:  # right
            self.send_msg('1')
        elif direction == self.DIRE_BACK:  # back
            self.send_msg('3')
        elif direction == self.DIRE_STOP:  # stop
            self.send_msg('4')
        else:  # front
            self.send_msg('2')
