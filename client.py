# -*- coding: utf-8 -*-
# @author: mdmbct
# @date:   5/4/2019 9:29 AM
# @last modified by: 
# @last modified time: 5/4/2019 9:29 AM
import socket
import time

SERVER_IP = "118.25.142.54"
SERVER_PORT = 80


class Client(object):
    def __init__(self):
        self.TL_RED = 0
        self.TL_YELLOW = 1
        self.TL_GREEN = 2
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((SERVER_IP, SERVER_PORT))
        self.tl_state = self.TL_RED

    def send_msg(self, msg):
        self.client_socket.send(msg.encode(encoding="utf-8"))

    def rec_msg(self):
        return self.client_socket.recv(1024).decode('utf-8')


if __name__ == '__main__':
    client = Client()
    client.send_msg("tl_state")
    time.sleep(3)
    msg = client.rec_msg()
    if msg != '':
        print(msg)
    # time1, time2, time3, time4 = 0, 0, 0, 0
    # g, y, r = False, False, False
    # time1 = time.time()
    # i = 0
    # while i <= 10000:
    #     msg = client.rec_msg()
    #     print(msg)
    #     g = True
    #     if msg == '1' and y is False:
    #         time2 = time.time()
    #         y = True
    #     elif msg == '0' and r is False:
    #         time3 = time.time()
    #         r = True
    #     i += 1
    # time4 = time.time()
    # print(time2 - time1)
    # print(time3 - time2)
    # print(time4 - time3)
