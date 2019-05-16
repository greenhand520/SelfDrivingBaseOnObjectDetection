# -*- coding: utf-8 -*-
# @author: mdmbct
# @date:   5/16/2019 10:12 PM
# @last modified by: 
# @last modified time: 5/16/2019 10:12 PM

# -*- coding: utf-8 -*-
# @author: mdmbct
# @date:   4/30/2019 10:34 PM
# @last modified by:
# @last modified time: 4/30/2019 10:34 PM

import socket, threading, time  # 导入socket库
import traffic_light_server

# 创建一个socket对象，AF_INET指定使用IPv4协议(AF_INET6代表IPV6)，SOCK_STREAM指定使用面向流的TCP协议
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 监听端口，127.0.0.1是本机地址，客户端必须在本机才能与其连接。端口大于1024
# s.bind(('127.0.0.1',8888))  #本机测试
s.bind(('0.0.0.0', 80))  # 服务端代码IP用内网IP
s.listen(5)  # 开始监听端口，数字表示等待连接的最大数量
print('waiting for connection')
LIGHT_RED = 0
LIGHT_YELLOW = 1
LIGHT_GREEN = 2

def tcplink(sock, addr):
    print('accept new connection from %s：%s' % addr)  # 注意这里addr是一个tuple所以有两个%s
    # sock.send(b'welcome')  # 向客户端返回welcome消息
    while True:
        # data = sock.recv(1024)  # 从客户端接受消息，最多1024字节
        # # time.sleep(2)
        # if not data or data.decode('utf-8') == 'exit':
        #     break
        # else:
        #     # print(data.decode('utf-8'))
        #     print(data)
        if sock.recv(1024).decode('utf-8') == "tl_state":
            print("traffic light state: ", ("RED", "YELLOW", "GREEN")[tl_state])
            sock.send(str(tl_state).encode('utf-8'))
        # sock.send(('hello,%s' % data.decode('utf-8')).encode('utf-8'))
    # sleep(10)    #向客户端返回加了hello的消息
    sock.close()    # 关闭
    print('connection from %s:%s closed' % addr)

def set_traffic_light_state():
    global tl_state
    while True:
        tl_state = LIGHT_GREEN
        time.sleep(10)
        tl_state = LIGHT_YELLOW
        time.sleep(3)
        tl_state = LIGHT_RED
        time.sleep(5)

while True:  # 服务器程序通过一个永久循环来接受来自多个客户端的连接
    import threading
    tl_thread = threading.Thread(target=set_traffic_light_state)
    tl_thread.start()
    sock, addr = s.accept()  # 接受一个新连接，用于接收和发送数据。addr是连接的客户端的地址
    t = threading.Thread(target=tcplink, args=(sock, addr))  # 创建一个新线程来处理TCP连接（这个很关键）
    t.start()
