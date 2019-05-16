# -*- coding: utf-8 -*-
# @author: mdmbct
# @date: 3/5/2019 10:43 PM
# @last modified by:
# @last modified time: 4/18/2019 6:59 PM

import paramiko
import zipfile
import os
import configparser
from util import Constant


class Uploader(object):

    def __init__(self, conf_path, local_path, server_path):
        self.conf_path = conf_path
        self.local_path = local_path
        self.server_path = server_path
        self.host = None
        self.port = None
        self.user = None
        self.pwd = None
        self.read_conf()

    def read_conf(self):
        cf = configparser.ConfigParser()
        cf.read(self.conf_path, encoding='utf-8')
        self.host = cf.get("server", "host")
        self.port = eval(cf.get("server", "port"))
        self.user = cf.get("server", "user")
        self.pwd = cf.get("server", "pwd")

    def compress_to_zip(self):
        dir = os.path.dirname(__file__) + '/' + self.local_path
        compressed_file = os.path.join(Constant.DATA_SET_PATH, "zip",
                                       self.local_path.split('/')[-1] + ".zip")  # 压缩后文件夹的名字
        print("compressed_file: ", compressed_file)
        z = zipfile.ZipFile(compressed_file, 'w', zipfile.ZIP_DEFLATED)  # 参数一：文件夹名
        for dirpath, dirnames, filenames in os.walk(dir):
            fpath = dirpath.replace(dir, '')  # 这一句很重要，不replace的话，就从根目录开始复制
            fpath = fpath and fpath + os.sep or ''  # 这句话理解我也点郁闷，实现当前文件夹以及包含的所有文件的压缩
            for filename in filenames:
                z.write(os.path.join(dirpath, filename), fpath + filename)
                # print('压缩成功')
        z.close()
        return compressed_file

    def upload(self):
        print("local_path: ", self.local_path)
        print("server_path: ", self.server_path)
        try:
            terminal = paramiko.Transport(self.host, self.port)
            terminal.connect(username=self.user, password=self.pwd)
            sftp = paramiko.SFTPClient.from_transport(terminal)
            print("server_file: ", os.path.join(self.server_path, self.local_path.split('/')[-1] + ".zip"))
            sftp.put(self.compress_to_zip(), os.path.join(self.server_path, self.local_path.split('/')[-1] + ".zip"))
            terminal.close()
        except Exception as e:
            print(e)


if __name__ == '__main__':
    uploader = Uploader("server_conf.conf", os.path.join(Constant.DATA_SET_PATH, "set_BetterMeLenovo_20190509195309"),
                        Constant.SERVER_DATA_PATH)
    uploader.upload()
    # /root/self_driving_data_set/static/source/data_set
