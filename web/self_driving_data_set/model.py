# -*- coding: utf-8 -*-
# @author: mdmbct
# @date:   5/2/2019 3:09 PM
# @last modified by: 
# @last modified time: 5/2/2019 3:09 PM
import glob
class DataSet(object):
    def __init__(self, uploader, upload_date, file_name):
        self.uploader = uploader
        self.upload_date = upload_date
        self.file_name = file_name



class TrainModel(object):
    def __init__(self, uploader, upload_date, desc, file_name):
        self.uploader = uploader
        self.upload_date = upload_date
        self.desc = desc
        self.file_name = file_name


def get_all_data_sets(path):
    print("path", path)
    set_paths = glob.glob(path + "*.7z") + glob.glob(path + "*.zip")
    print(set_paths)
    sets = []
    for path in set_paths:
        file_name = path.split('\\')[-1]
        info = file_name.split('_')
        sets.append(DataSet(info[1], info[2].split('.')[0], file_name))
    return sets

def get_all_model(path):
    model_paths = glob.glob(path + "*.7z") + glob.glob(path + "*.zip")
    train_models = []
    for path in model_paths:
        file_name = path.split('\\')[-1]
        info = file_name.split('_')
        train_models.append(TrainModel(info[1], info[2], info[3].split('.')[0], file_name))
    return train_models


if __name__ == '__main__':
    data_sets = get_all_data_sets("../source/data_set/")
    train_model = get_all_model("../source/model/")

    for set in data_sets:
        print(set.href)