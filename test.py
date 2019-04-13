import os
import datetime
from utils import Constant
model_path = Constant.MODEL_PATH + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + "/"
os.makedirs(model_path)

# os.chdir("model/")
# os.mkdir(path)
print(len("G:/self_drive/bgr_data\\4"))