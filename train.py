# 搭建深度学习模型
# 导入库
# 自动驾驶模型真实道路模拟行驶
import keras
import tensorflow
import sys
import os
import h5py
import random
import numpy as np
import glob
import math
import datetime
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import load_model, Model, Input
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from util import Constant

np.random.seed(0)


# os.environ["CUDA_VISIBLE_DEVICES"] = '1'


# step1,载入数据，并且分割为训练和验证集
# 问题，数据集太大了，已经超过计算机内存
def load_data(npz_path, trained_num=0, random_get=True):
    """
    载入npz文件
    :param npz_path: npz文件的路径
    :param trained_num: 希望载入的数量，不填的话就是npz_path下的所有文件
    :param random_get: 是否随机载入npz_path下的文件，默认是随机的
    :return:
    """
    # load
    image_array = np.zeros((1, Constant.IMG_HEIGHT, Constant.IMG_WIDTH, Constant.IMG_CHANNELS))  # 初始化
    label_array = np.zeros((1, 5), 'float')
    training_data = glob.glob(npz_path + '*.npz')
    npz_mum = len(training_data)
    print("There has %d npz files in the %s" % (npz_mum, npz_path))
    if not training_data:
        print("No training data in directory, exit")
        sys.exit()
    # print(training_data)
    if trained_num == 0 | trained_num > npz_mum:
        trained_num = npz_mum
        print("Will load %d npz files" % trained_num)
    else:
        print("Will load %d npz files" % trained_num)
    numbers = []  # to save the number of npz loaded in training_data
    if random_get:
        while len(numbers) < trained_num:
            number = random.randint(0, trained_num - 1)
            if number not in numbers:
                numbers.append(number)
    else:
        numbers = [i for i in range(trained_num)]
    print("These npz file will be loaded")
    print(numbers)
    i = 0
    for number in numbers:
        with np.load(training_data[number]) as data:
            # print(dir(data))
            print(data.keys())
            i = i + 1
            train_temp = data['train_imgs']  # img to npz加的
            train_labels_temp = data['train_labels']
        image_array = np.vstack((image_array, train_temp))  # 把文件读取都放入，内存
        print("image_array size %d" % (image_array.size))
        label_array = np.vstack((label_array, train_labels_temp))
        print("image_label size %d" % (label_array.size))
        print("loaded the %d npz files, finished %.3f%%" % (number, (i / trained_num) * 100))
    print("all loaded...")
    X = image_array[1:, :]
    y = label_array[1:, :]
    print('Image array shape: ' + str(X.shape))
    print('Label array shape: ' + str(y.shape))
    print(np.mean(X))
    print(np.var(X))

    # now we can split the data into a training (80), testing(20), and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_valid, y_train, y_valid


def batch_generate_arrays_from_file(file, file_batch_size):
    """
    从文件中分批载入训练数据,防止内存不够
    :param file: 训练数据
    :param file_batch_size: 一次载入文件的数量，和训练时的batch_size有所区别
    :return:
    """
    train_data = glob.glob(file)
    file_num = len(train_data)
    print("Loading the training data...")
    print("There has %d training data files." % (file_num))
    # if no data, exit，容错判断
    if not train_data:
        print("No training data in directory, exit")
        sys.exit()
    chunks = int(math.ceil(file_num / file_batch_size))
    print("Batch read the files, has %d turns." % (chunks))
    for chunk in range(0, chunks):
        image_array = np.zeros((1, Constant.IMG_HEIGHT, Constant.IMG_WIDTH, Constant.IMG_CHANNELS))  # 初始化
        label_array = np.zeros((1, 5), 'float32')
        current_turn_train_data = train_data[chunk * file_batch_size: (chunk + 1) * file_batch_size]
        i = 0;
        for npz in current_turn_train_data:
            with np.load(npz) as data:
                i += 1;
                print("The num of loaded file is %d" % (i))
                image_temp = data["train_imgs"]
                labels_temp = data['train_labels']
            image_array = np.vstack((image_array, image_temp))
            print("image_arry size: %d" % (image_array.size))
            label_array = np.vstack((label_array, labels_temp))
            print("image_label size: %d" % (label_array.size))
        print("The %d round train data has alresdy loaded." % (chunk))
        X = image_array[1:, :]
        y = label_array[1:, :]
        print('Image array shape: ' + str(X.shape))
        print('Label array shape: ' + str(y.shape))
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
        yield X_train, X_valid, y_train, y_valid
        image_array = []
        label_array = []
    print("Loaded all the train data.")


def image_laber_handler(file_name):
    """
    根据图片文件名 ，返回这张图片的label（即车子此时的方向）的array
    :param file_name:
    :return:
    """
    label = int(file_name[0])
    if label == 2:  # forward
        label_array = [0., 0., 1., 0., 0.]
    elif label == 3:  # back
        label_array = [0., 0., 0., 1., 0.]
    elif label == 0:  # left
        label_array = [1., 0., 0., 0., 0.]
    elif label == 1:  # right
        label_array = [0., 1., 0., 0., 0.]
    elif label == 4:  # stop
        label_array = [0., 0., 0., 0., 1.]
    return label_array


def images_array_handler(images_array):
    train_images = np.zeros([1, Constant.IMG_HEIGHT, Constant.IMG_WIDTH, Constant.IMG_CHANNELS], "float")
    for image_array in images_array:
        image_array = np.expand_dims(image_array, axis=0)  # 增加一个维度
        train_images = np.vstack((train_images, image_array))
    return train_images
    # 返回图片的数据（矩阵），和对应的标签值


# def draw_training_process_curve(history):
#     fig = plt.figure()  # 新建一张图
#     plt.plot(history.history['acc'], label='training acc')
#     plt.plot(history.history['val_acc'], label='val acc')
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(loc='lower right')
#     fig.savefig('VGG16' + str(model_id) + 'acc.png')
#     fig = plt.figure()
#     plt.plot(history.history['loss'], label='training loss')
#     plt.plot(history.history['val_loss'], label='val loss')
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(loc='upper right')
#     fig.savefig('VGG16' + str(model_id) + 'loss.png')
#
# def save_traing_result(history, model_id):
#
#     logFilePath = "log_" + Constant.TRAIN_LOG_PATH + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '.txt'
#     fobj = open(logFilePath, 'a')
#     fobj.write('x_train shape: ' + str(X_train.shape) + '\n')
#     fobj.write('x_test shape: ' + str(X_test.shape) + '\n')
#     fobj.write('-' * 50 + '\n')
#     fobj.write('model id: ' + str(model_id) + '\n')
#     fobj.write('epoch: ' + str(epochs) + '\n')
#     fobj.write('training accuracy: ' + str(history.history['acc'][-1]) + '\n')
#     fobj.write('model evaluation results: ' + str(score[0]) + '  ' + str(score[-1]) + '\n')
#     fobj.write('-' * 50 + '\n')
#     fobj.write('\n')
#     fobj.close()

# step2 建立模型
def build_model(keep_prob):
    """
    建立模型
    :param keep_prob: the rate of dropout
    :return:
    """
    print("开始编译模型")
    model = Sequential()
    model.add(
        Lambda(lambda x: (x / 127.5 - 1.0), input_shape=(Constant.IMG_HEIGHT, Constant.IMG_WIDTH, Constant.IMG_CHANNELS)))
    model.add(Conv2D(filters=24, kernel_size=(5, 5), activation='elu', strides=(2, 2),
                     input_shape=(Constant.IMG_HEIGHT, Constant.IMG_WIDTH, Constant.IMG_CHANNELS)))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(keep_prob))  # Dropout将在训练过程中每次更新参数时随机断开一定百分比（p）的输入神经元连接
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(5))
    model.summary()

    return model


# step3 训练模型
def train_model(model, learning_rate, nb_epoch, samples_per_epoch,
                batch_size, X_train, X_valid, y_train, y_valid):
    """
    训练传进来的model
    :param model:
    :param learning_rate:
    :param nb_epoch:
    :param samples_per_epoch:
    :param batch_size:
    :param X_train:
    :param X_valid:
    :param y_train:
    :param y_valid:
    :return:
    """
    # 值保存最好的模型存下来
    model_path = Constant.MODEL_PATH + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + "/"
    os.makedirs(model_path)
    checkpoint = ModelCheckpoint(filepath=model_path + 'model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min')
    # EarlyStopping patience：当earlystop被激活（如发现loss相比上一个epoch训练没有下降），
    # 则经过patience个epoch后停止训练。
    # mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。
    early_stop = EarlyStopping(monitor='loss', min_delta=.0005, patience=10,
                               verbose=1, mode='min')
    tensorboard = TensorBoard(log_dir=Constant.TRAIN_LOG_PATH + "tensorboard/", histogram_freq=0, batch_size=20,
                              write_graph=True, write_grads=True,
                              write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                              embeddings_metadata=None)
    # 编译神经网络模型，loss损失函数，optimizer优化器， metrics列表，包含评估模型在训练和测试时网络性能的指标
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    # 训练神经网络模型，batch_size梯度下降时每个batch包含的样本数，epochs训练多少轮结束，
    # verbose是否显示日志信息，validation_data用来验证的数据集
    history = model.fit_generator(batch_generator(X_train, y_train, batch_size),
                                  steps_per_epoch=samples_per_epoch / batch_size,
                                  epochs=nb_epoch,
                                  max_queue_size=1,
                                  validation_data=batch_generator(X_valid, y_valid, batch_size),
                                  validation_steps=len(X_valid) / batch_size,
                                  callbacks=[tensorboard, checkpoint, early_stop],
                                  verbose=2)
    # draw_training_process_curve(history)
    # print(checkpoint.model)
    # save_traing_result(history, checkpoint.model, )


# step4
# 可以一个batch一个batch进行训练，CPU和GPU同时开工
def batch_generator(X, y, batch_size):
    images = np.empty([batch_size, Constant.IMG_HEIGHT, Constant.IMG_WIDTH, Constant.IMG_CHANNELS])
    steers = np.empty([batch_size, 5])
    print("shape[0]: %d" % (X.shape[0]))
    while True:
        i = 0
        for index in np.random.permutation(X.shape[0]):
            images[i] = X[index]
            steers[i] = y[index]
            i += 1
            if i == batch_size:
                break
        yield (images, steers)


# step5 评估模型
# def evaluate(x_test, y_test):
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


def main():
    # 打印出超参数

    print('-' * 30)
    print('parameters')
    print('-' * 30)

    keep_prob = 0.5
    learning_rate = 0.0001
    nb_epoch = 20
    samples_per_epoch = 3000
    batch_size = 90
    # steps_per_epoch=samples_per_epoch / batch_size

    print('keep_prob = ', keep_prob)
    print('learning_rate = ', learning_rate)
    print('nb_epoch = ', nb_epoch)
    print('samples_per_epoch = ', samples_per_epoch)
    print('batch_size = ', batch_size)
    print('-' * 30)
    # 开始载入数据
    datas = load_data(Constant.NPZ_PATH, 150, random_get=True)
    print("数据加载完毕")
    # 编译模型
    model = build_model(keep_prob)
    train_model(model, learning_rate, nb_epoch, samples_per_epoch, batch_size, *datas)
    print("模型训练完毕")


if __name__ == '__main__':
    main()
