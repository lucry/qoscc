# -*- coding: UTF-8 -*-
"""
@author: hichenway
@知乎: 海晨威
@contact: lyshello123@163.com
@time: 2020/5/9 17:00
@license: Apache
主程序：包括配置，数据读取，日志记录，绘图，模型训练和预测
"""

# import thread
# import csv
import tensorflow as tf
import queue
import pandas as pd
from model.model_tensorflow import predict_add
import numpy as np
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

frame = "tensorflow"  # 可选： "keras", "pytorch", "tensorflow"
if frame == "pytorch":
    from model.model_pytorch import train, predict
elif frame == "keras":
    from model.model_keras import train, predict

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
elif frame == "tensorflow":
    from model.model_tensorflow import train, predict, train_add, predict_add

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示error， tf和keras下会有很多tf的warning，但不影响训练
else:
    raise Exception("Wrong frame seletion")

counter = False


class Config:
    # 数据参数
    # feature_columns = list(range(2, 9))     # 要作为feature的列，按原数据从0开始计算，也可以用list 如 [2, 3, 4, 5, 6, 7, 8] 设置
    # label_columns = [4, 5]                  # 要预测的列，按原数据从0开始计算, 如同时预测第四，五列 最低价和最高价

    feature_columns = list(range(0, 2))
    label_columns = [0, 1]
    # feature_columns = list(range(0, 3))
    # label_columns = [0, 1, 2]

    # label_in_feature_index = [feature_columns.index(i) for i in label_columns]  # 这样写不行
    label_in_feature_index = (lambda x, y: [x.index(i) for i in y])(feature_columns, label_columns)  # 因为feature不一定从0开始
    # 上一句代码解析：
    # x.index(i) 方法检测字符串x中是否包含子字符串i
    # feature_columns = 'qwertyuiop'
    # label_columns = ['q', 'p']
    # label_in_feature_index = (lambda x, y: [x.index(i) for i in y])(feature_columns, label_columns)
    # print(label_in_feature_index)
    # # 输出结果：
    # # [0, 9]

    predict_day = 1  # 预测未来几天

    # 网络参数
    input_size = len(feature_columns)
    output_size = len(label_columns)

    hidden_size = 128  # LSTM的隐藏层大小，也是输出大小
    lstm_layers = 2  # LSTM的堆叠层数
    dropout_rate = 0.2  # dropout概率
    time_step = 20  # 原来是20，这个参数很重要，是设置用前多少天的数据来预测，也是LSTM的time step数，请保证训练数据量大于它

    # 训练参数
    do_train = True
    do_predict = True
    add_train = True  # 是否载入已有模型参数进行增量训练
    shuffle_train_data = True  # 是否对训练数据做shuffle
    use_cuda = False  # 是否使用GPU训练

    train_data_rate = 0.85  # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    valid_data_rate = 0.15  # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

    # batch_size = 64->32
    batch_size = 32  # 一次训练所抓取的数据样本数量
    learning_rate = 0.001
    epoch = 8  # 原来为20->80，整个训练集被训练多少遍，不考虑早停的前提下
    patience = 50  # 训练多少epoch，验证集没提升就停掉，早停参数
    random_seed = 42  # 随机种子，保证可复现

    do_continue_train = False  # 每次训练把上一次的final_state作为下一次的init_state，仅用于RNN类型模型，目前仅支持pytorch
    continue_flag = ""  # 但实际效果不佳，可能原因：仅能以 batch_size = 1 训练
    if do_continue_train:
        shuffle_train_data = False
        batch_size = 1
        continue_flag = "continue_"

    # 训练模式
    # debug_mode = True
    debug_mode = False  # 调试模式下，是为了跑通代码，追求快
    debug_num = 500  # 仅用debug_num条数据来调试

    # 框架参数
    used_frame = frame  # 选择的深度学习框架，不同的框架模型保存后缀不一样
    model_postfix = {"pytorch": ".pth", "keras": ".h5", "tensorflow": ".ckpt"}
    model_name = "model_" + continue_flag + used_frame + model_postfix[used_frame]

    # 路径参数
    train_data_path = "./data/lb-tput-rtt-notime.csv"
    train_data_path_add_train = "./data/mobb-tput-rtt-notime.csv"
    # train_data_path = "./data/stock_data.csv"
    model_save_path = "./checkpoint/" + used_frame + "/"
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    do_log_print_to_screen = True
    do_log_save_to_file = True  # 是否将config和训练过程记录到log
    do_figure_save = True
    do_train_visualized = True  # 训练loss可视化，pytorch用visdom，tf用tensorboardX，实际上可以通用, keras没有
    if not os.path.exists(model_save_path):  # 判断括号里的文件是否存在的意思，括号内的可以是文件路径
        os.makedirs(model_save_path)  # makedirs 递归创建目录
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if do_train and (do_log_save_to_file or do_train_visualized):
        ## time.strftime(format[, t])接收以时间元组，并返回以可读字符串表示的当地时间，format -- 格式字符串：%Y 四位数的年份表示（000-9999），%m 月份（01-12），%d 月内中的一天（0-31），%H 24小时制小时数（0-23），%M 分钟数（00=59），%S 秒（00-59）
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + '_' + used_frame + "/"
        os.makedirs(log_save_path)


class Data:

    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name = self.read_data()

        self.data_num = self.data.shape[0]  # shape[0]输出矩阵的行数，shape[1]输出列数 【数字】
        self.train_num = int(self.data_num * self.config.train_data_rate)  # 矩阵的行数*训练数据率【数字】

        self.mean = np.mean(self.data, axis=0)  # 数据的均值，axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
        self.std = np.std(self.data, axis=0)  # 数据的方差
        self.norm_data = (self.data - self.mean) / self.std  # 归一化，去量纲【矩阵7*n】

        self.start_num_in_test = 0  # 测试集中前几天的数据会被删掉，因为它不够一个time_step

    def read_data(self):  # 读取初始数据
        if self.config.debug_mode:
            init_data = pd.read_csv(self.config.train_data, nrows=self.config.debug_num,
                                    usecols=self.config.feature_columns)
        else:
            init_data = pd.read_csv(self.config.train_data, usecols=self.config.feature_columns)  # usecols读取指定的列
        return init_data.values, init_data.columns.tolist()  # .columns.tolist() 是获取列名

    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:self.train_num]  # 取norm_data的前train_num部分数据
        label_data = self.norm_data[self.config.predict_day: self.config.predict_day + self.train_num,
                     self.config.label_in_feature_index]  # 将延后几天的数据作为label,取预测天数到训练数据+预测天数中的所有label_in_feature_index维数据
        # x[:,n]表示在全部数组（维）中取第n个数据，直观来说，x[:,n]就是取所有集合的第n个数据
        # print(feature_data)
        # print(label_data)

        if not self.config.do_continue_train:
            # 在非连续训练模式下，每time_step行数据会作为一个样本，两个样本错开一行，比如：1-20行，2-21行...train_num-time_step至train_num行
            train_x = [feature_data[i:i + self.config.time_step] for i in range(self.train_num - self.config.time_step)]
            train_y = [label_data[i:i + self.config.time_step] for i in range(self.train_num - self.config.time_step)]
        else:
            # 在连续训练模式下，每time_step行数据会作为一个样本，两个样本错开time_step行，
            # 比如：1-20行，21-40行。。。到数据末尾，然后又是 2-21行，22-41行。。。到数据末尾，……
            # 这样才可以把上一个样本的final_state作为下一个样本的init_state，而且不能shuffle
            # 目前本项目中仅能在pytorch的RNN系列模型中用
            train_x = [
                feature_data[start_index + i * self.config.time_step: start_index + (i + 1) * self.config.time_step]
                for start_index in range(self.config.time_step)
                for i in range((self.train_num - start_index) // self.config.time_step)]
            train_y = [
                label_data[start_index + i * self.config.time_step: start_index + (i + 1) * self.config.time_step]
                for start_index in range(self.config.time_step)
                for i in range((self.train_num - start_index) // self.config.time_step)]

        train_x, train_y = np.array(train_x), np.array(train_y)  # Numpy.array()用来产生数组

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)  # 划分训练和验证集，并打乱
        return train_x, valid_x, train_y, valid_y

    def get_test_data(self, return_label_data=False):
        feature_data = self.norm_data[self.train_num:]  # 取测试集数据
        sample_interval = min(feature_data.shape[0], self.config.time_step)  # 防止time_step大于测试集数量
        self.start_num_in_test = feature_data.shape[0] % sample_interval  # 这些天的数据不够一个sample_interval【数字】
        time_step_size = feature_data.shape[0] // sample_interval  # 整除 【数字】

        # 在测试数据中，每time_step行数据会作为一个样本，两个样本错开time_step行
        # 比如：1-20行，21-40行。。。到数据末尾。
        test_x = [feature_data[
                  self.start_num_in_test + i * sample_interval: self.start_num_in_test + (i + 1) * sample_interval]
                  for i in range(time_step_size)]
        if return_label_data:  # 实际应用中的测试集是没有label数据的
            label_data = self.norm_data[self.train_num + self.start_num_in_test:, self.config.label_in_feature_index]
            return np.array(test_x), label_data
        return np.array(test_x)


class Data_add_train:

    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name = self.read_data()

        self.data_num = self.data.shape[0]  # shape[0]输出矩阵的行数，shape[1]输出列数 【数字】
        self.train_num = int(self.data_num * self.config.train_data_rate)  # 矩阵的行数*训练数据率【数字】

        self.mean = np.mean(self.data, axis=0)  # 数据的均值，axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
        self.std = np.std(self.data, axis=0)  # 数据的方差
        self.norm_data = (self.data - self.mean) / self.std  # 归一化，去量纲【矩阵7*n】

        self.start_num_in_test = 0  # 测试集中前几天的数据会被删掉，因为它不够一个time_step

    def read_data(self):  # 读取初始数据
        if self.config.debug_mode:
            init_data = pd.read_csv(self.config.train_data_path_add_train, nrows=self.config.debug_num,
                                    usecols=self.config.feature_columns)
        else:
            init_data = pd.read_csv(self.config.train_data, usecols=self.config.feature_columns)  # usecols读取指定的列
            # data_csv = pd.read_csv(r"./data/mobb-tput-rtt-notime.csv") # 读取新增列后的csv文件
            # print("新增后csv文件数据为：")
            # print(data_csv)
        return init_data.values, init_data.columns.tolist()  # .columns.tolist() 是获取列名

    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:self.train_num]  # 取norm_data的前train_num部分数据
        label_data = self.norm_data[self.config.predict_day: self.config.predict_day + self.train_num,
                     self.config.label_in_feature_index]  # 将延后几天的数据作为label,取预测天数到训练数据+预测天数中的所有label_in_feature_index维数据
        # x[:,n]表示在全部数组（维）中取第n个数据，直观来说，x[:,n]就是取所有集合的第n个数据
        # print(feature_data)
        # print(label_data)

        if not self.config.do_continue_train:
            # 在非连续训练模式下，每time_step行数据会作为一个样本，两个样本错开一行，比如：1-20行，2-21行...train_num-time_step至train_num行
            train_x = [feature_data[i:i + self.config.time_step] for i in range(self.train_num - self.config.time_step)]
            train_y = [label_data[i:i + self.config.time_step] for i in range(self.train_num - self.config.time_step)]
        else:
            # 在连续训练模式下，每time_step行数据会作为一个样本，两个样本错开time_step行，
            # 比如：1-20行，21-40行。。。到数据末尾，然后又是 2-21行，22-41行。。。到数据末尾，……
            # 这样才可以把上一个样本的final_state作为下一个样本的init_state，而且不能shuffle
            # 目前本项目中仅能在pytorch的RNN系列模型中用
            train_x = [
                feature_data[start_index + i * self.config.time_step: start_index + (i + 1) * self.config.time_step]
                for start_index in range(self.config.time_step)
                for i in range((self.train_num - start_index) // self.config.time_step)]
            train_y = [
                label_data[start_index + i * self.config.time_step: start_index + (i + 1) * self.config.time_step]
                for start_index in range(self.config.time_step)
                for i in range((self.train_num - start_index) // self.config.time_step)]

        train_x, train_y = np.array(train_x), np.array(train_y)  # Numpy.array()用来产生数组

        # train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
        #                                                       random_state=self.config.random_seed,
        #                                                       shuffle=self.config.shuffle_train_data)   # 划分训练和验证集，并打乱
        # return train_x, valid_x, train_y, valid_y
        return train_x, train_y

    def get_test_data(self, return_label_data=False):
        feature_data = self.norm_data[self.train_num:]  # 取测试集数据
        sample_interval = min(feature_data.shape[0], self.config.time_step)  # 防止time_step大于测试集数量
        self.start_num_in_test = feature_data.shape[0] % sample_interval  # 这些天的数据不够一个sample_interval【数字】
        time_step_size = feature_data.shape[0] // sample_interval  # 整除 【数字】

        # 在测试数据中，每time_step行数据会作为一个样本，两个样本错开time_step行
        # 比如：1-20行，21-40行。。。到数据末尾。
        test_x = [feature_data[
                  self.start_num_in_test + i * sample_interval: self.start_num_in_test + (i + 1) * sample_interval]
                  for i in range(time_step_size)]
        if return_label_data:  # 实际应用中的测试集是没有label数据的
            label_data = self.norm_data[self.train_num + self.start_num_in_test:, self.config.label_in_feature_index]
            return np.array(test_x), label_data
        return np.array(test_x)


def load_logger(config):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    if config.do_log_print_to_screen:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                      fmt='[ %(asctime)s ] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # FileHandler
    if config.do_log_save_to_file:
        file_handler = RotatingFileHandler(config.log_save_path + "out.log", maxBytes=1024000, backupCount=5)
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 把config信息也记录到log 文件中
        config_dict = {}
        for key in dir(config):
            if not key.startswith("_"):
                config_dict[key] = getattr(config, key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)

    return logger


def draw(config: Config, origin_data: Data, logger, predict_norm_data: np.ndarray):
    label_data = origin_data.data[origin_data.train_num + origin_data.start_num_in_test:,
                 config.label_in_feature_index]  # x[n,:]表示在n个数组（维）中取全部数据，直观来说，x[n,:]就是取第n集合的所有数据
    predict_data = predict_norm_data * origin_data.std[config.label_in_feature_index] + \
                   origin_data.mean[config.label_in_feature_index]  # 通过保存的均值和方差还原数据
    assert label_data.shape[0] == predict_data.shape[
        0], "The element number in origin and predicted data is different"  # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。

    label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_index]
    label_column_num = len(config.label_columns)

    # label 和 predict 是错开config.predict_day天的数据的
    # 下面是两种norm后的loss的计算方式，结果是一样的，可以简单手推一下
    # label_norm_data = origin_data.norm_data[origin_data.train_num + origin_data.start_num_in_test:,
    #              config.label_in_feature_index]
    # loss_norm = np.mean((label_norm_data[config.predict_day:] - predict_norm_data[:-config.predict_day]) ** 2, axis=0)
    # logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    loss = np.mean((label_data[config.predict_day:] - predict_data[:-config.predict_day]) ** 2, axis=0)
    loss_norm = loss / (origin_data.std[config.label_in_feature_index] ** 2)
    logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))  # str() 函数将对象转化为适于人阅读的形式

    label_X = range(origin_data.data_num - origin_data.train_num - origin_data.start_num_in_test)  # 得到实际的测试数据量
    predict_X = [x + config.predict_day for x in label_X]

    #  if not sys.platform.startswith('linux'):    # 无桌面的Linux下无法输出，如果是有桌面的Linux，如Ubuntu，可去掉这一行
    for i in range(label_column_num):
        plt.figure(i + 1)  # 预测数据绘制
        print("标签")
        plt.plot(label_X, label_data[:, i], 'r', label='label')
        plt.plot(predict_X, predict_data[:, i], 'c', label='predict')
        plt.title("Predict  {}  with {}".format(label_name[i], config.used_frame))
        logger.info("The predicted  {} for the next {} (s) is: ".format(label_name[i], config.predict_day) +
                    str(np.squeeze(predict_data[-config.predict_day:, i])))
        if config.do_figure_save:
            print("draw")
            plt.savefig(config.figure_save_path + "{}predict_{}_with_{}.png".format(config.continue_flag, label_name[i],
                                                                                    config.used_frame))

    plt.show()


def data_restore(config: Config, origin_data: Data, logger, predict_norm_data: np.ndarray):
    predict_data = predict_norm_data * origin_data.std[config.label_in_feature_index] + \
                   origin_data.mean[config.label_in_feature_index]  # 通过保存的均值和方差还原数据
    return predict_data


# class add_function:
#     def add_data(self):
#         csv_file=open('./data/mobb-tput-rtt-notime.csv')    #打开csv文件
#         csv_reader_lines = csv.reader(csv_file)   #逐行读取csv文件
#         date=[]    #创建列表准备接收csv各行数据
#         line = 0
#         for one_line in csv_reader_lines:
#             date.append(one_line)    #将读取的csv分行数据按行存入列表‘date’中
#             line = line + 1    #统计行数
#         i=0
#         while i < line:
#             # print (date[i][0])    #访问列表date中的数据验证读取成功（这里是打印所有学生的姓名）
#             add_data = date[i]
#             i = i+1
#         return add_data

#     # def add_train(self):

#     #     return latset_model

#     # def add_predict(self):
#     #     return rtt,tput


def main(config):
    logger = load_logger(config)
    # Python try...except 语句捕获异常。它用于测试代码中写在“try”语句中的错误。如果遇到错误，则运行“except”块的内容。
    try:
        np.random.seed(config.random_seed)  # 设置随机种子，保证可复现
        data_gainer = Data(config)
        data_gainer_add_train = Data_add_train(config)
        test_p = queue.Queue(20)
        # train_p = queue.Queue(1000)
        train_p = queue.Queue(20)

        # if config.do_train:
        #     if config.add_train:
        #         train_x, train_y = data_gainer_add_train.get_train_and_valid_data()
        #         train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.15,
        #                                             random_state=config.random_seed,
        #                                             shuffle=config.shuffle_train_data)   # 划分训练和验证集，并打乱
        #         # print(len(train_y))
        #         # print(len(train_x))
        #         # print(len(valid_x))
        #         # print(len(valid_y))
        #
        #         for line in range(len(train_x)):
        #             # train_p.put([train_x[line:line+40], train_y[line:line+40], valid_x[line:line+40], valid_y[line:line+40]])
        #             train_p.put([train_x[line], train_y[line], valid_x[line], valid_y[line]])
        #             train_XX, train_YY, valid_XX, valid_YY = train_p.get()
        #             # train(config, logger, [train_XX, train_YY, valid_XX, valid_YY])
        #
        #             train_add(config, logger, [train_XX, train_YY, valid_XX, valid_YY],counter)
        #             # time.sleep(100)
        #
        #
        #     else:
        #         train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data()
        #         train(config, logger, [train_X, train_Y, valid_X, valid_Y])

        if config.do_predict:
            if config.add_train:
                test_X, test_Y = data_gainer_add_train.get_test_data(return_label_data=True)
                for line in range(len(test_X)):
                    test_p.put(test_X[line])
                    test_XX = test_p.get()
                    pred_result = predict_add(config, test_XX)  # 这里输出的是未还原的归一化预测数据
                    # draw(config, data_gainer, logger, pred_result)
                    pred_result = data_restore(config, data_gainer, logger, pred_result)
                    print(pred_result)
            else:
                test_X, test_Y = data_gainer.get_test_data(return_label_data=True)
                print(test_X)
                pred_result = predict(config, test_X)  # 这里输出的是未还原的归一化预测数据
                draw(config, data_gainer, logger, pred_result)


    except Exception:
        logger.error("Run Error", exc_info=True)


if __name__ == "__main__":
    import argparse

    # argparse方便于命令行下输入参数，可以根据需要增加更多
    parser = argparse.ArgumentParser()  # 创建一个解析对象S
    # 向该对象中添加你要关注的命令行参数和选项
    # parser.add_argument("-t", "--do_train", default=False, type=bool, help="whether to train")
    # parser.add_argument("-p", "--do_predict", default=True, type=bool, help="whether to train")
    # parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    # parser.add_argument("-e", "--epoch", default=20, type=int, help="epochs num")
    args = parser.parse_args()  # 进行解析

    con = Config()
    for key in dir(args):  # dir(args) 函数获得args所有的属性
        if not key.startswith("_"):  # 去掉 args 自带属性，比如__name__等
            setattr(con, key, getattr(args,
                                      key))  # 将属性值赋给Config;   getattr(a, 'bar')  # 获取属性 bar 值;setattr(a, 'bar', 5)  # 设置属性 bar 值

    main(con)
