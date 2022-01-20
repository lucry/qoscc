import time
import sys
import queue
# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from logging.handlers import RotatingFileHandler
from sklearn.model_selection import train_test_split
from model import *

class Config:
    # 数据参数
    feature_columns = [0]
    label_columns = [0]
    label_in_feature_index = (lambda x, y: [x.index(i) for i in y])(feature_columns, label_columns)  # 因为feature不一定从0开始
    scale_infactor = 5

    predict_day = 1  # 预测未来几天

    # 网络参数
    input_size = len(feature_columns)
    output_size = len(label_columns)

    hidden_size = 128  # LSTM的隐藏层大小，也是输出大小
    lstm_layers = 2  # LSTM的堆叠层数
    dropout_rate = 0.2  # dropout概率
    time_step = 20  # 原来是20，这个参数很重要，是设置用前多少天的数据来预测，也是LSTM的time step数，请保证训练数据量大于它
    predict_step = 1 # predict window size

    # 训练参数
    do_train = True
    do_predict = True
    add_train = True  # 是否载入已有模型参数进行增量训练
    shuffle_train_data = True  # 是否对训练数据做shuffle
    use_cuda = False  # 是否使用GPU训练

    train_data_rate = 0.85  # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    valid_data_rate = 0.15  # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

    # batch_size = 64->32
    batch_size = 16  # 一次训练所抓取的数据样本数量
    learning_rate = 0.001
    epoch = 50  # 原来为20->80，整个训练集被训练多少遍，不考虑早停的前提下
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
    used_frame = "tensorflow"  # 选择的深度学习框架，不同的框架模型保存后缀不一样
    model_postfix = {"pytorch": ".pth", "keras": ".h5", "tensorflow": ".ckpt"}
    model_name = "model_" + continue_flag + used_frame + model_postfix[used_frame]

    # 路径参数
    train_data = "./data/lb-tput-rtt-notime.csv"
    # train_data_path_add_train = "./data/mobb-tput-rtt-notime.csv"
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


class Data:
    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name = self.read_data()

        self.data_num = self.data.shape[0]  # shape[0]输出矩阵的行数，shape[1]输出列数 【数字】
        self.train_num = int(self.data_num * self.config.train_data_rate)  # 矩阵的行数*训练数据率【数字】
        # self.train_num = int(self.data_num)  # 矩阵的行数*训练数据率【数字】
        print("train_num is: ", self.train_num)

        # self.mean = np.mean(self.data, axis=0)  # 数据的均值，axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
        # self.std = np.std(self.data, axis=0)  # 数据的方差
        self.max = np.max(self.data, axis=0) #maximum data
        self.min = np.min(self.data, axis=0) #minimum data
        print("self.max is: ", self.max)
        print("self.min is: ", self.min)
        self.norm_data = (self.data-self.min) / (self.max-self.min)
        # self.norm_data = (self.data - self.mean) / self.std  # 归一化，去量纲【矩阵7*n】

        self.start_num_in_test = 0  # 测试集中前几天的数据会被删掉，因为它不够一个time_step

    def read_data(self):  # 读取初始数据
        values = []
        with open(self.config.train_data, "r") as f:
            line = f.readline() # label line
            label = line.split(",")
            # print("first line is: ", line)
            # print("lenth is: ", column)
            while True:
                line = f.readline()
                line = line.split(",")
                # print("line is: ", line)
                if (len(line)>1):
                    data = [float(i) for i in line]
                    values.append([data[0]/config.scale_infactor])
                else:
                    break
        # print("values are: ", values)
        values = np.array(values, float)
        return values,label[0]


    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:self.train_num]  # 取norm_data的前train_num部分数据
        label_data = self.norm_data[self.config.time_step: self.config.time_step + self.train_num+self.config.predict_step-1]  # 将延后几天的数据作为label,取预测天数到训练数据+预测天数中的所有label_in_feature_index维数据
        # x[:,n]表示在全部数组（维）中取第n个数据，直观来说，x[:,n]就是取所有集合的第n个数据
        # print(feature_data)
        # print(label_data)

        if not self.config.do_continue_train:
            # 在非连续训练模式下，每time_step行数据会作为一个样本，两个样本错开一行，比如：1-20行，2-21行...train_num-time_step至train_num行
            train_x = [feature_data[i:i + self.config.time_step] for i in range(self.train_num-self.config.time_step)]
            train_y = [label_data[i:i + self.config.predict_step] for i in range(self.train_num-self.config.time_step)]
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
                label_data[start_index + i * self.config.predict_step: start_index + (i + 1) * self.config.predict_step]
                for start_index in range(self.config.predict_step)
                for i in range((self.train_num - start_index) // self.config.time_step)]

        train_x, train_y = np.array(train_x), np.array(train_y)  # Numpy.array()用来产生数组
        # print("train_x is: ", train_x)
        # print("train_y is: ", train_y)
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
            label_data = self.norm_data[self.train_num + self.start_num_in_test]
            return np.array(test_x), label_data
        return np.array(test_x)

def draw(config, origin_data, logger, valid_data, predict_norm_data):
    label_data = valid_data * (origin_data.max[config.label_in_feature_index]-origin_data.min[config.label_in_feature_index]) + \
                   origin_data.min[config.label_in_feature_index]  # x[n,:]表示在n个数组（维）中取全部数据，直观来说，x[n,:]就是取第n集合的所有数据
    predict_data = predict_norm_data * (origin_data.max[config.label_in_feature_index]-origin_data.min[config.label_in_feature_index]) + \
                   origin_data.min[config.label_in_feature_index]   # 通过保存的均值和方差还原数据
    assert label_data.shape[0]==predict_data.shape[0], "The element number in origin and predicted data is different"#assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。

    label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_index]
    label_column_num = len(config.label_columns)

    # label 和 predict 是错开config.predict_day天的数据的
    # 下面是两种norm后的loss的计算方式，结果是一样的，可以简单手推一下
    # label_norm_data = origin_data.norm_data[origin_data.train_num + origin_data.start_num_in_test:,
    #              config.label_in_feature_index]
    # loss_norm = np.mean((label_norm_data[config.predict_day:] - predict_norm_data[:-config.predict_day]) ** 2, axis=0)
    # logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    loss = np.mean((label_data - predict_data ) ** 2, axis=0)
    # loss_norm = loss/(origin_data.std[config.label_in_feature_index] ** 2)
    logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss)) #str() 函数将对象转化为适于人阅读的形式

    label_X = range(np.shape(label_data)[0])   #得到实际的测试数据量
    # predict_X = [ x + config.predict_day for x in label_X]

    #  if not sys.platform.startswith('linux'):    # 无桌面的Linux下无法输出，如果是有桌面的Linux，如Ubuntu，可去掉这一行
    for i in range(label_column_num):
        plt.figure(i+1)                     # 预测数据绘制
        print("标签")
        plt.plot(label_X, label_data[:, i],'r',label='label')
        plt.plot(label_X, predict_data[:, i],'c', label='predict')
        plt.title("Predict  {}  with {}".format(label_name[i], config.used_frame))
        logger.info("The predicted  {} for the next {} (s) is: ".format(label_name[i], config.predict_day) +
                str(np.squeeze(predict_data[-config.predict_day:, i])))
        if config.do_figure_save:
            print("draw")
            plt.savefig(config.figure_save_path+"{}predict_{}_with_{}.png".format(config.continue_flag, label_name[i], config.used_frame))

    plt.show()


if __name__ == "__main__":
    config = Config()
    logger = load_logger(config)
    params = Params('params.json')
    try:
        np.random.seed(config.random_seed)  # 设置随机种子，保证可复现
        data_gainer = Data(config)
        test_p = queue.Queue(20)
        train_p = queue.Queue(20)
        train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data()

        # if config.do_train_visualized:  # loss可视化
        #     from tensorboardX import SummaryWriter
        #
        #     train_writer = SummaryWriter(config.log_save_path + "Train")
        #     eval_writer = SummaryWriter(config.log_save_path + "Eval")

        if config.do_train:
            with tf.variable_scope("stock_predict",
                                   reuse=tf.AUTO_REUSE):  # , tf.variable_scope("stock_predict", reuse=tf.AUTO_REUSE):#
                model = LSTMNetwork(params,False)
            train_len = len(train_X)  # = 1242
            valid_len = len(valid_X)  # = 220

            if config.use_cuda:  # 开启GPU训练会有很多警告，但不影响训练
                sess_config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)  # 在CUDA可用时会自动选择GPU，否则CPU
                sess_config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 显存占用率
                sess_config.gpu_options.allow_growth = True  # 初始化时不全部占满GPU显存, 按需分配

            else:
                sess_config = None
            with tf.Session(config=sess_config) as sess:
                sess.run(tf.global_variables_initializer())  # 初始化模型的参数

                valid_loss_min = float("inf")
                bad_epoch = 0
                global_step = 0
                for epoch in range(config.epoch):
                    logger.info("Epoch {}/{}".format(epoch, config.epoch))
                    # 训练
                    train_loss_array = []
                    # print("shape train_X: ", np.shape(train_X[0 * config.batch_size: (0 + 1) * config.batch_size]))
                    # print("shape train_Y: ", np.shape(train_Y[0 * config.batch_size: (0 + 1) * config.batch_size]))
                    for step in range(train_len // config.batch_size):  # //取整除 - 返回商的整数部分（向下取整）
                        feed_dict = {model.X: train_X[step * config.batch_size: (step + 1) * config.batch_size],
                                     # 用feed_dict以字典的方式填充占位，将数据投入到[dist]中
                                     model.Y: train_Y[step * config.batch_size: (step + 1) * config.batch_size]}
                        train_loss, _ = sess.run([model.loss, model.optim], feed_dict=feed_dict)
                        train_loss_array.append(train_loss)
                        # if config.do_train_visualized and global_step % 100 == 0:  # 每一百步显示一次
                        #     train_writer.add_scalar('Train_Loss', train_loss, global_step + 1)
                        global_step += 1

                    # 验证与早停
                    valid_loss_array = []
                    for step in range(valid_len // config.batch_size):
                        feed_dict = {model.X: valid_X[step * config.batch_size: (step + 1) * config.batch_size],
                                     # 用feed_dict以字典的方式填充占位，将数据投入到[dist]中
                                     model.Y: valid_Y[step * config.batch_size: (step + 1) * config.batch_size]}
                        valid_loss = sess.run(model.loss, feed_dict=feed_dict)
                        valid_loss_array.append(valid_loss)

                    train_loss_cur = np.mean(train_loss_array)
                    valid_loss_cur = np.mean(valid_loss_array)
                    logger.info("The train loss is {:.6f}. ".format(train_loss_cur) +
                                "The valid loss is {:.6f}.".format(valid_loss_cur))
                    # if config.do_train_visualized:
                    #     train_writer.add_scalar('Epoch_Loss', train_loss_cur, epoch + 1)
                    #     eval_writer.add_scalar('Epoch_Loss', valid_loss_cur, epoch + 1)
                    # 早停判断
                    if valid_loss_cur < valid_loss_min:
                        valid_loss_min = valid_loss_cur
                        bad_epoch = 0
                        model.saver.save(sess, config.model_save_path + config.model_name)
                    else:
                        bad_epoch += 1
                        if bad_epoch >= config.patience:
                            logger.info(" The training stops early in epoch {}".format(epoch))
                            break



        # if config.do_predict:
        #     # valid_X, valid_Y = data_gainer.get_test_data(return_label_data=True)
        #     # print(valid_X)
        #
        #     params.dict["dropout_rate"] = 0  # 预测模式要调为1
        #
        #     tf.reset_default_graph()  # 清除默认图的堆栈，并设置全局图为默认图
        #     with tf.variable_scope("stock_predict", reuse=tf.AUTO_REUSE):  # 共享变量
        #         model = LSTMNetwork(params, False)
        #
        #     test_len = len(valid_X)
        #     with tf.Session() as sess:  # session是客户端与整个TensorFlow系统交互的接口
        #         module_file = tf.train.latest_checkpoint(config.model_save_path)
        #         model.saver.restore(sess, module_file)
        #
        #         result = np.zeros((test_len, config.output_size))  # np.zeros((2, 1)) 生成两行一列的0矩阵
        #         for step in range(test_len):
        #             feed_dict = {model.X: valid_X[step: (step + 1)]}
        #             test_pred = sess.run(model.pred, feed_dict=feed_dict)
        #             result[step: (step + 1)] = test_pred[0, :, :]
        #
        #     # print("np.shape(valid_Y) is: ", np.shape(valid_Y))
        #     # print("np.shape(result) is: ", np.shape(result))
        #     valid_Y = np.reshape(valid_Y, [test_len, config.output_size])
        #     draw(config, data_gainer, logger, valid_Y, result)

    except Exception:
        logger.error("Run Error", exc_info=True)

