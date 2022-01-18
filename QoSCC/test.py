import pickle
import random
import os
import signal
import sys
import threading
import time

import numpy as np
import tensorflow as tf

from utils import logger, Params
from model import DRLNetwork, LSTMNetwork

class PY_Agent:
    def __init__(self, jobname, task):
        self.cap_thr = 0.0  # network capacity of throughput
        self.cap_rtt = 0.0  # network capacity of RTT
        self.curr_thr = 0.0  # calculated throughput in this interval
        self.curr_rtt = 0.0  # calculated RTT in this interval
        self.target_thr = 0.0  # target throughput
        self.target_rtt = 0.0  # target RTT
        self.w_thr = 0.5
        self.w_rtt = 0.5

        self.job_name = jobname
        self.task = task
        self.eval = False
        self.load = False
        self.epoch = 0
        self.file_path = os.path.dirname(__file__)
        self.params = Params(os.path.join(self.file_path, 'params.json'))
        self.train_dir = os.path.join(self.file_path, self.params.dict['logdir'], self.job_name + str(self.task))
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.s_dim = self.params.dict['state_dim']
        self.a_dim = self.params.dict['action_dim']
        if self.params.dict['use_hard_target'] == True:
            self.params.dict['tau'] = 1.0


	#make session
        with tf.Graph().as_default():
            # tf.compat.v1.set_random_seed(1234)
            tf.set_random_seed(1234)
            random.seed(1234)
            np.random.seed(1234)
            summary_writer = tf.summary.FileWriterCache.get(self.train_dir)
            # summary_writer = tf.compat.v1.summary.FileWriterCache(self.train_dir)
            self.agent = DRLNetwork(self.s_dim, self.a_dim, batch_size=self.params.dict['batch_size'],
                                    summary=summary_writer,
                                    gamma=self.params.dict['gamma'],
                                    lr_a=self.params.dict['lr_a'], lr_c=self.params.dict['lr_c'],
                                    stddev=self.params.dict['stddev'],
                                    LOSS_TYPE=self.params.dict['LOSS_TYPE'], tau=self.params.dict['tau'],
                                    PER=self.params.dict['PER'],
                                    noise_type=self.params.dict['noise_type'],
                                    noise_exp=self.params.dict['noise_exp'],
                                    mem_size=self.params.dict['memsize'])
            dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
            shapes = [[self.s_dim], [self.a_dim], [1], [self.s_dim], [1]]
            self.queue = tf.FIFOQueue(10000, dtypes, shapes, shared_name='rp_buf')

            self.agent.build_learn()
            self.agent.create_tf_summary()

            if self.params.dict['use_cuda']:
                sess_config = tf.ConfigProto(log_device_placement=True,
                                             allow_soft_placement=True)  # 在CUDA可用时会自动选择GPU，否则CPU
                sess_config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 显存占用率
                sess_config.gpu_options.allow_growth = True  # 初始化时不全部占满GPU显存, 按需分配
            else:
                sess_config = None
            self.mon_sess = tf.Session(config=sess_config)
            self.agent.assign_sess(self.mon_sess)

            self.s0 = np.zeros([self.s_dim])
            self.s1 = np.zeros([self.s_dim])
            self.a = [0]

if __name__ == "__main__":
	print("Hello world!")
