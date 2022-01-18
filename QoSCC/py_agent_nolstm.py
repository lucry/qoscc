import math
import pickle
import random
import os
import signal
import sys
import threading
import time
import sysv_ipc
import logging
import argparse

import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from utils import logger, Params
from model import DRLNetwork, LSTMNetwork


class PY_Agent:
    def __init__(self, shmem, shmem_rl):
        self.target_thr = 100.0  # target throughput
        self.target_rtt = 0.1  # target RTT
        self.w_thr = 0.5
        self.w_rtt = 0.5

        self.load = False
        self.counter = 0
        self.file_path = os.path.dirname(__file__)
        self.params = Params(os.path.join(self.file_path, 'params.json'))
        self.train_dir = os.path.join(self.file_path, self.params.dict['logdir'])
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.s_dim = self.params.dict['state_dim']
        self.a_dim = self.params.dict['action_dim']
        if self.params.dict['use_hard_target'] == True:
            self.params.dict['tau'] = 1.0

        #make sessio
        tf.set_random_seed(1234)
        random.seed(1234)
        np.random.seed(1234)
        summary_writer = tf.summary.FileWriterCache.get(self.train_dir)
        self.drl_model = DRLNetwork(self.s_dim, self.a_dim, batch_size=self.params.dict['batch_size'],
                                summary=summary_writer,
                                gamma=self.params.dict['gamma'],
                                epsilon=self.params.dict['epsilon'],
                                epsilon_min=self.params.dict['epsilon_min'],
                                explore_step=self.params.dict['explore_step'],
                                lr_a=self.params.dict['lr_a'], lr_c=self.params.dict['lr_c'],
                                action_scale=self.params.dict['action_scale'], action_range=self.params.dict['action_range'],
                                stddev=self.params.dict['stddev'],
                                LOSS_TYPE=self.params.dict['LOSS_TYPE'], tau=self.params.dict['tau'],
                                PER=self.params.dict['PER'],
                                noise_type=self.params.dict['noise_type'],
                                noise_exp=self.params.dict['noise_exp'],
                                mem_size=self.params.dict['memsize'])
        self.drl_model.build_learn()
        self.drl_model.create_tf_summary()

        if self.params.dict['use_cuda']:
            sess_config = tf.ConfigProto(log_device_placement=True,
                                         allow_soft_placement=True)  # auto-select GPU if cuda is available, else CPG
            sess_config.gpu_options.per_process_gpu_memory_fraction = 0.7  # Video memory occupancy rate
            sess_config.gpu_options.allow_growth = True  # GPU memory is not fully occupied during initialization, allocated on demand
        else:
            sess_config = None
        self.mon_sess = tf.Session(config=sess_config)
        self.drl_model.assign_sess(self.mon_sess)
        if self.load is False:
            self.drl_model.init_target()


        self.lstm_model = LSTMNetwork(self.params, True)

        self.s0 = np.zeros([self.s_dim])
        self.a = np.random.random() #random init action

        self.shmem = sysv_ipc.SharedMemory(shmem)
        self.shmem_rl = sysv_ipc.SharedMemory(shmem_rl)
        self.prev_ack = -1

        signal.signal(signal.SIGINT, self.handler_term)
        signal.signal(signal.SIGTERM, self.handler_term)

        self.shmem_rl.write("88888")
        print("Write information in the share memory...")

        if os.path.exists("state.log"):
            os.remove("state.log")


    def handler_term(self):
        print("python program terminated usking Kill -15")
        sys.exit(0)

    def func_reward(self, x1, x2):
        if x1>x2:
            return 1
        else:
            return x1/(x2+1e-7)

    def get_state(self):
        logging.basicConfig(level=logging.INFO, filename='state.log', filemode='a',
                            format='%(asctime)s - %(levelname)s: %(message)s')

        error_cnt = 0
        succeed = False
        while True:
            try:
                state_mem = self.shmem.read()
            except sysv_ipc.ExistentialError:
                print("No shared memory Now, python ends gracefully :)")
                sys.exit(0)
            state_mem = state_mem.decode('unicode_escape')
            i = state_mem.find('\0')
            if i != -1:
                state_mem = state_mem[:i]
                read_mem = np.fromstring(state_mem, dtype=float, sep=' ')
                try:
                    ack = read_mem[0]
                except :
                    ack = self.prev_ack
                    time.sleep(0.0001)
                    continue
                try:
                    s0 = read_mem[1:]
                except:
                    print("s0 waring")
                    time.sleep(0.0001)
                    continue
                if ack != self.prev_ack:
                    succeed = True
                    break
            error_cnt += 1
            if error_cnt > 240000:
                # self.keep_alive = False
                print("After 4 min, We didn't get any state from the server. Actor is going down down down ...\n")
                sys.exit(0)
            time.sleep(0.0001)

        if succeed == False:
            raise ValueError('Read nothing new from shrmem for a long time')


        state = np.zeros(1)
        if len(s0) == self.params.dict['input_dim']:
            max_thr = s0[0]
            min_rtt = s0[1]
            sending_rate = s0[2]
            # print("sending rate is: ", sending_rate)
            self.target_thr = s0[3]
            self.target_rtt = s0[4]
            self.w_thr = s0[5]
            self.w_rtt = s0[6]
            reward =  s0[7]
            sending_period = s0[8]
            cap_bw, cap_rtt = self.predict_network(max_thr, min_rtt)
            state[0] = max_thr
            state = np.append(state, [min_rtt])
            # state = np.append(state, [cap_bw])
            # state = np.append(state, [cap_rtt])
            state = np.append(state, [sending_rate])
            # print("python: get a state ", ack, s0)


            # if (sending_period > 0):
            #     reward = self.w_thr * self.func_reward(sending_rate, self.target_thr)
            # else:
            #     reward = -1


            self.drl_model.store_experience(self.s0, self.a, reward, state, False)
            # print(self.s0 == state)
            # print("get a state. Length of rp_buf is: ", self.drl_model.rp_buffer.ptr)
            self.s0 = state
            # self.a = sending_rate
            self.prev_ack = ack

            logging.info("state: "+str(s0)+" "+str(cap_bw)+str(cap_rtt))
            logging.info("reward: "+str(reward)+" "+ str(self.w_thr) + " " + str(self.func_reward(sending_rate, self.target_thr)) + " " +
                         str(self.func_reward(cap_bw, self.target_thr)) + " " + str(self.w_rtt) + " " + str(self.func_reward(self.target_rtt, min_rtt)) + " "+
                         str(self.func_reward(self.target_rtt, cap_rtt)))
            # self.num += 1
            logging.info("action: " + str(self.a) + " " + str(ack))

            return state, reward, ack, False
        else:
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@ an ilegal state.")
            return state, 0, -1, True




    def write_action(self, a1, ack):
        msg = str(ack)+" "+str(a1)+" \0"
        # print("write into shared memory: ", msg)
        self.shmem_rl.write(msg)
        pass


    def predict_network(self, max_thr, min_rtt):
        self.lstm_model.store([max_thr, min_rtt])
        res = self.lstm_model.predict()
        cap_thr = res[0][0]
        cap_rtt = res[0][1]
        # print("cap_thr is: ", cap_thr)
        # print("cap_rtt is: ", cap_rtt)
        return cap_thr, cap_rtt


    def get_action(self, s1, ack, error_code):
        # logging.basicConfig(level=logging.INFO, filename='state.log', filemode='w',
        #                     format='%(asctime)s - %(levelname)s: %(message)s')

        if not error_code:
            a1 = self.drl_model.get_action(s1, self.params.dict['use_noise'])
            a1 = a1[0][0][0]
            self.a = a1
            self.write_action(a1, ack)
            # print("wirte action: ", a1, ack)
        else:
            # print("Invalid state received...")
            # print("a1 is: ", self.a)
            self.write_action(self.a, self.prev_ack)
            # print("wirte  prev action: ", self.a, self.prev_ack)


    def train(self):
        # take enough samples for training
        if self.drl_model.rp_buffer.ptr > 200 or self.drl_model.rp_buffer.full:
            self.drl_model.train_step()
            if self.params.dict['use_hard_target'] == False:
                self.drl_model.target_update()
                if self.counter % self.params.dict['hard_target'] == 0:
                    current_opt_step = self.drl_model.sess.run(self.drl_model.global_step)
                    logger.info("Optimize step:{}".format(current_opt_step))
                    logger.info("rp_buffer ptr:{}".format(self.drl_model.rp_buffer.ptr))
                    logger.info("rp_buffer is full:{}".format(self.drl_model.rp_buffer.full))
            else:
                if self.counter % self.params.dict['hard_target'] == 0:
                    self.drl_model.target_update()
                    current_opt_step = self.drl_model.sess.run(self.drl_model.global_step)
                    logger.info("Optimize step:{}".format(current_opt_step))
                    logger.info("rp_buffer ptr:{}".format(self.drl_model.rp_buffer.ptr))
                    logger.info("rp_buffer is full:{}".format(self.drl_model.rp_buffer.full))

            self.counter += 1


'''
main function
'''
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--sh_key', type=int, required=True, help='Share memory key')
    parser.add_argument('--sh_key_rl', type=int, required=True, help='Share RL memory key')
    config = parser.parse_args()
    shrmem = config.sh_key
    shrmem_rl = config.sh_key_rl

    agent = PY_Agent(shrmem, shrmem_rl)
    while True:
        t1 = time.time()
        s1, reward, ack, error_code = agent.get_state()
        t2 = time.time()
        # print("Get state time is: ", t2-t1)
        agent.train()
        t3 = time.time()
        # print("Training time is: ", t3 - t2)
        agent.get_action(s1, ack, error_code)
        t4 = time.time()
        # print("Get action time is: ", t4 - t3)
