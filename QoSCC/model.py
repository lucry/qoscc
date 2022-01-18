import os
import random

import numpy as np
import tensorflow as tf

from utils import *

def create_input_op_shape(obs, tensor):
    input_shape = [x or -1 for x in tensor.shape.as_list()]
    return np.reshape(obs, input_shape)

class Actor:
    def __init__(self, s_dim, a_dim, h1_dim, h2_dim, action_scale=1.0, name='actor'):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.action_scale = action_scale
        self.name = name

    def build(self, s, is_training):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            h1 = tf.layers.dense(s, units=self.h1_dim, name='fc1')
            h1 = tf.layers.batch_normalization(h1, training=is_training, scale=False)
            h1 = tf.nn.leaky_relu(h1)

            h2 = tf.layers.dense(h1, units=self.h2_dim, name='fc2')
            h2 = tf.layers.batch_normalization(h2, training=is_training, scale=False)
            h2 = tf.nn.leaky_relu(h2)

            output = tf.layers.dense(h2, units=self.a_dim, activation=tf.nn.tanh)
            scale_output = tf.multiply(output, self.action_scale)

        return scale_output

    def train_var(self): # return all parameters
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Critic:
    def __init__(self, s_dim, a_dim, h1_dim, h2_dim, name='critic'):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.name = name

    def build(self, s, action):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            h1 = tf.layers.dense(s, units=self.h1_dim, activation=tf.nn.leaky_relu, name='fc1')
            h2 = tf.layers.dense(tf.concat([h1, action], -1), units=self.h2_dim, activation=tf.nn.leaky_relu, name='fc2')
            output = tf.layers.dense(h2, units=1)

        return output

    def train_var(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class DRLNetwork:
    def __init__(self, s_dim, a_dim, gamma=0.995, epsilon=0.75, epsilon_min=0.1, explore_step=1000, batch_size=8, lr_a=1e-4, lr_c=1e-3, h1_dim=128,
                 h2_dim=128, action_scale=1.0, action_range=(-1.0, 1.0), summary=None, stddev=0.1,
                 LOSS_TYPE='HUBER', tau=1e-3, PER=False, noise_type=3, noise_exp=50000, mem_size=1e5):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        # self.action_scale = action_scale
        self.action_range = action_range
        self.gamma = gamma
        self.epsilon = epsilon
        self.explore_step = (epsilon-epsilon_min) / explore_step
        self.epsilon_min = epsilon_min
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.stddev =stddev
        self.LOSS_TYPE = LOSS_TYPE
        self.train_dir = './train_dir'
        self.step_epochs = tf.Variable(0, trainable=False, name='epoch')
        # self.global_step = tf.train.get_or_create_global_step(graph=None)
        self.global_step = tf.compat.v1.train.get_or_create_global_step(graph=None)
        self.saver = tf.train.Saver()
        self.begin_train = False

        self.s0 = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='s0')
        self.s1 = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='s1')
        self.is_training = tf.placeholder(tf.bool, name='Actor_is_training')
        self.action = tf.placeholder(tf.float32, shape=[None, a_dim], name='action')

        # Main Actor/Critic Network
        self.actor = Actor(self.s_dim, self.a_dim, self.h1_dim, self.h2_dim, action_scale=action_scale, name='main_actor')
        self.critic = Critic(self.s_dim, self.a_dim, self.h1_dim, self.h2_dim, name='main_critic')
        self.actor_out = self.actor.build(self.s0, True)
        self.critic_out = self.critic.build(self.s0, self.action)
        self.critic_actor_out = self.critic.build(self.s0, self.actor_out)

        #Target Actor/Critic Network
        self.target_actor = Actor(self.s_dim, self.a_dim, self.h1_dim, self.h2_dim, action_scale=action_scale, name='target_actor')
        self.target_actor_out = self.target_actor.build(self.s1, False)

        self.target_actor_update_op = self.target_update_op(self.target_actor.train_var(), self.actor.train_var(), tau)
        self.target_actor_init_op = self.target_init_op(self.target_actor.train_var(), self.actor.train_var())
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        self.terminal = tf.placeholder(tf.float32, shape=[None, 1], name='is_terminal')
        self.reward = tf.placeholder(tf.float32, shape=[None, 1], name='reward')
        self.y = self.reward + self.gamma * (1-self.terminal)*self.critic_actor_out


        self.summary_writer = summary

        self.noise_type = noise_type
        self.noise_exp = noise_exp
        if noise_type == 1:
            self.actor_noise = OU_Noise(mu=np.zeros(a_dim), sigma=float(self.stddev) * np.ones(a_dim),dt=1,exp=self.noise_exp)
        elif noise_type == 2:
            ## Gaussian with gradually decay
            self.actor_noise = G_Noise(mu=np.zeros(a_dim), sigma=float(self.stddev) * np.ones(a_dim), explore =self.noise_exp)
        elif noise_type == 3:
            ## Gaussian without gradually decay
            self.actor_noise = G_Noise(mu=np.zeros(a_dim), sigma=float(self.stddev) * np.ones(a_dim), explore = None,theta=0.1)
        elif noise_type == 4:
            ## Gaussian without gradually decay
            self.actor_noise = G_Noise(mu=np.zeros(a_dim), sigma=float(self.stddev) * np.ones(a_dim), explore = EXPLORE,theta=0.1,mode="step",step=NSTEP)
        elif noise_type == 5:
            self.actor_noise = None
        else:
            self.actor_noise = OU_Noise(mu=np.zeros(a_dim), sigma=float(self.stddev) * np.ones(a_dim),dt=0.5)

        self.rp_buffer = ReplayBuffer(int(mem_size), s_dim, a_dim, batch_size=batch_size)



    def target_update_op(self, target, vars, tau):
        return [tf.assign(target[i], vars[i]*tau+target[i]*(1-tau)) for i in range(len(vars))]


    def target_init_op(self, target, vars):
        return [tf.assign(target[i], vars[i]) for i in range(len(vars))]


    def build_learn(self):
        self.actor_optimizer = tf.train.AdamOptimizer(self.lr_a)
        self.critic_optimizer = tf.train.AdamOptimizer(self.lr_c)

        self.actor_train_op = self.build_actor_train_op()
        self.critic_train_op = self.build_critic_train_op()


    def build_actor_train_op(self):
        self.actor_loss = -tf.reduce_mean(self.critic_actor_out) # calculate the average of tensor
        return self.actor_optimizer.minimize(self.actor_loss, var_list=self.actor.train_var(), global_step=self.global_step)

    def build_critic_train_op(self):
        def mse(y, ypred, weights=1.0):
            error = tf.square(y-ypred)
            weighted_error = tf.reduce_mean(error*weights)
            return weighted_error

        loss_function = {
            'HUBER' : tf.compat.v1.losses.huber_loss,
            'MSE' : mse
        }
        self.critic_loss = loss_function[self.LOSS_TYPE](self.y, self.critic_out)
        loss_op = self.critic_optimizer.minimize(self.critic_loss, var_list=self.critic.train_var(), global_step=self.global_step)

        return loss_op


    def create_tf_summary(self):
        tf.summary.scalar('Loss/critic_loss', self.critic_loss)
        tf.summary.scalar('Loss/actor_loss', self.actor_loss)
        self.summary_op = tf.summary.merge_all()


    def init_target(self):
        self.sess.run(self.target_actor_init_op)
        # self.sess.run(self.target_critic_init_op)


    def assign_sess(self, sess):
        self.sess = sess
        init = tf.global_variables_initializer()
        self.sess.run(init)


    def target_update(self):
        # self.sess.run([self.target_actor_update_op, self.target_critic_update_op])
        self.sess.run([self.target_actor_update_op])


    def get_action(self, s, use_noise=True):
        explore = random.random()
        if explore < self.epsilon:# or not self.begin_train:
            action = [[[random.uniform(self.action_range[0], self.action_range[1])]]]
            # print("take an random action.")
        else:
            fd = {self.s0: create_input_op_shape(s, self.s0), self.is_training:False}
            action = self.sess.run([self.actor_out], feed_dict=fd)
            # print("model action: ", action)
        if use_noise:
            noise = self.actor_noise(action[0])
            action += noise
            action = np.clip(action, self.action_range[0], self.action_range[1])
        # print("drl output is: ", action)
        if self.epsilon>self.epsilon_min:
            self.epsilon -= self.explore_step

        return action


    def get_q(self, s, a):
        fd = {self.s0: create_input_op_shape(s, self.s0),
              self.action: create_input_op_shape(a, self.action)}

        return self.sess.run([self.critic_out], feed_dict=fd)

    def get_q_actor(self, s):
        fd = {self.s0: create_input_op_shape(s, self.s0)}

        return self.sess.run([self.critic_actor_out], feed_dict=fd)

    def train_step(self):
        extra_update_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'main_actor' in v.name]

        self.begin_train = True
        batch_samples = self.rp_buffer.sample()
        fd = {self.s0: create_input_op_shape(batch_samples[0], self.s0),
              self.action: create_input_op_shape(batch_samples[1], self.action),
              self.reward: create_input_op_shape(batch_samples[2], self.reward),
              self.s1: create_input_op_shape(batch_samples[3], self.s1),
              self.terminal: create_input_op_shape(batch_samples[4], self.terminal),
              self.is_training: True
              }

        self.sess.run([self.critic_train_op], feed_dict=fd)
        # print("self.y is: ",self.y.eval(session=self.sess))
        self.sess.run([self.actor_train_op, extra_update_ops], feed_dict=fd)
        summary, step = self.sess.run([self.summary_op, self.global_step], feed_dict=fd)
        self.summary_writer.add_summary(summary, global_step=step)


    def save_model(self, step=None):
        self.saver.save(self.sess, os.path.join(self.train_dir, 'model'), global_step =step)


    def store_experience(self, s0, a, r, s1, terminal):
        self.rp_buffer.store(s0, a, r, s1, terminal)


    def store_many_experience(self, s0, a, r, s1, terminal, length):
        self.rp_buffer.store_many(s0, a, r, s1, terminal, length)


    def sample_experince(self):
        return self.rp_buffer.sample()


    def load_model(self, name=None):
        if name is not None:
            print(os.path.join(self.train_dir, name))
            self.saver.restore(self.sess, os.path.join(self.train_dir, name))
        else:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.train_dir))


    def update_step_epochs(self, epoch):
        self.sess.run(tf.assign(self.step_epochs, epoch))


    def get_step_epochs(self):
        return self.sess.run(self.step_epochs)


class LSTMNetwork:
    def __init__(self, params, load, sess=tf.Session(), h_dim=128, lstm_layers=2):
        self.params = params
        self.h_dim = h_dim
        self.lstm_layer = lstm_layers

        self.X = tf.placeholder(tf.float32, [None, self.params.dict['history_window'], self.params.dict['capacity_dim']])
        self.Y = tf.placeholder(tf.float32, [None, self.params.dict['predict_window'], self.params.dict['capacity_dim']])
        self.net()
        self.operate(sess, load)
        self.train_loss = 0


        self.dataset = TimeSequence(self.params.dict['history_window'], self.params.dict['predict_window'])
        self.max = np.array([3080, 100])
        self.min = np.array([0, 20])


    def net(self):
        def dropout_cell():
            basicLstm = tf.nn.rnn_cell.LSTMCell(self.h_dim, state_is_tuple=True, activation=tf.nn.relu)  # 构建固定隐藏层的LSTM
            dropoutLstm = tf.nn.rnn_cell.DropoutWrapper(basicLstm,
                                                        output_keep_prob=1 - self.params.dict['dropout_rate'])  # 封装一个dropout的函数
            return dropoutLstm

        # 多层LSTMCell ：tf.nn.rnn_cell.MultiRNNCell
        cell = tf.nn.rnn_cell.MultiRNNCell([dropout_cell() for _ in range(self.lstm_layer)])

        # tf.nn.dynamic_rnn 函数是tensorflow封装的用来实现递归神经网络（RNN）的函数
        output_rnn, _ = tf.nn.dynamic_rnn(cell=cell, inputs=self.X, dtype=tf.float32)

        output_rnn = tf.slice(output_rnn, [0,self.params.dict['history_window']-1,0],[-1,-1,-1])
        # shape of output_rnn is: [batch_size, history_window, hidden_size], units=2

        self.pred = tf.layers.dense(inputs=output_rnn,
                                    units=self.params.dict['capacity_dim'])  # 全连接层  相当于添加一个层，通常在CNN的尾部进行重新拟合，减少特征信息的损失

    def operate(self, sess, load):
        # print("shape of self.pred is: ", tf.shape(self.pred))
        # self.loss = tf.reduce_mean(tf.square(tf.reshape(self.pred, [-1]) - tf.reshape(self.Y, [-1])))
        self.loss = tf.reduce_mean(tf.square(self.pred- self.Y))
        self.optim = tf.train.AdamOptimizer(self.params.dict['lr_l']).minimize(self.loss)  # tf.train.AdamOptimizer()函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。
        self.sess = sess
        init = tf.global_variables_initializer()
        self.sess.run(init)
        if (load):
            self.saver = tf.train.import_meta_graph(self.params.dict['model_save_path']+"model_tensorflow.ckpt.meta")
        else:
            self.saver = tf.train.Saver(tf.global_variables())  # tf.train.Saver()保存模型，tf.global_variables()功能：返回所有变量.

    def store(self, data):
        self.dataset.store(data)

    def train(self, is_incremental):
        if is_incremental:
            X,Y = self.dataset.get_dataset(self.dataset.increment)
        else:
            X, Y = self.dataset.get_dataset(self.dataset.history)
        X = (X-self.min) / (self.max-self.min)
        Y = (Y-self.min) / (self.max-self.min)
        for i in range(np.shape(X)[0] // self.params.dict['batch_size']):
            train_X = X[i*self.params.dict['batch_size']:(i+1)*self.params.dict['batch_size']]
            train_Y = Y[i*self.params.dict['batch_size']:(i+1)*self.params.dict['batch_size']]
            feed_dict = {self.X:train_X, self.Y:train_Y}
            train_loss, _ = self.sess.run([self.loss, self.optim], feed_dict=feed_dict)

    def predict(self):
        pred_X = np.zeros([1, self.params.dict['history_window'], self.params.dict['capacity_dim']])
        X = np.array(self.dataset.latest_hw_seq)
        pred_X[0,(self.params.dict['history_window']-np.shape(X)[0]):, :] = X[:,:]
        pred_X = (pred_X - self.min) / (self.max - self.min)
        feed_dict = {self.X:pred_X}
        pred = self.sess.run(self.pred, feed_dict=feed_dict)
        pred = pred[0,:,:]
        # print("pred is: ", pred)
        pred = pred*(self.max-self.min) +self.min
        return pred




