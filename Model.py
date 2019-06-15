#!/usr/bin/python
# -*- coding: utf-8

import tensorflow as tf

########################################################################
# RNN モデル　クラス
########################################################################
class Model():
    def __init__(self,
                 dataset, series_length=72, feature_count=10,
                 num_of_neuron=20, rnn='BasicRNNell',
                 target_feature=['6702_close'], target_feature_count=1):
        self.SERIES_LENGTH = series_length
        FEATURE_COUNT = feature_count
        TARGET_FEATURE_COUNT = target_feature_count
        NUM_OF_NEURON = num_of_neuron
        self.TARGET_FEATURE = target_feature
        RNN = rnn

        #######################################################################
        # placeholder
        with tf.name_scope('input'):  # tensorboard用
            # 入力（placeholderメソッドの引数は、データ型、テンソルのサイズ）
            # 訓練データ
            self.x = tf.placeholder(tf.float32, [None, self.SERIES_LENGTH, FEATURE_COUNT])
            # 教師データ
            self.y = tf.placeholder(tf.float32, [None, TARGET_FEATURE_COUNT])

        #######################################################################
        # RNN Cell
        with tf.name_scope('RNN'):  # tensorboard用
            # RNNセルの作成
            if RNN == 'BasicRNNCell':
                print('BasicRNNCell')
                self.cell = tf.nn.rnn_cell.BasicRNNCell(NUM_OF_NEURON)
                initial_state = self.cell.zero_state(tf.shape(self.x)[0], dtype=tf.float32)
                self.outputs, self.last_state = tf.nn.dynamic_rnn(self.cell,
                                                                  self.x,
                                                                  initial_state=initial_state,
                                                                  dtype=tf.float32)
            elif RNN == 'BasicLSTMCell':
                print('BasicLSTMCell')
                self.cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_OF_NEURON)
                initial_state = self.cell.zero_state(tf.shape(self.x)[0], dtype=tf.float32)
                self.outputs, self.last_state = tf.nn.dynamic_rnn(self.cell,
                                                                  self.x,
                                                                  initial_state=initial_state,
                                                                  dtype=tf.float32)
            else:
                print('No RNN Cell Defined')
                exit(0)

        #######################################################################
        # 全結合
        with tf.name_scope('prediction'):  # tensorboard用
            # 重み
            # with tf.name_scope('W'):
            self.w = tf.Variable(tf.zeros([NUM_OF_NEURON, TARGET_FEATURE_COUNT]))
            # バイアス
            # with tf.name_scope('b'):
            self.b = tf.Variable([0.1] * TARGET_FEATURE_COUNT)
            # 最終出力（予測）
            if RNN == 'BasicRNNCell':
                self.prediction = tf.matmul(self.last_state, self.w) + self.b
            elif RNN == 'BasicLSTMCell':
                self.prediction = tf.matmul(self.last_state[1], self.w) + self.b
                # cell output equals to the hidden state. In case of LSTM, it's the short-term part of the tuple
                # (second element of LSTMStateTuple).
                # <https://stats.stackexchange.com/questions/330176/what-is-the-output-of-a-tf-nn-dynamic-rnn> より

        #########################################################################
        # 評価関数郡
        with tf.name_scope('optimization'):   # tensorboard用
            # 損失関数（平均絶対誤差：MAE）と最適化（Adam）
            self.loss = tf.reduce_mean(tf.square(self.y - self.prediction))
            adam = tf.train.AdamOptimizer(0.01)
            self.optimizer = adam.minimize(self.loss)

            # 精度評価: 誤差(%)の平均
            train_mean_t = dataset.train_mean[self.TARGET_FEATURE]
            train_std_t = dataset.train_std[self.TARGET_FEATURE]
            #self.train_std_t_reshape = tf.reshape(_tile, [batch_size, TARGET_FEATURE_COUNT])
            #self.train_mean_t_reshape = tf.reshape(tf.tile(train_mean_t, [batch_size]), [batch_size, TARGET_FEATURE_COUNT])
            #self.accuracy = tf.reduce_mean(self.train_std_t_reshape)
            #self.accuracy = tf.reduce_mean(tf.divide(self.prediction * train_std_t_reshape + train_mean_t_reshape,
            #                                         self.y          * train_std_t_reshape + train_mean_t_reshape))
            self.accuracy = tf.reduce_mean(tf.Variable([1.0]))

            # 精度評価: 誤差のばらつき(%)
            diff_mean, diff_var = tf.nn.moments(self.y - self.prediction, axes=[0])
            #self.acc_stddev = tf.reduce_mean(tf.divide(tf.multiply(tf.sqrt(diff_var), train_std_t), train_mean_t))  # tf ver1.5
            self.acc_stddev = tf.reduce_mean(tf.Variable([1.0]))
