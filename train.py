#!/usr/bin/python
# -*- coding: Shift_JIS -*-

##
#  �Q�l�ɂ����̂͂���
#  RNN: <https://deepinsider.jp/tutor/introtensorflow/buildrnn>
#  LSTM <https://www.slideshare.net/aitc_jp/20180127-tensorflowrnnlstm>
##

########################################################################
import os
import numpy as np
import tensorflow as tf
import TimeSeriesDataSet
import merge_companies
import argparse
from datetime import datetime

########################################################################
# RNN ���f���@�N���X
########################################################################
class Model():
    def __init__(self, dataset):

        #######################################################################
        # placeholder
        with tf.name_scope('input'):  # tensorboard�p
            # ���́iplaceholder���\�b�h�̈����́A�f�[�^�^�A�e���\���̃T�C�Y�j
            # �P���f�[�^
            self.x = tf.placeholder(tf.float32, [None, SERIES_LENGTH, FEATURE_COUNT])
            # ���t�f�[�^
            self.y = tf.placeholder(tf.float32, [None, TARGET_FEATURE_COUNT])

        #######################################################################
        # RNN Cell
        with tf.name_scope('RNN'):  # tensorboard�p
            # RNN�Z���̍쐬
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
        # �S����
        with tf.name_scope('prediction'):  # tensorboard�p
            # �d��
            # with tf.name_scope('W'):
            self.w = tf.Variable(tf.zeros([NUM_OF_NEURON, TARGET_FEATURE_COUNT]))
            # �o�C�A�X
            # with tf.name_scope('b'):
            self.b = tf.Variable([0.1] * TARGET_FEATURE_COUNT)
            # �ŏI�o�́i�\���j
            if RNN == 'BasicRNNCell':
                self.prediction = tf.matmul(self.last_state, self.w) + self.b
            elif RNN == 'BasicLSTMCell':
                self.prediction = tf.matmul(self.last_state[1], self.w) + self.b
                # cell output equals to the hidden state. In case of LSTM, it's the short-term part of the tuple
                # (second element of LSTMStateTuple).
                # <https://stats.stackexchange.com/questions/330176/what-is-the-output-of-a-tf-nn-dynamic-rnn> ���

        #########################################################################
        # �]���֐��S
        with tf.name_scope('optimization'):   # tensorboard�p
            # �����֐��i���ϐ�Ό덷�FMAE�j�ƍœK���iAdam�j
            self.loss = tf.reduce_mean(tf.square(self.y - self.prediction))
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

            # ���x�]��: �덷(%)�̕���
            train_mean_t = dataset.train_mean[TARGET_FEATURE]
            train_std_t = dataset.train_std[TARGET_FEATURE]
            self.accuracy = tf.reduce_mean(tf.divide(self.prediction*train_std_t+train_mean_t,
                                                     self.y*train_std_t+train_mean_t))

            # ���x�]��: �덷�̂΂��(%)
            diff_mean, diff_var = tf.nn.moments(self.y - self.prediction, axes=[0])
            # self.acc_stddev = tf.reduce_mean(tf.math.sqrt(diff_var) * train_std_t / train_mean_t)  # tf ver1.12
            self.acc_stddev = tf.reduce_mean(tf.sqrt(diff_var) * train_std_t / train_mean_t)  # tf ver1.5

##########################################################################
# �w�K�f�[�^�ƃe�X�g�f�[�^�̓ǂݍ��݁A�����B
##########################################################################
class TrainTestDataSet():
    def __init__(self, args):
        stock_merged_cc = merge_companies.merge_companies(args.cc)
        dataset = TimeSeriesDataSet.TimeSeriesDataSet(stock_merged_cc)
        self.feature_count = dataset.feature_count
        self.train_dataset, self.test_dataset = dataset.divide_dataset(rate=TRAIN_DATA_LENGTH_RATE, series_length=SERIES_LENGTH)

        # �W����
        self.train_mean = self.train_dataset.mean()
        self.train_std = self.train_dataset.std()
        self.standardized_train_dataset = self.train_dataset.standardize()
        self.standardized_test_dataset = self.test_dataset.standardize(mean=self.train_mean, std=self.train_std)

##########################################################################
# arg �p�[�T�̐���
parser = argparse.ArgumentParser(description='�\���l�Ɛ^�l�̔�r�A�ۑ��A����')

# �I�v�V�����Q�̐ݒ�
parser.add_argument('--cc', nargs='*', help='company code')
# parser.add_argument('--feature', nargs='*', help='[open, close, high, low, volume, highopen]',
#                    default=['open', 'close', 'high', 'low', 'highopen'])
# parser.add_argument('--quote', nargs='*', help='[USD, EUR]', default=[])
parser.add_argument('--target_feature', help='6702_close', default='')
parser.add_argument('--rnn', help='[BasicLSTMCell|BasicRNNCell]', default='BasicRNNCell')

args = parser.parse_args()  # �����̉�͂����s

print('cc: '      + ",".join(args.cc))
# print('feature: ' + ",".join(args.feature))
# print('quote: '   + ",".join(args.quote))
print('rnn: '     + str(args.rnn))

#############################################################
# ���̓f�[�^�̂����g���[�j���O�Ɏg���f�[�^�̊����B�c��͕]���p�B
TRAIN_DATA_LENGTH_RATE = 0.9
# �w�K���Ԓ�
SERIES_LENGTH = 72
#############################################################
# �����V�[�h�̏������i���l�͉��ł��悢�j
np.random.seed(12345)

#############################################################
#
#############################################################
dataset2 = TrainTestDataSet(args)
print('train data {} to {}, {} data' .format(dataset2.train_dataset.series_data.index[0],
                                             dataset2.train_dataset.series_data.index[-1],
                                             dataset2.train_dataset.series_length))
print('test data {} to {}, {} data' .format(dataset2.test_dataset.series_data.index[0],
                                            dataset2.test_dataset.series_data.index[-1],
                                            dataset2.test_dataset.series_length))

#############################################################
# �p�����[�^�[
# �����ʐ�
FEATURE_COUNT = dataset2.feature_count
# �j���[������
NUM_OF_NEURON = 60
# �œK���Ώۃp�����[�^
TARGET_FEATURE = args.target_feature
TARGET_FEATURE_COUNT = 1
# BasicRNNCell or BasicLSTMCell
RNN = args.rnn  # rnn[0]

#######################################################################
#
#######################################################################
model = Model(dataset2)

#######################################################################
# �o�b�`�T�C�Y
BATCH_SIZE = 16    # 64

# �w�K��
NUM_TRAIN = 1000   # 10000

# �w�K���̏o�͕p�x
OUTPUT_BY = 50     # 500

# �����֐��̏o�͂��g���[�X�ΏۂƂ���
tf.summary.scalar('MAE', model.loss)
tf.summary.scalar('ACC', model.accuracy)
tf.summary.scalar('STD', model.acc_stddev)
tf.summary.histogram('weight', model.w)
tf.summary.histogram('bias', model.b)
merged_log = tf.summary.merge_all()

########################################################################
# logs�f�B���N�g���ɏo�͂��郉�C�^�[���쐬���ė��p
sess = tf.InteractiveSession()
print('session initialize')
with tf.summary.FileWriter('logs', sess.graph) as writer:
    # �w�K�̎��s
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # test�p�f�[�^�Z�b�g�̃o�b�`���s�p�f�[�^�̍쐬
    test_batch = dataset2.standardized_test_dataset.test_batch(SERIES_LENGTH, TARGET_FEATURE)

    # �o�b�`�w�K
    for i in range(NUM_TRAIN):
        # �w�K�f�[�^�쐬
        batch = dataset2.standardized_train_dataset.next_batch(SERIES_LENGTH, BATCH_SIZE, TARGET_FEATURE)

        # ���O�o��
        if i % OUTPUT_BY == 0:
            # ���O�̏o��
            mae = sess.run(model.loss, feed_dict={model.x: batch[0], model.y: batch[1]})
            [summary, acc, acc2] = sess.run([merged_log, model.accuracy, model.acc_stddev],
                                            feed_dict={model.x: test_batch[0], model.y: test_batch[1]})
            now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            print('{:s}: step {:5d}, loss {:.3f}, acc {:.3f}, std {:.3f}'.format(now, i, mae, acc, acc2))
            writer.add_summary(summary, global_step=i)

        # �w�K�̎��s
        _ = sess.run(model.optimizer, feed_dict={model.x: batch[0], model.y: batch[1]})

    # �ŏI���O
    mae = sess.run(model.loss, feed_dict={model.x: batch[0], model.y: batch[1]})
    [acc, acc2] = sess.run([model.accuracy, model.acc_stddev],
                           feed_dict={model.x: test_batch[0], model.y: test_batch[1]})
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print('{:s}: step {:5d}, loss {:.3f}, acc {:.3f}, std {:.3f}' .format(now, NUM_TRAIN, mae, acc, acc2))
    writer.add_summary(summary, global_step=NUM_TRAIN)

# �ۑ�
cwd = os.getcwd()
if os.name == 'nt':  # for windows
    saver.save(sess, cwd+"\\model.ckpt")  ## for windows?
else:
    saver.save(sess, cwd+"/model.ckpt")  ## for linux?

