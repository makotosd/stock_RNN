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

# �s�v��̏���
stock_merged_cc = merge_companies.merge_companies(args.cc)
dataset = TimeSeriesDataSet.TimeSeriesDataSet(stock_merged_cc)
train_dataset, test_dataset = dataset.divide_dataset(rate=TRAIN_DATA_LENGTH_RATE, series_length=SERIES_LENGTH)
print('train data {} to {}, {} data' .format(train_dataset.series_data.index[0],
                                             train_dataset.series_data.index[-1],
                                             train_dataset.series_length))
print('test data {} to {}, {} data' .format(test_dataset.series_data.index[0],
                                            test_dataset.series_data.index[-1],
                                            test_dataset.series_length))

#############################################################
# �p�����[�^�[
# �����ʐ�
FEATURE_COUNT = dataset.feature_count
# �j���[������
NUM_OF_NEURON = 60
# �œK���Ώۃp�����[�^
TARGET_FEATURE = args.target_feature
TARGET_FEATURE_COUNT = 1
# BasicRNNCell or BasicLSTMCell
RNN = args.rnn  # rnn[0]

if 1:
    # �W����
    train_mean = train_dataset.mean()
    train_std = train_dataset.std()
    standardized_train_dataset = train_dataset.standardize()
    standardized_test_dataset = test_dataset.standardize(mean=train_mean, std=train_std)


with tf.name_scope('input'):   # tensorboard�p
    # ���́iplaceholder���\�b�h�̈����́A�f�[�^�^�A�e���\���̃T�C�Y�j
    # �P���f�[�^
    x = tf.placeholder(tf.float32, [None, SERIES_LENGTH, FEATURE_COUNT])
    # ���t�f�[�^
    y = tf.placeholder(tf.float32, [None, TARGET_FEATURE_COUNT])

#######################################################################
with tf.name_scope('RNN'):   # tensorboard�p
    # RNN�Z���̍쐬
    if RNN == 'BasicRNNCell':
        print('BasicRNNCell')
        cell = tf.nn.rnn_cell.BasicRNNCell(NUM_OF_NEURON)
        initial_state = cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
        outputs, last_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)
    elif RNN == 'BasicLSTMCell':
        print('BasicLSTMCell')
        cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_OF_NEURON)
        initial_state = cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
        outputs, last_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)
    else:
        print('No RNN Cell Defined')
        exit(0)

#######################################################################
# �S����
with tf.name_scope('prediction'):   # tensorboard�p
    # �d��
    #with tf.name_scope('W'):
    w = tf.Variable(tf.zeros([NUM_OF_NEURON, TARGET_FEATURE_COUNT]))
    # �o�C�A�X
    #with tf.name_scope('b'):
    b = tf.Variable([0.1] * TARGET_FEATURE_COUNT)
    # �ŏI�o�́i�\���j
    if RNN == 'BasicRNNCell':
        prediction = tf.matmul(last_state, w) + b
    elif RNN == 'BasicLSTMCell':
        prediction = tf.matmul(last_state[1], w) + b
        # cell output equals to the hidden state. In case of LSTM, it's the short-term part of the tuple
        # (second element of LSTMStateTuple).
        # <https://stats.stackexchange.com/questions/330176/what-is-the-output-of-a-tf-nn-dynamic-rnn> ���

with tf.name_scope('optimization'):   # tensorboard�p
    # �����֐��i���ϐ�Ό덷�FMAE�j�ƍœK���iAdam�j
    loss = tf.reduce_mean(tf.square(y - prediction))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # ���x�]��: �덷(%)�̕���
    train_mean_t = train_mean[TARGET_FEATURE]
    train_std_t = train_std[TARGET_FEATURE]
    accuracy = tf.reduce_mean(tf.divide(prediction*train_std_t+train_mean_t, y*train_std_t+train_mean_t))

    # ���x�]��: �덷�̂΂��(%)
    diff_mean, diff_var = tf.nn.moments(y - prediction, axes=[0])
    acc_stddev = tf.reduce_mean(tf.math.sqrt(diff_var) * train_std_t / train_mean_t)

#######################################################################
# �o�b�`�T�C�Y
BATCH_SIZE = 16    # 64

# �w�K��
NUM_TRAIN = 1000   # 10000

# �w�K���̏o�͕p�x
OUTPUT_BY = 50     # 500

# �����֐��̏o�͂��g���[�X�ΏۂƂ���
tf.summary.scalar('MAE', loss)
tf.summary.scalar('ACC', accuracy)
tf.summary.scalar('STD', acc_stddev)
tf.summary.histogram('weight', w)
tf.summary.histogram('bias', b)
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
    test_batch = standardized_test_dataset.test_batch(SERIES_LENGTH, TARGET_FEATURE)

    # �o�b�`�w�K
    for i in range(NUM_TRAIN):
        # �w�K�f�[�^�쐬
        batch = standardized_train_dataset.next_batch(SERIES_LENGTH, BATCH_SIZE, TARGET_FEATURE)

        # ���O�o��
        if i % OUTPUT_BY == 0:
            # ���O�̏o��
            mae = sess.run(loss, feed_dict={x: batch[0], y: batch[1]})
            [summary, acc, acc2] = sess.run([merged_log, accuracy, acc_stddev], feed_dict={x: test_batch[0], y: test_batch[1]})
            now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            print('{:s}: step {:5d}, loss {:.3f}, acc {:.3f}, std {:.3f}'.format(now, i, mae, acc, acc2))
            writer.add_summary(summary, global_step=i)

        # �w�K�̎��s
        _ = sess.run(optimizer, feed_dict={x: batch[0], y: batch[1]})

    # �ŏI���O
    mae = sess.run(loss, feed_dict={x: batch[0], y: batch[1]})
    [acc, acc2] = sess.run([accuracy, acc_stddev], feed_dict={x: test_batch[0], y: test_batch[1]})
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print('{:s}: step {:5d}, loss {:.3f}, acc {:.3f}, std {:.3f}' .format(now, NUM_TRAIN, mae, acc, acc2))
    writer.add_summary(summary, global_step=NUM_TRAIN)

# �ۑ�
cwd = os.getcwd()
if os.name == 'nt':  # for windows
    saver.save(sess, cwd+"\\model.ckpt")  ## for windows?
else:
    saver.save(sess, cwd+"/model.ckpt")  ## for linux?

