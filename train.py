#!/usr/bin/python
# -*- coding: Shift_JIS -*-

##
#  TODO: �����A�ܓ���̗\�z
#  TODO: ���̓p�����[�^(��А�)�𑝂₷�B
##

########################################################################
import os
import sys
import numpy as np
import tensorflow as tf
import TimeSeriesDataSet
import merge_companies

ccs = sys.argv
ccs.pop(0)  # �擪(script��)���폜
stock_merged_cc = merge_companies.merge_companies(ccs)


#############################################################
# list 7
# �s�v��̏���
#target_columns = ['1330_open', '1330_close', '6701_open', '6701_close', '6702_open', '6702_close'] # cc���n�[�h�ɖ��܂��Ă�B
target_columns = ['1330_open', '1330_close', '1330_high', '1330_low',
                  '6701_open', '6701_close', '6701_high', '6701_low',
                  '6702_open', '6702_close', '6702_high', '6702_low']  # cc���n�[�h�ɖ��܂��Ă�B
air_quality = stock_merged_cc[target_columns]

#######################################################################################
# list 8 + list 13


# �����V�[�h�̏������i���l�͉��ł��悢�j
np.random.seed(12345)

dataset = TimeSeriesDataSet.TimeSeriesDataSet(air_quality)
train_dataset = dataset['2001': '2007']  # 2001-2007�N�����g���[�j���O�f�[�^�ɂ���B


########################################################################
# list 10
sess = tf.InteractiveSession()

# �p�����[�^�[
# �w�K���Ԓ�
SERIES_LENGTH = 72
# �����ʐ�
FEATURE_COUNT = dataset.feature_count

# ���́iplaceholder���\�b�h�̈����́A�f�[�^�^�A�e���\���̃T�C�Y�j
# �P���f�[�^
x = tf.placeholder(tf.float32, [None, SERIES_LENGTH, FEATURE_COUNT])
# ���t�f�[�^
y = tf.placeholder(tf.float32, [None, FEATURE_COUNT])

#######################################################################
# list 11
# RNN�Z���̍쐬
cell = tf.nn.rnn_cell.BasicRNNCell(20)
initial_state = cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
outputs, last_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)

#######################################################################
# list 12

# �S����
# �d��
w = tf.Variable(tf.zeros([20, FEATURE_COUNT]))
# �o�C�A�X
b = tf.Variable([0.1] * FEATURE_COUNT)
# �ŏI�o�́i�\���j
prediction = tf.matmul(last_state, w) + b

# �����֐��i���ϐ�Ό덷�FMAE�j�ƍœK���iAdam�j
loss = tf.reduce_mean(tf.map_fn(tf.abs, y - prediction))
optimizer = tf.train.AdamOptimizer().minimize(loss)

#######################################################################
# list 14
# �o�b�`�T�C�Y
BATCH_SIZE = 16

# �w�K��
NUM_TRAIN = 10000

# �w�K���̏o�͕p�x
OUTPUT_BY = 500

# �W����
train_mean = train_dataset.mean()
train_std = train_dataset.std()
standardized_train_dataset = train_dataset.standardize()

# �w�K�̎��s
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
for i in range(NUM_TRAIN):
    batch = standardized_train_dataset.next_batch(SERIES_LENGTH, BATCH_SIZE)
    mae, _ = sess.run([loss, optimizer], feed_dict={x: batch[0], y: batch[1]})
    if i % OUTPUT_BY == 0:
        print('step {:d}, error {:.2f}'.format(i, mae))

# �ۑ�
cwd = os.getcwd()
if os.name == 'nt':  # for windows
    saver.save(sess, cwd+"\\model.ckpt")  ## for windows?
else:
    saver.save(sess, cwd+"/model.ckpt")  ## for linux?

