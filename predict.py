#!/usr/bin/python
# -*- coding: Shift_JIS -*-
#
#�@�w�K�ς݂̃��f����ǂݍ��ށB
#  �e�X�g�f�[�^��ǂݍ��ށB
#  �T���v���S��������炵�Ȃ���A�\���l�̃��X�g�����B


import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import merge_companies
import TimeSeriesDataSet

######################################################################
# list 15


def rnn_predict(input_dataset, current_time):
    # �W����
    previous = input_dataset.tail(SERIES_LENGTH).standardize(mean=train_mean, std=train_std)
    # �\���Ώۂ̎���
    predict_time = current_time  # previous.times[-1] + np.timedelta64(1, 'h')  # TODO: ���̍s��1���Ԃ��ƌ��ߑł���������Ă�B���f�[�^�̎��̍s��index�������Ă���B

    # �\��
    batch_x = previous.as_array()
    predict_data = prediction.eval({x: batch_x})

    # ���ʂ̃f�[�^�t���[�����쐬
    df_standardized = pd.DataFrame(predict_data, columns=input_dataset.series_data.columns, index=[predict_time])
    # �W�����̋t����
    return train_mean + train_std * df_standardized

#############################################################
ccs = sys.argv
ccs.pop(0)  # �擪(script��)���폜
stock_merged_cc = merge_companies.merge_companies(ccs)


# list 7
# �s�v��̏���
target_columns = ['1330_open', '1330_close', '6701_open', '6701_close', '6702_open', '6702_close'] # cc���n�[�h�ɖ��܂��Ă�B
dataset = TimeSeriesDataSet.TimeSeriesDataSet(stock_merged_cc[target_columns])
train_dataset = dataset['2001': '2007']  # 2005�N�����g���[�j���O�f�[�^�ɂ���B
test_dataset = dataset['2007': ]

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

# �W����
train_mean = train_dataset.mean()
train_std = train_dataset.std()

#######################################################################
# list 11
# RNN�Z���̍쐬
cell = tf.nn.rnn_cell.BasicRNNCell(20)
initial_state = cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
outputs, last_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)

# �S����
# �d��
w = tf.Variable(tf.zeros([20, FEATURE_COUNT]))
# �o�C�A�X
b = tf.Variable([0.1] * FEATURE_COUNT)
# �ŏI�o�́i�\���j
prediction = tf.matmul(last_state, w) + b

##
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# restore
saver = tf.train.Saver()
cwd = os.getcwd()
if os.name == 'nt':
    saver.restore(sess, cwd + "\\model.ckpt")
else:
    saver.restore(sess, cwd + "/model.ckpt")

predict_dataset = pd.DataFrame([], columns=dataset.series_data.columns)
for current_time in test_dataset.times:
    predict_result = rnn_predict(dataset[dataset.series_data.index < current_time], current_time)
    predict_dataset = predict_dataset.append(predict_result)

print(predict_dataset)





