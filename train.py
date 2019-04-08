#!/usr/bin/python
# -*- coding: Shift_JIS -*-

##
#  �Q�l�ɂ����̂͂���
#  RNN: <https://deepinsider.jp/tutor/introtensorflow/buildrnn>
#  LSTM <https://www.slideshare.net/aitc_jp/20180127-tensorflowrnnlstm>
##

########################################################################
import os
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from datetime import datetime
import Model
import TrainTestDataSet

##########################################################################
# �N���I�v�V�����̃p�[�X
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

# �����V�[�h�̏������i���l�͉��ł��悢�j
np.random.seed(12345)

#############################################################
# �w�K�f�[�^�A�e�X�g�f�[�^�̓ǂݍ��݁A���H�Ȃǂ̏���
#############################################################
# ���̓f�[�^�̂����g���[�j���O�Ɏg���f�[�^�̊����B�c��͕]���p�B
TRAIN_DATA_LENGTH_RATE = 0.9
# �w�K���Ԓ�
SERIES_LENGTH = 72

dataset2 = TrainTestDataSet.TrainTestDataSet(args.cc, train_data_length_rate=TRAIN_DATA_LENGTH_RATE,
                                             series_length=SERIES_LENGTH)
print('train data {} to {}, {} data' .format(dataset2.train_dataset.series_data.index[0],
                                             dataset2.train_dataset.series_data.index[-1],
                                             dataset2.train_dataset.series_length))
print('test data {} to {}, {} data' .format(dataset2.test_dataset.series_data.index[0],
                                            dataset2.test_dataset.series_data.index[-1],
                                            dataset2.test_dataset.series_length))

#######################################################################
# RNN Model�̐���
#######################################################################
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
model = Model.Model(dataset=dataset2, series_length=SERIES_LENGTH, feature_count=FEATURE_COUNT,
              num_of_neuron=NUM_OF_NEURON, rnn=RNN,
              target_feature=TARGET_FEATURE, target_feature_count=TARGET_FEATURE_COUNT)

#######################################################################
# �w�K�̎��s
#######################################################################
# �o�b�`�T�C�Y
BATCH_SIZE = 16    # 64

# �w�K��
NUM_TRAIN = 200   # 10000

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
directory_log = 'logs/' + args.target_feature
if os.path.exists(directory_log):
    shutil.rmtree(directory_log)
os.makedirs(directory_log)

output_log = pd.DataFrame()
z_columns = ['date', 'Iteration', 'N_OF_NEURON', 'BATCH_SIZE', 'loss', 'accuracy', 'stddev']
with tf.summary.FileWriter(directory_log, sess.graph) as writer:
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
            output_log = output_log.append(pd.Series([now, i, NUM_OF_NEURON, BATCH_SIZE, mae, acc, acc2],
                                                     name=i, index=z_columns))
            writer.add_summary(summary, global_step=i)

        # �w�K�̎��s
        _ = sess.run(model.optimizer, feed_dict={model.x: batch[0], model.y: batch[1]})

    # �ŏI���O
    mae = sess.run(model.loss, feed_dict={model.x: batch[0], model.y: batch[1]})
    [acc, acc2] = sess.run([model.accuracy, model.acc_stddev],
                           feed_dict={model.x: test_batch[0], model.y: test_batch[1]})
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print('{:s}: step {:5d}, loss {:.3f}, acc {:.3f}, std {:.3f}' .format(now, NUM_TRAIN, mae, acc, acc2))
    output_log = output_log.append(pd.Series([now, NUM_TRAIN, NUM_OF_NEURON, BATCH_SIZE, mae, acc, acc2],
                                             name=NUM_TRAIN, index=z_columns))
    writer.add_summary(summary, global_step=NUM_TRAIN)

# ���f���̕ۑ�
cwd = os.getcwd()
directory_model = cwd + "/" + "model/" + args.target_feature
os.makedirs(directory_model, exist_ok=True)
if os.name == 'nt':  # for windows
    saver.save(sess, directory_model+"\\model.ckpt")  ## for windows?
else:
    saver.save(sess, directory_model+"/model.ckpt")  ## for linux?

# �w�K�̐��ڂ�csv�ۑ�
output_log = output_log[z_columns]  # ���Ԃ��`�����Ƃ���ɕ��ёւ���B
output_log.set_index('date', inplace=True)
output_log.to_csv(directory_model + "/" + "training.csv")
