#!/usr/bin/python
# -*- coding: Shift_JIS -*-

##
#  参考にしたのはここ
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
# RNN モデル　クラス
########################################################################
class Model():
    def __init__(self, dataset):

        #######################################################################
        # placeholder
        with tf.name_scope('input'):  # tensorboard用
            # 入力（placeholderメソッドの引数は、データ型、テンソルのサイズ）
            # 訓練データ
            self.x = tf.placeholder(tf.float32, [None, SERIES_LENGTH, FEATURE_COUNT])
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
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

            # 精度評価: 誤差(%)の平均
            train_mean_t = dataset.train_mean[TARGET_FEATURE]
            train_std_t = dataset.train_std[TARGET_FEATURE]
            self.accuracy = tf.reduce_mean(tf.divide(self.prediction*train_std_t+train_mean_t,
                                                     self.y*train_std_t+train_mean_t))

            # 精度評価: 誤差のばらつき(%)
            diff_mean, diff_var = tf.nn.moments(self.y - self.prediction, axes=[0])
            # self.acc_stddev = tf.reduce_mean(tf.math.sqrt(diff_var) * train_std_t / train_mean_t)  # tf ver1.12
            self.acc_stddev = tf.reduce_mean(tf.sqrt(diff_var) * train_std_t / train_mean_t)  # tf ver1.5

##########################################################################
# 学習データとテストデータの読み込み、準備。
##########################################################################
class TrainTestDataSet():
    def __init__(self, args):
        stock_merged_cc = merge_companies.merge_companies(args.cc)
        dataset = TimeSeriesDataSet.TimeSeriesDataSet(stock_merged_cc)
        self.feature_count = dataset.feature_count
        self.train_dataset, self.test_dataset = dataset.divide_dataset(rate=TRAIN_DATA_LENGTH_RATE, series_length=SERIES_LENGTH)

        # 標準化
        self.train_mean = self.train_dataset.mean()
        self.train_std = self.train_dataset.std()
        self.standardized_train_dataset = self.train_dataset.standardize()
        self.standardized_test_dataset = self.test_dataset.standardize(mean=self.train_mean, std=self.train_std)

##########################################################################
# arg パーサの生成
parser = argparse.ArgumentParser(description='予測値と真値の比較、保存、可視化')

# オプション群の設定
parser.add_argument('--cc', nargs='*', help='company code')
# parser.add_argument('--feature', nargs='*', help='[open, close, high, low, volume, highopen]',
#                    default=['open', 'close', 'high', 'low', 'highopen'])
# parser.add_argument('--quote', nargs='*', help='[USD, EUR]', default=[])
parser.add_argument('--target_feature', help='6702_close', default='')
parser.add_argument('--rnn', help='[BasicLSTMCell|BasicRNNCell]', default='BasicRNNCell')

args = parser.parse_args()  # 引数の解析を実行

print('cc: '      + ",".join(args.cc))
# print('feature: ' + ",".join(args.feature))
# print('quote: '   + ",".join(args.quote))
print('rnn: '     + str(args.rnn))

#############################################################
# 入力データのうちトレーニングに使うデータの割合。残りは評価用。
TRAIN_DATA_LENGTH_RATE = 0.9
# 学習時間長
SERIES_LENGTH = 72
#############################################################
# 乱数シードの初期化（数値は何でもよい）
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
# パラメーター
# 特徴量数
FEATURE_COUNT = dataset2.feature_count
# ニューロン数
NUM_OF_NEURON = 60
# 最適化対象パラメータ
TARGET_FEATURE = args.target_feature
TARGET_FEATURE_COUNT = 1
# BasicRNNCell or BasicLSTMCell
RNN = args.rnn  # rnn[0]

#######################################################################
#
#######################################################################
model = Model(dataset2)

#######################################################################
# バッチサイズ
BATCH_SIZE = 16    # 64

# 学習回数
NUM_TRAIN = 1000   # 10000

# 学習中の出力頻度
OUTPUT_BY = 50     # 500

# 損失関数の出力をトレース対象とする
tf.summary.scalar('MAE', model.loss)
tf.summary.scalar('ACC', model.accuracy)
tf.summary.scalar('STD', model.acc_stddev)
tf.summary.histogram('weight', model.w)
tf.summary.histogram('bias', model.b)
merged_log = tf.summary.merge_all()

########################################################################
# logsディレクトリに出力するライターを作成して利用
sess = tf.InteractiveSession()
print('session initialize')
with tf.summary.FileWriter('logs', sess.graph) as writer:
    # 学習の実行
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # test用データセットのバッチ実行用データの作成
    test_batch = dataset2.standardized_test_dataset.test_batch(SERIES_LENGTH, TARGET_FEATURE)

    # バッチ学習
    for i in range(NUM_TRAIN):
        # 学習データ作成
        batch = dataset2.standardized_train_dataset.next_batch(SERIES_LENGTH, BATCH_SIZE, TARGET_FEATURE)

        # ログ出力
        if i % OUTPUT_BY == 0:
            # ログの出力
            mae = sess.run(model.loss, feed_dict={model.x: batch[0], model.y: batch[1]})
            [summary, acc, acc2] = sess.run([merged_log, model.accuracy, model.acc_stddev],
                                            feed_dict={model.x: test_batch[0], model.y: test_batch[1]})
            now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            print('{:s}: step {:5d}, loss {:.3f}, acc {:.3f}, std {:.3f}'.format(now, i, mae, acc, acc2))
            writer.add_summary(summary, global_step=i)

        # 学習の実行
        _ = sess.run(model.optimizer, feed_dict={model.x: batch[0], model.y: batch[1]})

    # 最終ログ
    mae = sess.run(model.loss, feed_dict={model.x: batch[0], model.y: batch[1]})
    [acc, acc2] = sess.run([model.accuracy, model.acc_stddev],
                           feed_dict={model.x: test_batch[0], model.y: test_batch[1]})
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print('{:s}: step {:5d}, loss {:.3f}, acc {:.3f}, std {:.3f}' .format(now, NUM_TRAIN, mae, acc, acc2))
    writer.add_summary(summary, global_step=NUM_TRAIN)

# 保存
cwd = os.getcwd()
if os.name == 'nt':  # for windows
    saver.save(sess, cwd+"\\model.ckpt")  ## for windows?
else:
    saver.save(sess, cwd+"/model.ckpt")  ## for linux?

