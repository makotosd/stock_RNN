#!/usr/bin/python
# -*- coding: Shift_JIS -*-

##
#  参考にしたのはここ
#  RNN: <https://deepinsider.jp/tutor/introtensorflow/buildrnn>
#  LSTM <https://www.slideshare.net/aitc_jp/20180127-tensorflowrnnlstm>
##
#  TODO: 二日先、五日先の予想
#  TODO: 入力パラメータ(会社数)を増やす。
##

########################################################################
import os
import numpy as np
import tensorflow as tf
import TimeSeriesDataSet
import merge_companies
import argparse
from datetime import datetime

# arg パーサの生成
parser = argparse.ArgumentParser(description='予測値と真値の比較、保存、可視化')

# オプション群の設定
parser.add_argument('--cc', nargs='*', help='company code')
# parser.add_argument('--output', help='予測値と真値の結果(csv)の出力。可視化は行わない。')
# parser.add_argument('--input', help='予測値と真値の結果(csv)の入力。予測は行わない。')
parser.add_argument('--feature', nargs='*', help='[open, close, high, low, volume, highopen]',
                    default=['open', 'close', 'high', 'low', 'highopen'])
parser.add_argument('--quote', nargs='*', help='[USD, EUR]', default=[])
parser.add_argument('--target_feature', nargs='*', help='[6702_close, 6702_low], default=[]')
parser.add_argument('--rnn', nargs=1, help='[BasicLSTMCell|BasicRNNCell]', default='BasicRNNCell')

args = parser.parse_args()  # 引数の解析を実行

print('cc: '      + ",".join(args.cc))
print('feature: ' + ",".join(args.feature))
print('quote: '   + ",".join(args.quote))
print('rnn: '     + str(args.rnn))

#############################################################
# 乱数シードの初期化（数値は何でもよい）
np.random.seed(12345)

# 不要列の除去
# target_columns = ['1330_open', '1330_close', '1330_high', '1330_low', '1330_volume', '1330_highopen',
#                   '6701_open', '6701_close', '6701_high', '6701_low', '6701_volume', '6701_highopen',
#                   '6702_open', '6702_close', '6702_high', '6702_low', '6702_volume', '6702_highopen']
target_columns = args.quote
for cc in args.cc:
    for feature in args.feature:
        cc_f = cc + '_' + feature
        target_columns.append(cc_f)
stock_merged_cc = merge_companies.merge_companies(args.cc)
dataset = TimeSeriesDataSet.TimeSeriesDataSet(stock_merged_cc[target_columns])
train_dataset = dataset['2001': '2016']

########################################################################
sess = tf.InteractiveSession()

# パラメーター
# 学習時間長
SERIES_LENGTH = 72
# 特徴量数
FEATURE_COUNT = dataset.feature_count
# ニューロン数
NUM_OF_NEURON = 30
# 最適化対象パラメータ
TARGET_FEATURE = args.target_feature
TARGET_FEATURE_COUNT = len(args.target_feature)
# BasicRNNCell or BasicLSTMCell
RNN = args.rnn[0]

# 入力（placeholderメソッドの引数は、データ型、テンソルのサイズ）
# 訓練データ
x = tf.placeholder(tf.float32, [None, SERIES_LENGTH, FEATURE_COUNT])
# 教師データ
y = tf.placeholder(tf.float32, [None, TARGET_FEATURE_COUNT])

#######################################################################
# list 11
# RNNセルの作成
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
# list 12

# 全結合
# 重み
w = tf.Variable(tf.zeros([NUM_OF_NEURON, TARGET_FEATURE_COUNT]))
# バイアス
b = tf.Variable([0.1] * TARGET_FEATURE_COUNT)
# 最終出力（予測）
if RNN == 'BasicRNNCell':
    # prediction = tf.matmul(last_state, w) + b
    prediction = tf.matmul(last_state, w) + b
elif RNN == 'BasicLSTMCell':
    prediction = tf.matmul(last_state[-1], w) + b

# 損失関数（平均絶対誤差：MAE）と最適化（Adam）
loss = tf.reduce_mean(tf.map_fn(tf.abs, y - prediction))
optimizer = tf.train.AdamOptimizer().minimize(loss)

#######################################################################
# list 14
# バッチサイズ
BATCH_SIZE = 64

# 学習回数
NUM_TRAIN = 10000

# 学習中の出力頻度
OUTPUT_BY = 500

# 標準化
train_mean = train_dataset.mean()
train_std = train_dataset.std()
standardized_train_dataset = train_dataset.standardize()

# 学習の実行
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
for i in range(NUM_TRAIN):
    batch = standardized_train_dataset.next_batch(SERIES_LENGTH, BATCH_SIZE, TARGET_FEATURE)
    mae, _ = sess.run([loss, optimizer], feed_dict={x: batch[0], y: batch[1]})
    if i % OUTPUT_BY == 0:
        now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        print('{:s}: step {:d}, error {:.3f}'.format(now, i, mae))

# 保存
cwd = os.getcwd()
if os.name == 'nt':  # for windows
    saver.save(sess, cwd+"\\model.ckpt")  ## for windows?
else:
    saver.save(sess, cwd+"/model.ckpt")  ## for linux?

