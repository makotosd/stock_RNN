#!/usr/bin/python
# -*- coding: Shift_JIS -*-

##
#  TODO: 二日先、五日先の予想
#  TODO: 入力パラメータ(会社数)を増やす。
##

########################################################################
import os
import sys
import numpy as np
import tensorflow as tf
import TimeSeriesDataSet
import merge_companies

ccs = sys.argv
ccs.pop(0)  # 先頭(script名)を削除
stock_merged_cc = merge_companies.merge_companies(ccs)


#############################################################
# list 7
# 不要列の除去
#target_columns = ['1330_open', '1330_close', '6701_open', '6701_close', '6702_open', '6702_close'] # ccがハードに埋まってる。
target_columns = ['1330_open', '1330_close', '1330_high', '1330_low',
                  '6701_open', '6701_close', '6701_high', '6701_low',
                  '6702_open', '6702_close', '6702_high', '6702_low']  # ccがハードに埋まってる。
air_quality = stock_merged_cc[target_columns]

#######################################################################################
# list 8 + list 13


# 乱数シードの初期化（数値は何でもよい）
np.random.seed(12345)

dataset = TimeSeriesDataSet.TimeSeriesDataSet(air_quality)
train_dataset = dataset['2001': '2007']  # 2001-2007年分をトレーニングデータにする。


########################################################################
# list 10
sess = tf.InteractiveSession()

# パラメーター
# 学習時間長
SERIES_LENGTH = 72
# 特徴量数
FEATURE_COUNT = dataset.feature_count

# 入力（placeholderメソッドの引数は、データ型、テンソルのサイズ）
# 訓練データ
x = tf.placeholder(tf.float32, [None, SERIES_LENGTH, FEATURE_COUNT])
# 教師データ
y = tf.placeholder(tf.float32, [None, FEATURE_COUNT])

#######################################################################
# list 11
# RNNセルの作成
cell = tf.nn.rnn_cell.BasicRNNCell(20)
initial_state = cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
outputs, last_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)

#######################################################################
# list 12

# 全結合
# 重み
w = tf.Variable(tf.zeros([20, FEATURE_COUNT]))
# バイアス
b = tf.Variable([0.1] * FEATURE_COUNT)
# 最終出力（予測）
prediction = tf.matmul(last_state, w) + b

# 損失関数（平均絶対誤差：MAE）と最適化（Adam）
loss = tf.reduce_mean(tf.map_fn(tf.abs, y - prediction))
optimizer = tf.train.AdamOptimizer().minimize(loss)

#######################################################################
# list 14
# バッチサイズ
BATCH_SIZE = 16

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
    batch = standardized_train_dataset.next_batch(SERIES_LENGTH, BATCH_SIZE)
    mae, _ = sess.run([loss, optimizer], feed_dict={x: batch[0], y: batch[1]})
    if i % OUTPUT_BY == 0:
        print('step {:d}, error {:.2f}'.format(i, mae))

# 保存
cwd = os.getcwd()
if os.name == 'nt':  # for windows
    saver.save(sess, cwd+"\\model.ckpt")  ## for windows?
else:
    saver.save(sess, cwd+"/model.ckpt")  ## for linux?

