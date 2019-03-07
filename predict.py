#!/usr/bin/python
# -*- coding: Shift_JIS -*-
#
#　学習済みのモデルを読み込む。
#  テストデータを読み込む。
#  サンプル郡を一個ずつずらしながら、予測値のリストを作る。


import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import merge_companies
import TimeSeriesDataSet

######################################################################
# list 15


def rnn_predict(input_dataset):
    # 標準化
    previous = TimeSeriesDataSet.TimeSeriesDataSet(input_dataset).tail(SERIES_LENGTH).standardize(mean=train_mean, std=train_std)
    # 予測対象の時刻
    predict_time = previous.times[-1] + np.timedelta64(1, 'h')  # TODO: 次の行を1時間ごと決め打ちしちゃってる。元データの次の行のindexをもってくる。

    # 予測
    batch_x = previous.as_array()
    predict_data = prediction.eval({x: batch_x})

    # 結果のデータフレームを作成
    df_standardized = pd.DataFrame(predict_data, columns=input_dataset.columns, index=[predict_time])
    # 標準化の逆操作
    return train_mean + train_std * df_standardized

#############################################################
ccs = sys.argv
ccs.pop(0)  # 先頭(script名)を削除
stock_merged_cc = merge_companies.merge_companies(ccs)


# list 7
# 不要列の除去
target_columns = ['1330_open', '1330_close', '6701_open', '6701_close', '6702_open', '6702_close'] # ccがハードに埋まってる。
air_quality = stock_merged_cc[target_columns]


# hogehoge
dataset = TimeSeriesDataSet.TimeSeriesDataSet(air_quality)
train_dataset = dataset['2005': '2006']  # 2005年分をトレーニングデータにする。
test_dataset = dataset['2006': ]

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

# 標準化
train_mean = train_dataset.mean()
train_std = train_dataset.std()

#######################################################################
# list 11
# RNNセルの作成
cell = tf.nn.rnn_cell.BasicRNNCell(20)
initial_state = cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
outputs, last_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)

# 全結合
# 重み
w = tf.Variable(tf.zeros([20, FEATURE_COUNT]))
# バイアス
b = tf.Variable([0.1] * FEATURE_COUNT)
# 最終出力（予測）
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

predict_air_quality = pd.DataFrame([], columns=air_quality.columns)
for current_time in test_dataset.times:
    print(current_time)
    print([air_quality.index])
    print([air_quality.index < current_time])
    predict_result = rnn_predict(air_quality[air_quality.index < current_time])
    predict_air_quality = predict_air_quality.append(predict_result)

print(predict_air_quality)





