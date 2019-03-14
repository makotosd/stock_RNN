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


def rnn_predict(input_dataset, current_time, train_mean, train_std, prediction, x):
    # 標準化
    previous = input_dataset.tail(SERIES_LENGTH).standardize(mean=train_mean, std=train_std)
    # 予測対象の時刻
    predict_time = current_time  # previous.times[-1] + np.timedelta64(1, 'h')  # TODO: 次の行を1時間ごと決め打ちしちゃってる。元データの次の行のindexをもってくる。

    # 予測
    batch_x = previous.as_array()
    predict_data = prediction.eval({x: batch_x})

    # 結果のデータフレームを作成
    df_standardized = pd.DataFrame(predict_data, columns=input_dataset.series_data.columns, index=[predict_time])
    # 標準化の逆操作
    return train_mean + train_std * df_standardized


#############################################################
def predict(stock_merged_cc):
    # list 7
    # 不要列の除去
    # target_columns = ['1330_open', '1330_close', '6701_open', '6701_close', '6702_open', '6702_close'] # ccがハードに埋まってる。
    target_columns = ['1330_open', '1330_close', '1330_high', '1330_low',
                      '6701_open', '6701_close', '6701_high', '6701_low',
                      '6702_open', '6702_close', '6702_high', '6702_low']  # ccがハードに埋まってる。
    dataset = TimeSeriesDataSet.TimeSeriesDataSet(stock_merged_cc[target_columns])
    train_dataset = dataset['2001': '2007']  # 2005年分をトレーニングデータにする。
    test_dataset = dataset['2007': ]

    # パラメーター
    # 学習時間長
    global SERIES_LENGTH
    SERIES_LENGTH = 72
    # 特徴量数
    global FEATURE_COUNT
    FEATURE_COUNT = dataset.feature_count

    # ニューロン数
    global NUM_OF_NEURON
    NUM_OF_NEURON = 60

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
    cell = tf.nn.rnn_cell.BasicRNNCell(NUM_OF_NEURON)
    initial_state = cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
    outputs, last_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)

    # 全結合
    # 重み
    w = tf.Variable(tf.zeros([NUM_OF_NEURON, FEATURE_COUNT]))
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

    predict_dataset = pd.DataFrame([], columns=dataset.series_data.columns)
    for current_time in test_dataset.times:
        predict_result = rnn_predict(dataset[dataset.series_data.index < current_time],
                                     current_time,
                                     train_mean, train_std,
                                     prediction,
                                     x)

        predict_dataset = predict_dataset.append(predict_result)

    return predict_dataset


###########################################################
#   main
#
###########################################################
if __name__ == "__main__":
    ccs = sys.argv
    ccs.pop(0)  # 先頭(script名)を削除
    input_dataset = merge_companies.merge_companies(ccs)
    predict_dataset = predict(input_dataset)
    print(predict_dataset)
