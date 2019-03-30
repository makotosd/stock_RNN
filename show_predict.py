#!/usr/bin/python
# -*- coding: Shift_JIS -*-

#######################################################################
#  jupyter notebook上で動かす。真の値と、予測値をグラフ化します。
#######################################################################
import os
import pandas as pd
import cufflinks as cf
import argparse
import tensorflow as tf
import merge_companies
import TimeSeriesDataSet


def rnn_predict(input_dataset, current_time, train_mean, train_std, prediction, x, sess):
    # 標準化
    previous = input_dataset.tail(SERIES_LENGTH).standardize(mean=train_mean, std=train_std)
    # 予測対象の時刻
    predict_time = current_time

    # 予測
    batch_x = previous.as_array()
    predict_data = prediction.eval({x: batch_x}, session=sess)

    # 結果のデータフレームを作成
    df_standardized = pd.DataFrame(predict_data, columns=TARGET_FEATURE, index=[predict_time])
    # 標準化の逆操作
    return train_mean[TARGET_FEATURE] + train_std[TARGET_FEATURE] * df_standardized


#############################################################
def predict(stock_merged_cc, company_codes, features, quote, target_feature):
    # 不要列の除去
    # target_columns = quote
    # for cc in company_codes:
    #     for feature in features:
    #         cc_f = cc + '_' + feature
    #         target_columns.append(cc_f)
    # dataset = TimeSeriesDataSet.TimeSeriesDataSet(stock_merged_cc[target_columns])
    dataset = TimeSeriesDataSet.TimeSeriesDataSet(stock_merged_cc)
    train_dataset = dataset['2001': '2016']
    test_dataset = dataset['2017': ]

    # パラメーター
    # 学習時間長
    global SERIES_LENGTH
    SERIES_LENGTH = 72
    # 特徴量数
    global FEATURE_COUNT
    FEATURE_COUNT = dataset.feature_count
    # ニューロン数
    global NUM_OF_NEURON
    NUM_OF_NEURON = 30
    # 最適化対象
    global TARGET_FEATURE
    global TARGET_FEATURE_COUNT
    TARGET_FEATURE = target_feature
    TARGET_FEATURE_COUNT = len(TARGET_FEATURE)

    with tf.name_scope('input'):  # tensorboard用
        # 入力（placeholderメソッドの引数は、データ型、テンソルのサイズ）
        # 訓練データ
        x = tf.placeholder(tf.float32, [None, SERIES_LENGTH, FEATURE_COUNT])
        # 教師データ
        y = tf.placeholder(tf.float32, [None, TARGET_FEATURE_COUNT])

    # 標準化
    train_mean = train_dataset.mean()
    train_std = train_dataset.std()

    #######################################################################
    # RNNセルの作成
    print(RNN)
    with tf.name_scope('RNN'):  # tensorboard用
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
            print('No RNN Cell')
            exit(0)

    with tf.name_scope('prediction'):  # tensorboard用
        # 全結合
        # 重み
        w = tf.Variable(tf.zeros([NUM_OF_NEURON, TARGET_FEATURE_COUNT]))
        # バイアス
        b = tf.Variable([0.1] * TARGET_FEATURE_COUNT)
        # 最終出力（予測）
        if RNN == 'BasicRNNCell':
            prediction = tf.matmul(last_state, w) + b
        elif RNN == 'BasicLSTMCell':
            prediction = tf.matmul(last_state[1], w) + b

    ##
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # restore
    if os.name == 'nt':
        ckpt = tf.train.get_checkpoint_state('./')
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        saver.restore(sess, cwd + "/model.ckpt")

    predict_dataset = pd.DataFrame([], columns=TARGET_FEATURE)
    for current_time in test_dataset.times:
        predict_result = rnn_predict(dataset[dataset.series_data.index < current_time],
                                     current_time,
                                     train_mean, train_std,
                                     prediction,
                                     x,
                                     sess)

        predict_dataset = predict_dataset.append(predict_result)

    return predict_dataset


###########################################################
#   main
###########################################################
global RNN
RNN = ""
if __name__ == "__main__":

    # arg パーサの生成
    parser = argparse.ArgumentParser(description='予測値と真値の比較、保存、可視化')

    # オプション群の設定
    parser.add_argument('--cc', nargs='*', help='company code')
    parser.add_argument('--output', help='予測値と真値の結果(csv)の出力。可視化は行わない。')
    parser.add_argument('--input', help='予測値と真値の結果(csv)の入力。予測は行わない。')
    parser.add_argument('--feature', nargs='*', help='[open, close, high, low, volume]',
                        default=['open', 'close', 'high', 'low', 'volume'])
    parser.add_argument('--quote', nargs='*', help='[USD, EUR]', default=[])
    parser.add_argument('--target_feature', nargs='*', help='[6702_close, 6702_low], default=[]')
    parser.add_argument('--rnn', nargs=1, help='[BasicLSTMCell|BasicRNNCell]', default='BasicRNNCell')

    args = parser.parse_args()  # 引数の解析を実行

    # グローバル変数
    RNN = args.rnn[0]

    # 真値の読み込み
    input_dataset = merge_companies.merge_companies(args.cc)

    #  予測を実行
    if args.input is None:  # 読み込みファイルの指定がない　＝　予測の実施
        predict_dataset = predict(input_dataset, args.cc, args.feature, args.quote, args.target_feature)

        if args.output is not None:  # 書き込みファイルの指定がある ＝ ファイルを書く
            predict_dataset.to_csv(str(args.output))

    else:
        predict_dataset = pd.read_csv(args.input, index_col=0)

    # 正解データと予測データ
    correct_data = input_dataset[input_dataset.index >= '2001']
    predict_data = predict_dataset

    cf.go_offline()  # plotlyをオフラインで実行。
    for feature in predict_dataset.columns:
        plot_data = pd.DataFrame({
            'correct': correct_data[feature],
            'predicted': predict_data[feature]
        }).iplot(
          asFigure = True,
          title = feature
        )
        plot_data['layout']['paper_bgcolor'] = '#FFFFFF'
        plot_data.iplot()
