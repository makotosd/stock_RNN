#!/usr/bin/python
# -*- coding: Shift_JIS -*-

#######################################################################
import pandas as pd
import cufflinks as cf
import argparse
import tensorflow as tf
import Model
import TrainTestDataSet
import os.path
import datetime
import pandas.io.sql as psql
import mysql.connector
from urllib.parse import urlparse
import numpy as np

#############################################################
# Predict_Simulation_and_Store
#############################################################
def pss(sess, dataset, model, cc, target_feature, num_of_neuron, rnn, iter):
    predict_dataset = predict(session=sess, dataset=dataset, model=model)
    simulation_result, simulation_stats, simulation_hist = simulation(dataset.test_dataset.series_data,
                                                                                   predict_dataset, target_feature)
    put_simulation_result_sql(cc=cc, target_feature=target_feature,
                                           num_of_neuron=num_of_neuron, rnn=rnn, iter=iter,
                                           stats=simulation_stats)

    return simulation_result, simulation_stats, simulation_hist

#############################################################
def predict(session, dataset, model):

    predict_dataset = pd.DataFrame([], columns=[model.TARGET_FEATURE])
    for idx in range(len(dataset.test_dataset) - model.SERIES_LENGTH):
        predict_time = dataset.test_dataset.series_data.index[idx + model.SERIES_LENGTH]
        input_dataset = dataset.test_dataset[idx : idx + model.SERIES_LENGTH].standardize(mean=dataset.train_mean,
                                                                                    std=dataset.train_std)
        predict_data_std = model.prediction.eval({model.x: input_dataset.as_array()}, session=session)
        predict_df_std = pd.DataFrame(predict_data_std, columns=[model.TARGET_FEATURE], index=[predict_time])
        predict_df = dataset.train_mean[model.TARGET_FEATURE] + dataset.train_std[model.TARGET_FEATURE] * predict_df_std
        predict_dataset = predict_dataset.append(predict_df)

    return predict_dataset

###########################################################
def simulation(correct, predict, target_feature):

    BIN_SIZE = 20
    target_company = target_feature[0:4]

    ret_column = ['buy', 'sell', 'gain_a', 'gain_r']
    ret = pd.DataFrame(index=[], columns=ret_column)
    feature_buy = target_feature
    for index, row in predict.iterrows():
        buy = int(row[feature_buy])
        price_high = correct.loc[index, target_company + '_high']
        price_low  = correct.loc[index, target_company + '_low']
        if (price_low <= buy and price_high >= buy):
            # 売買成立
            price_close = correct.loc[index][target_company + '_close']
            gain_a = price_close - buy
            gain_r = gain_a / buy * 100
            s = pd.Series([buy, price_close, gain_a, gain_r], index=ret_column)
        else:
            s = pd.Series()

        ret.loc[index] = s

    count_all = len(ret)                     # a. 全体
    count_buy = ret['buy'].count()           # b. 売買成立数(母数)
    mean_buy = ret['buy'].mean()             # c. 購入平均[円]
    mean_sell = ret['sell'].mean()           # i. 売却平均[円]
    mean_a = ret['gain_a'].mean()            # d. (売却-購入)の平均[円]
    mean_buy_ratio = mean_a / mean_buy       # e. d / 購入平均
    std_a  = ret['gain_a'].std()             # f. d の標準偏差
    mean_r = ret['gain_r'].mean()            # g. (売却 - 購入)/購入の平均[円]
    std_r  = ret['gain_r'].std()             # h. (売却 - 購入)/購入の標準偏差[円]
    if count_buy > 0:
        value_counts = ret['gain_a'].value_counts(bins=BIN_SIZE, sort=False, normalize=True, dropna=False)
    else:
        value_counts = np.array([np.nan] * BIN_SIZE)

    stats = pd.Series([count_all, count_buy, mean_buy, mean_sell, mean_a, mean_buy_ratio, std_a, mean_r, std_r],
                      index=['count_all', 'count_buy', 'mean_buy', 'mean_sell', 'mean_gain', 'mean_buy_ratio', 'std_gain', 'mean_gain_r', 'std_gain_r'])

    return ret, stats, value_counts

##########################################################################
# put Simulation Result into SQL db.
##########################################################################
def put_simulation_result_sql(cc, target_feature, num_of_neuron, rnn, stats, iter = -1):
    url = urlparse('mysql+pymysql://stockdb:bdkcots@192.168.1.11:3306/stockdb')  # for Ops
    # url = urlparse('mysql+pymysql://stock@localhost:3306/stockdb')  # for Dev

    stats['cc'] = target_feature[0:4]
    stats['target_feature'] = target_feature
    stats['companion'] = ','.join(cc)
    stats['num_of_neuron'] = num_of_neuron
    stats['training_iter'] = iter
    stats['rnn'] = rnn
    stats['datetime'] = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    # sql用にNaNをNoneに置き換える。
    stats = stats.where((pd.notnull(stats)), None)

    table_name = "simulation_stats"

    conn = mysql.connector.connect(
        host = url.hostname,
        port = url.port,
        user = url.username,
        database = url.path[1:],
        password = url.password
    )

    # すでにデータがあるかどうかを確認
    sql = 'SELECT COUNT(*) FROM %s WHERE cc="%s" ' \
          'AND target_feature="%s" ' \
          'AND companion="%s" ' \
          'AND num_of_neuron=%d ' \
          'AND training_iter=%d ' \
          'AND rnn="%s";' % (table_name, stats['cc'], stats['target_feature'],
                             stats['companion'], stats['num_of_neuron'],
                             stats['training_iter'], stats['rnn'])
    result_mysql = psql.execute(sql, conn)

    #
    if result_mysql.fetchone()[0] != 0:  # データがある場合は一旦削除
        sql = 'DELETE FROM %s WHERE cc="%s" AND target_feature="%s" AND companion="%s" AND num_of_neuron=%d AND ' \
              'training_iter=%d AND rnn="%s";' % (table_name, stats['cc'], stats['target_feature'], stats['companion'],
                                                  stats['num_of_neuron'], stats['training_iter'], stats['rnn'])
        psql.execute(sql, conn)

    # keys = ",".join(stats.index)
    keys = ",".join(['count_all', 'count_buy', 'mean_buy', 'mean_sell', 'mean_gain',
        'mean_buy_ratio', 'std_gain', 'mean_gain_r', 'std_gain_r', 'cc', 'target_feature',
        'companion', 'num_of_neuron', 'training_iter', 'rnn', 'datetime'])
    values = "%d,%d,%f,%f,%f,%f,%f,%f,%f,'%s','%s','%s',%d,%d,'%s','%s'" % (
        stats['count_all'], stats['count_buy'], stats['mean_buy'], stats['mean_sell'], stats['mean_gain'],
        stats['mean_buy_ratio'], stats['std_gain'], stats['mean_gain_r'], stats['std_gain_r'], stats['cc'], stats['target_feature'],
        stats['companion'], stats['num_of_neuron'], stats['training_iter'], stats['rnn'], stats['datetime'])
    sql = "INSERT INTO {} ({}) VALUES ({});".format(table_name, keys, values)
    psql.execute(sql, conn)
    conn.commit()
    conn.close()

###########################################################
#   main
###########################################################
RNN = ""
if __name__ == "__main__":

    # arg パーサの生成
    parser = argparse.ArgumentParser(description='予測値と真値の比較、保存、可視化')

    # オプション群の設定
    parser.add_argument('--cc', nargs='*', help='company code')
    parser.add_argument('--target_feature', help='6702_close', default='')
    parser.add_argument('--rnn', help='[BasicLSTMCell|BasicRNNCell]', default='BasicRNNCell')
    parser.add_argument('--num_of_neuron', help='60', default='60', type=int)
    parser.add_argument('--show_input', action='store_true', default=False)
    parser.add_argument('--do_simulation', action='store_true', default=False)

    args = parser.parse_args()  # 引数の解析を実行

    #############################################################
    # 学習データ、テストデータの読み込み、加工などの準備
    #############################################################
    # 入力データのうちトレーニングに使うデータの割合。残りは評価用。
    TRAIN_DATA_LENGTH_RATE = 0.9
    # 学習時間長
    SERIES_LENGTH = 72

    dataset = TrainTestDataSet.TrainTestDataSet(args.cc, train_data_length_rate=TRAIN_DATA_LENGTH_RATE,
                                                series_length=SERIES_LENGTH)
    print('train data {} to {}, {} data'.format(dataset.train_dataset.series_data.index[0],
                                                dataset.train_dataset.series_data.index[-1],
                                                dataset.train_dataset.series_length))
    print('test data {} to {}, {} data'.format(dataset.test_dataset.series_data.index[0],
                                               dataset.test_dataset.series_data.index[-1],
                                               dataset.test_dataset.series_length))

    #######################################################################
    # RNN Modelの生成
    #######################################################################
    if args.do_simulation == True:
        # 特徴量数
        FEATURE_COUNT = dataset.feature_count
        # ニューロン数
        NUM_OF_NEURON = args.num_of_neuron
        # 最適化対象パラメータ
        TARGET_FEATURE = args.target_feature
        TARGET_FEATURE_COUNT = 1
        # BasicRNNCell or BasicLSTMCell
        RNN = args.rnn  # rnn[0]

        #######################################################################
        model = Model.Model(dataset=dataset, series_length=SERIES_LENGTH, feature_count=FEATURE_COUNT,
                            num_of_neuron=NUM_OF_NEURON, rnn=RNN,
                            target_feature=TARGET_FEATURE, target_feature_count=TARGET_FEATURE_COUNT)
        ##
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        # restore
        directory_model = "./model/" + args.target_feature
        if os.path.exists(directory_model):
            ckpt = tf.train.get_checkpoint_state(directory_model)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('model diretory does NOT exists.: {}'.format(directory_model))
            exit(0)

        ########################################################################
        #  予測を実行
        ########################################################################
        predict_dataset = pd.DataFrame()
        predict_dataset = predict(session=sess, dataset=dataset, model=model)
        simulation_result, simulation_stats, simulation_hist = simulation(dataset.test_dataset.series_data,
                                                                        predict_dataset, TARGET_FEATURE)

    ########################################################################
    #  グラフ化
    ########################################################################
    correct_data = pd.concat([dataset.train_dataset.series_data,
                              dataset.test_dataset.series_data]).drop_duplicates()
    cf.go_offline()  # plotlyをオフラインで実行。

    if(args.do_simulation == True):
        predict_data = predict_dataset

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

    if(args.show_input == True):
        for feature in correct_data.columns:
            plot_data = pd.DataFrame({
                'correct': correct_data[feature],
            }).iplot(
              asFigure = True,
              title = feature
            )
            plot_data['layout']['paper_bgcolor'] = '#FFFFFF'
            plot_data.iplot()