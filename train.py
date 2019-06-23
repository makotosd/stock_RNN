#!/usr/bin/python
# -*- coding: Shift_JIS -*-

##
#  参考にしたのはここ
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
import show_predict

##########################################################################
# save model
##########################################################################
def save_model(sess, saver, n_iter, target_feature, output_log, z_columns):
    cwd = os.getcwd()
    directory_model = cwd + "/" + "model/" + "X".join(target_feature)
    os.makedirs(directory_model, exist_ok=True)
    saver.save(sess, directory_model + "/model.ckpt", global_step=n_iter)  ## for linux?

    output_log = output_log[z_columns]  # 順番を定義したとおりに並び替える。
    output_log.set_index('date', inplace=True)
    output_log.to_csv(directory_model + "/" + "training.csv")

##########################################################################
# train
##########################################################################
def train(cc='6702', target_feature=['6702_close'], rnn='BasicRNNCell',
          num_of_neuron=60, batch_size=32, num_train=2000):
    #############################################################
    # 学習データ、テストデータの読み込み、加工などの準備
    #############################################################
    # 入力データのうちトレーニングに使うデータの割合。残りは評価用。
    TRAIN_DATA_LENGTH_RATE = 0.9
    # 学習時間長
    SERIES_LENGTH = 72

    dataset2 = TrainTestDataSet.TrainTestDataSet(cc, train_data_length_rate=TRAIN_DATA_LENGTH_RATE,
                                                 series_length=SERIES_LENGTH)
    print('train data {} to {}, {} data'.format(dataset2.train_dataset.series_data.index[0],
                                                dataset2.train_dataset.series_data.index[-1],
                                                dataset2.train_dataset.series_length))
    print('test data {} to {}, {} data'.format(dataset2.test_dataset.series_data.index[0],
                                               dataset2.test_dataset.series_data.index[-1],
                                               dataset2.test_dataset.series_length))

    #######################################################################
    # RNN Modelの生成
    #######################################################################
    # 特徴量数
    FEATURE_COUNT = dataset2.feature_count
    # ニューロン数
    NUM_OF_NEURON = num_of_neuron
    # 最適化対象パラメータ
    TARGET_FEATURE = target_feature
    TARGET_FEATURE_COUNT = len(target_feature)
    # BasicRNNCell or BasicLSTMCell
    RNN = rnn  # rnn[0]

    #######################################################################
    model = Model.Model(dataset=dataset2, series_length=SERIES_LENGTH, feature_count=FEATURE_COUNT,
                        num_of_neuron=NUM_OF_NEURON, rnn=RNN,
                        target_feature=TARGET_FEATURE, target_feature_count=TARGET_FEATURE_COUNT)

    #######################################################################
    # 学習の実行
    #######################################################################
    # バッチサイズ
    BATCH_SIZE = batch_size  # 64

    # 学習回数
    NUM_TRAIN = num_train  # 10000

    # 学習中の出力頻度
    OUTPUT_BY = 50  # 500

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
    directory_log = 'logs/' + "X".join(target_feature)
    if os.path.exists(directory_log):
        shutil.rmtree(directory_log)
    os.makedirs(directory_log)

    output_log = pd.DataFrame()
    z_columns = ['date', 'Iteration', 'N_OF_NEURON', 'BATCH_SIZE', 'loss', 'accuracy', 'stddev']
    with tf.summary.FileWriter(directory_log, sess.graph) as writer:
        # 学習の実行
        saver = tf.train.Saver(max_to_keep=None)
        directory_model = "./model/" + "X".join(target_feature)

        # もしmodelが存在していたら、読み込んで続きを学習する。
        if os.path.exists(directory_model):
            ckpt = tf.train.get_checkpoint_state(directory_model)
            saver.restore(sess, ckpt.model_checkpoint_path)
            step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            if not step < NUM_TRAIN:
                print("There are same feature data. exit.")
                return 0
        else:
            sess.run(tf.initialize_all_variables())
            step = 0

        # test用データセットのバッチ実行用データの作成
        test_batch = dataset2.standardized_test_dataset.test_batch(SERIES_LENGTH, TARGET_FEATURE)

        # バッチ学習
        for i in range(step, NUM_TRAIN):
            # 学習データ作成
            batch = dataset2.standardized_train_dataset.next_batch(SERIES_LENGTH, BATCH_SIZE, TARGET_FEATURE)

            # ログ出力
            if i % OUTPUT_BY == 0:
                # ログの出力
                loss_train = sess.run(model.loss, feed_dict={model.x: batch[0], model.y: batch[1]})
                [summary, loss_test, acc2] = sess.run([merged_log, model.loss, model.acc_stddev],
                                                feed_dict={model.x: test_batch[0], model.y: test_batch[1]})
                # Predict, Simulation and Store
                show_predict.pss(sess=sess, dataset=dataset2, model=model,
                                 cc=cc, target_feature=target_feature, num_of_neuron=num_of_neuron,
                                 rnn=rnn, iter=i)

                now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                print('{:s}: step {:5d}, loss_train {:.3f}, loss_test {:.3f}, std {:.3f}'.format(now, i, loss_train, loss_test, acc2))
                output_log = output_log.append(pd.Series([now, i, NUM_OF_NEURON, BATCH_SIZE, loss_train, loss_test, acc2],
                                                         name=i, index=z_columns))
                writer.add_summary(summary, global_step=i)

                save_model(sess, saver, i, target_feature, output_log, z_columns)

            # 学習の実行
            _ = sess.run(model.optimizer, feed_dict={model.x: batch[0], model.y: batch[1]})

        # 最終ログ
        loss_train = sess.run(model.loss, feed_dict={model.x: batch[0], model.y: batch[1]})
        [loss_test, acc2] = sess.run([model.loss, model.acc_stddev],
                               feed_dict={model.x: test_batch[0], model.y: test_batch[1]})
        # Predict, Simulation and Store
        show_predict.pss(sess=sess, dataset=dataset2, model=model,
                                   cc=cc, target_feature=target_feature, num_of_neuron=num_of_neuron,
                                   rnn=rnn, iter=NUM_TRAIN)

        now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        print('{:s}: step {:5d}, loss_train {:.3f}, loss_test {:.3f}, std {:.3f}'.format(now, NUM_TRAIN, loss_train, loss_test, acc2))
        output_log = output_log.append(pd.Series([now, NUM_TRAIN, NUM_OF_NEURON, BATCH_SIZE, loss_train, loss_test, acc2],
                                                 name=NUM_TRAIN, index=z_columns))
        writer.add_summary(summary, global_step=NUM_TRAIN)

        # モデルの保存
        save_model(sess, saver, NUM_TRAIN, target_feature, output_log, z_columns)

    del model
    del dataset2

##########################################################################
# main
##########################################################################
if __name__ == "__main__":

    ##########################################################################
    # 起動オプションのパース
    ##########################################################################
    # arg パーサの生成
    parser = argparse.ArgumentParser(description='予測値と真値の比較、保存、可視化')

    # オプション群の設定
    parser.add_argument('--cc', nargs='*', help='company code')
    # parser.add_argument('--feature', nargs='*', help='[open, close, high, low, volume, highopen]',
    #                    default=['open', 'close', 'high', 'low', 'highopen'])
    # parser.add_argument('--quote', nargs='*', help='[USD, EUR]', default=[])
    parser.add_argument('--target_feature', nargs='*', help='6702_close', default='[]')
    parser.add_argument('--rnn', help='[BasicLSTMCell|BasicRNNCell]', default='BasicRNNCell')
    parser.add_argument('--num_of_neuron', help='60', default='60', type=int)
    parser.add_argument('--batch_size', help='60', default='32', type=int)
    parser.add_argument('--num_train', help='2000', default='2000', type=int)

    args = parser.parse_args()  # 引数の解析を実行

    print('cc: '      + ",".join(args.cc))
    # print('feature: ' + ",".join(args.feature))
    # print('quote: '   + ",".join(args.quote))
    print('rnn: '     + str(args.rnn))

    # 乱数シードの初期化（数値は何でもよい）
    np.random.seed(12345)

    train(cc=args.cc, target_feature=args.target_feature, rnn=args.rnn,
          batch_size=args.batch_size, num_of_neuron=args.num_of_neuron, num_train=args.num_train)

