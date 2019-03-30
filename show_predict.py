#!/usr/bin/python
# -*- coding: Shift_JIS -*-

#######################################################################
#  jupyter notebook��œ������B�^�̒l�ƁA�\���l���O���t�����܂��B
#######################################################################
import os
import pandas as pd
import cufflinks as cf
import argparse
import tensorflow as tf
import merge_companies
import TimeSeriesDataSet


def rnn_predict(input_dataset, current_time, train_mean, train_std, prediction, x, sess):
    # �W����
    previous = input_dataset.tail(SERIES_LENGTH).standardize(mean=train_mean, std=train_std)
    # �\���Ώۂ̎���
    predict_time = current_time

    # �\��
    batch_x = previous.as_array()
    predict_data = prediction.eval({x: batch_x}, session=sess)

    # ���ʂ̃f�[�^�t���[�����쐬
    df_standardized = pd.DataFrame(predict_data, columns=TARGET_FEATURE, index=[predict_time])
    # �W�����̋t����
    return train_mean[TARGET_FEATURE] + train_std[TARGET_FEATURE] * df_standardized


#############################################################
def predict(stock_merged_cc, company_codes, features, quote, target_feature):
    # �s�v��̏���
    # target_columns = quote
    # for cc in company_codes:
    #     for feature in features:
    #         cc_f = cc + '_' + feature
    #         target_columns.append(cc_f)
    # dataset = TimeSeriesDataSet.TimeSeriesDataSet(stock_merged_cc[target_columns])
    dataset = TimeSeriesDataSet.TimeSeriesDataSet(stock_merged_cc)
    train_dataset = dataset['2001': '2016']
    test_dataset = dataset['2017': ]

    # �p�����[�^�[
    # �w�K���Ԓ�
    global SERIES_LENGTH
    SERIES_LENGTH = 72
    # �����ʐ�
    global FEATURE_COUNT
    FEATURE_COUNT = dataset.feature_count
    # �j���[������
    global NUM_OF_NEURON
    NUM_OF_NEURON = 30
    # �œK���Ώ�
    global TARGET_FEATURE
    global TARGET_FEATURE_COUNT
    TARGET_FEATURE = target_feature
    TARGET_FEATURE_COUNT = len(TARGET_FEATURE)

    with tf.name_scope('input'):  # tensorboard�p
        # ���́iplaceholder���\�b�h�̈����́A�f�[�^�^�A�e���\���̃T�C�Y�j
        # �P���f�[�^
        x = tf.placeholder(tf.float32, [None, SERIES_LENGTH, FEATURE_COUNT])
        # ���t�f�[�^
        y = tf.placeholder(tf.float32, [None, TARGET_FEATURE_COUNT])

    # �W����
    train_mean = train_dataset.mean()
    train_std = train_dataset.std()

    #######################################################################
    # RNN�Z���̍쐬
    print(RNN)
    with tf.name_scope('RNN'):  # tensorboard�p
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

    with tf.name_scope('prediction'):  # tensorboard�p
        # �S����
        # �d��
        w = tf.Variable(tf.zeros([NUM_OF_NEURON, TARGET_FEATURE_COUNT]))
        # �o�C�A�X
        b = tf.Variable([0.1] * TARGET_FEATURE_COUNT)
        # �ŏI�o�́i�\���j
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

    # arg �p�[�T�̐���
    parser = argparse.ArgumentParser(description='�\���l�Ɛ^�l�̔�r�A�ۑ��A����')

    # �I�v�V�����Q�̐ݒ�
    parser.add_argument('--cc', nargs='*', help='company code')
    parser.add_argument('--output', help='�\���l�Ɛ^�l�̌���(csv)�̏o�́B�����͍s��Ȃ��B')
    parser.add_argument('--input', help='�\���l�Ɛ^�l�̌���(csv)�̓��́B�\���͍s��Ȃ��B')
    parser.add_argument('--feature', nargs='*', help='[open, close, high, low, volume]',
                        default=['open', 'close', 'high', 'low', 'volume'])
    parser.add_argument('--quote', nargs='*', help='[USD, EUR]', default=[])
    parser.add_argument('--target_feature', nargs='*', help='[6702_close, 6702_low], default=[]')
    parser.add_argument('--rnn', nargs=1, help='[BasicLSTMCell|BasicRNNCell]', default='BasicRNNCell')

    args = parser.parse_args()  # �����̉�͂����s

    # �O���[�o���ϐ�
    RNN = args.rnn[0]

    # �^�l�̓ǂݍ���
    input_dataset = merge_companies.merge_companies(args.cc)

    #  �\�������s
    if args.input is None:  # �ǂݍ��݃t�@�C���̎w�肪�Ȃ��@���@�\���̎��{
        predict_dataset = predict(input_dataset, args.cc, args.feature, args.quote, args.target_feature)

        if args.output is not None:  # �������݃t�@�C���̎w�肪���� �� �t�@�C��������
            predict_dataset.to_csv(str(args.output))

    else:
        predict_dataset = pd.read_csv(args.input, index_col=0)

    # �����f�[�^�Ɨ\���f�[�^
    correct_data = input_dataset[input_dataset.index >= '2001']
    predict_data = predict_dataset

    cf.go_offline()  # plotly���I�t���C���Ŏ��s�B
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
