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

#############################################################
def predict(dataset, model):

    predict_dataset = pd.DataFrame([], columns=[TARGET_FEATURE])
    for idx in range(len(dataset.test_dataset) - SERIES_LENGTH):
        predict_time = dataset.test_dataset.series_data.index[idx + SERIES_LENGTH]
        input_dataset = dataset.test_dataset[idx : idx + SERIES_LENGTH].standardize(mean=dataset.train_mean,
                                                                                    std=dataset.train_std)
        predict_data_std = model.prediction.eval({model.x: input_dataset.as_array()}, session=sess)
        predict_df_std = pd.DataFrame(predict_data_std, columns=[TARGET_FEATURE], index=[predict_time])
        predict_df = dataset.train_mean[TARGET_FEATURE] + dataset.train_std[TARGET_FEATURE] * predict_df_std
        predict_dataset = predict_dataset.append(predict_df)

    return predict_dataset

###########################################################
def simulation(correct, predict, target_company):

    ret_column = ['buy', 'sell', 'gain_a', 'gain_r']
    ret = pd.DataFrame(index=[], columns=ret_column)
    feature_buy = target_company + '_low'
    for index, row in predict.iterrows():
        buy = int(row[feature_buy])
        price_high = correct.loc[index, target_company + '_high']
        price_low  = correct.loc[index, target_company + '_low']
        if (price_low <= buy and price_high >= buy):
            # ��������
            price_close = correct.loc[index][target_company + '_close']
            gain_a = price_close - buy
            gain_r = gain_a / buy
            s = pd.Series([buy, price_close, gain_a, gain_r], index=ret_column)
            ret.loc[index] = s

    return ret

###########################################################
#   main
###########################################################
RNN = ""
if __name__ == "__main__":

    # arg �p�[�T�̐���
    parser = argparse.ArgumentParser(description='�\���l�Ɛ^�l�̔�r�A�ۑ��A����')

    # �I�v�V�����Q�̐ݒ�
    parser.add_argument('--cc', nargs='*', help='company code')
    parser.add_argument('--target_feature', help='6702_close', default='')
    parser.add_argument('--rnn', help='[BasicLSTMCell|BasicRNNCell]', default='BasicRNNCell')
    parser.add_argument('--num_of_neuron', help='60', default='60', type=int)
    parser.add_argument('--show_input', action='store_true', default=False)
    parser.add_argument('--do_simulation', action='store_true', default=False)

    args = parser.parse_args()  # �����̉�͂����s

    #############################################################
    # �w�K�f�[�^�A�e�X�g�f�[�^�̓ǂݍ��݁A���H�Ȃǂ̏���
    #############################################################
    # ���̓f�[�^�̂����g���[�j���O�Ɏg���f�[�^�̊����B�c��͕]���p�B
    TRAIN_DATA_LENGTH_RATE = 0.9
    # �w�K���Ԓ�
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
    # RNN Model�̐���
    #######################################################################
    if args.do_simulation == True:
        # �����ʐ�
        FEATURE_COUNT = dataset.feature_count
        # �j���[������
        NUM_OF_NEURON = args.num_of_neuron
        # �œK���Ώۃp�����[�^
        TARGET_FEATURE = args.target_feature
        TARGET_FEATURE_COUNT = 1
        # BasicRNNCell or BasicLSTMCell
        RNN = args.rnn  # rnn[0]
        # Company Code for Simulation
        CC_simulation = TARGET_FEATURE[0:4]

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
        #  �\�������s
        ########################################################################
        predict_dataset = pd.DataFrame()
        predict_dataset = predict(dataset, model)
        simulation_result = simulation(dataset.test_dataset.series_data, predict_dataset, CC_simulation)

    ########################################################################
    #  �O���t��
    ########################################################################
    correct_data = pd.concat([dataset.train_dataset.series_data,
                              dataset.test_dataset.series_data]).drop_duplicates()
    cf.go_offline()  # plotly���I�t���C���Ŏ��s�B

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