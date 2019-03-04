#!/usr/bin/python
# -*- coding: Shift_JIS -*-

########################################################################
#  list 1
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from urllib.request import urlretrieve
from urllib.parse import urlparse
from zipfile import ZipFile


########################################################################
def download_file(url, output_dir, overwrite=False):
    # URL����t�@�C�������擾
    parse_result = urlparse(url)
    file_name = os.path.basename(parse_result.path)
    # �o�͐�t�@�C���p�X
    destination = os.path.join(output_dir, file_name)


    # ���Ӗ��ȃ_�E�����[�h��h�����߁A�㏑���ioverwrite�̎w�肩���_�E�����[�h�̏ꍇ�̂݃_�E�����[�h�����{����
    if overwrite or not os.path.exists(destination):
        # �o�͐�f�B���N�g���̍쐬
        os.makedirs(output_dir)
        # �_�E�����[�h
        urlretrieve(url, destination)
    return destination

zip_file = download_file('https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip', './UCI_data/')


#######################################################################
#######################################################################
#######################################################################
# list 3

with ZipFile(zip_file) as z:
  with z.open('AirQualityUCI.xlsx') as f:
    air_quality = pd.read_excel(
      f,
      index_col=0, parse_dates={'DateTime': [0, 1]}, #1
      na_values=[-200.0],                            #2
      convert_float=False                            #3
    )


#############################################################
# list 7
# �s�v��̏���
target_columns = ['T', 'AH', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)']
air_quality = air_quality[target_columns]

#######################################################################################
# list 8 + list 13


# �����V�[�h�̏������i���l�͉��ł��悢�j
np.random.seed(12345)

# �N���X��`
class TimeSeriesDataSet:

    def __init__(self, dataframe):
        self.feature_count = len(dataframe.columns)
        self.series_length = len(dataframe)
        self.series_data = dataframe.astype('float32')

    def __getitem__(self, n):
        return TimeSeriesDataSet(self.series_data[n])

    def __len__(self):
        return len(self.series_data)

    @property
    def times(self):
        return self.series_data.index

    def next_batch(self, length, batch_size):
        """
        �A������length���Ԃ̃f�[�^�����1���Ԃ̌덷����p�f�[�^���擾����B
        �Ō��1���Ԃ͍ŏI�o�̓f�[�^�B
        """
        max_start_index = len(self) - length
        design_matrix = []
        expectation = []
        while len(design_matrix) < batch_size:
            start_index = np.random.choice(max_start_index)
            end_index = start_index + length + 1
            values = self.series_data[start_index:end_index]
            if (values.count() == length + 1).all():  # �؂�o�����f�[�^���Ɍ����l���Ȃ�
                train_data = values[:-1]
                true_value = values[-1:]
                design_matrix.append(train_data.as_matrix())
                expectation.append(np.reshape(true_value.as_matrix(), [self.feature_count]))

        return np.stack(design_matrix), np.stack(expectation)

    def append(self, data_point):
        dataframe = pd.DataFrame(data_point, columns=self.series_data.columns)
        self.series_data = self.series_data.append(dataframe)

    def tail(self, n):
        return TimeSeriesDataSet(self.series_data.tail(n))

    def as_array(self):
        return np.stack([self.series_data.as_matrix()])

    def mean(self):
        return self.series_data.mean()

    def std(self):
        return self.series_data.std()

    def standardize(self, mean=None, std=None):
        if mean is None:
            mean = self.mean()
        if std is None:
            std = self.std()
        return TimeSeriesDataSet((self.series_data - mean) / std)

########################################################################
# list 9


dataset = TimeSeriesDataSet(air_quality)
train_dataset = dataset[dataset.times.year < 2005]
test_dataset = dataset[dataset.times.year >= 2005]

########################################################################
# list 10
sess = tf.InteractiveSession()

# �p�����[�^�[
# �w�K���Ԓ�
SERIES_LENGTH = 72
# �����ʐ�
FEATURE_COUNT = dataset.feature_count

# ���́iplaceholder���\�b�h�̈����́A�f�[�^�^�A�e���\���̃T�C�Y�j
# �P���f�[�^
x = tf.placeholder(tf.float32, [None, SERIES_LENGTH, FEATURE_COUNT])
# ���t�f�[�^
y = tf.placeholder(tf.float32, [None, FEATURE_COUNT])

#######################################################################
# list 11
# RNN�Z���̍쐬
cell = tf.nn.rnn_cell.BasicRNNCell(20)
initial_state = cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
outputs, last_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)

#######################################################################
# list 12

# �S����
# �d��
w = tf.Variable(tf.zeros([20, FEATURE_COUNT]))
# �o�C�A�X
b = tf.Variable([0.1] * FEATURE_COUNT)
# �ŏI�o�́i�\���j
prediction = tf.matmul(last_state, w) + b

# �����֐��i���ϐ�Ό덷�FMAE�j�ƍœK���iAdam�j
loss = tf.reduce_mean(tf.map_fn(tf.abs, y - prediction))
optimizer = tf.train.AdamOptimizer().minimize(loss)

#######################################################################
# list 14
# �o�b�`�T�C�Y
BATCH_SIZE = 16

# �w�K��
NUM_TRAIN = 10000

# �w�K���̏o�͕p�x
OUTPUT_BY = 500

# �W����
train_mean = train_dataset.mean()
train_std = train_dataset.std()
standardized_train_dataset = train_dataset.standardize()

# �w�K�̎��s
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
for i in range(NUM_TRAIN):
    batch = standardized_train_dataset.next_batch(SERIES_LENGTH, BATCH_SIZE)
    mae, _ = sess.run([loss, optimizer], feed_dict={x: batch[0], y: batch[1]})
    if i % OUTPUT_BY == 0:
        print('step {:d}, error {:.2f}'.format(i, mae))

# �ۑ�
cwd = os.getcwd()
if os.name == 'nt':  # for windows
    saver.save(sess, cwd+"\\model.ckpt")  ## for windows?
else:
    saver.save(sess, cwd+"/model.ckpt")  ## for linux?

