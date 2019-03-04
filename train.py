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
    # URLからファイル名を取得
    parse_result = urlparse(url)
    file_name = os.path.basename(parse_result.path)
    # 出力先ファイルパス
    destination = os.path.join(output_dir, file_name)


    # 無意味なダウンロードを防ぐため、上書き（overwriteの指定か未ダウンロードの場合のみダウンロードを実施する
    if overwrite or not os.path.exists(destination):
        # 出力先ディレクトリの作成
        os.makedirs(output_dir)
        # ダウンロード
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
# 不要列の除去
target_columns = ['T', 'AH', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)']
air_quality = air_quality[target_columns]

#######################################################################################
# list 8 + list 13


# 乱数シードの初期化（数値は何でもよい）
np.random.seed(12345)

# クラス定義
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
        連続したlength時間のデータおよび1時間の誤差測定用データを取得する。
        最後の1時間は最終出力データ。
        """
        max_start_index = len(self) - length
        design_matrix = []
        expectation = []
        while len(design_matrix) < batch_size:
            start_index = np.random.choice(max_start_index)
            end_index = start_index + length + 1
            values = self.series_data[start_index:end_index]
            if (values.count() == length + 1).all():  # 切り出したデータ中に欠損値がない
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

