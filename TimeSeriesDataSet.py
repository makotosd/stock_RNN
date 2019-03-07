#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

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


