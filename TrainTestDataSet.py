#!/usr/bin/python
# -*- coding: utf-8

import merge_companies
import TimeSeriesDataSet

class TrainTestDataSet():
    def __init__(self, cc, train_data_length_rate=0.9, series_length=72):
        TRAIN_DATA_LENGTH_RATE = train_data_length_rate
        SERIES_LENGTH = series_length

        stock_merged_cc = merge_companies.merge_companies_mysql(cc)
        dataset = TimeSeriesDataSet.TimeSeriesDataSet(stock_merged_cc)
        self.feature_count = dataset.feature_count
        self.train_dataset, self.test_dataset = dataset.divide_dataset(rate=TRAIN_DATA_LENGTH_RATE,
                                                                       series_length=SERIES_LENGTH)

        # 標準化
        self.train_mean = self.train_dataset.mean()
        self.train_std = self.train_dataset.std()
        self.standardized_train_dataset = self.train_dataset.standardize()
        self.standardized_test_dataset = self.test_dataset.standardize(mean=self.train_mean, std=self.train_std)