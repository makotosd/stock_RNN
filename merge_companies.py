#!/usr/bin/python
# -*- coding: utf8
# usage: python merge_companies.py code_a code_b ... code_z
#    ex) python merge_companies.py 1330 6701 6702

import sys
import os
import fnmatch
import pandas as pd

## ドルとユーロを読む
def read_quate():
    dirname = "./stock_cc_year/"
    filename = 'quote.csv'

    quote = pd.read_csv(dirname + filename)
    quote['X'] = pd.to_datetime(quote['date'], format='%Y/%m/%d %H:%M:%S')
    quote = quote.set_index('X')
    quote = quote[['USD', 'EUR']]

    return quote

# 個社特殊事情への対応
def refine_by_company(dataset, cc):
    ret = dataset
    if cc =='6701':   ## NECは2017-09-27に株価が統合(10倍)された。
        retA = dataset[dataset.index < '2017-09-27'] * 10  # '2017-09-27'以前は、価格を10倍
        retA['volume'] = dataset['volume']                 # でも、volumeは元のまま
        retB = dataset[dataset.index >= '2017-09-27']
        ret = pd.concat([retA, retB])

    return ret

def merge_companies(ccs):

    dataset = pd.DataFrame()
    for cc in ccs:
        # print(cc)
        dirname = "./stock_cc_year/"
        filename = 'stocks_%s_1d_*.csv' % cc

        ccdataset = pd.DataFrame()
        for file in os.listdir(dirname):
            if fnmatch.fnmatch(file, filename):
                print(file)
                # csvのread
                readdata = pd.read_csv(dirname + file)

                # dateカラムを日付型のindexにする。
                readdata['X'] = pd.to_datetime(readdata['date'], format='%Y/%m/%d %H:%M:%S')
                readdata.set_index('X', inplace=True)
                readdata.drop(columns=['date'], inplace=True)

                # 個社特別対応
                readdata = refine_by_company(readdata, cc)

                # (high - open)^2のカラムの追加
                h_o = pd.DataFrame()
                h_o['highopen'] = (readdata['high'] - readdata['open'])**1
                readdata = pd.concat([readdata, h_o], axis=1)

                # 複数年データの結合
                if len(ccdataset) == 0:
                    ccdataset = readdata
                else:
                    ccdataset = pd.concat([ccdataset, readdata])

        ccdataset = ccdataset.sort_index()

        # カラム名にCCをつける
        for i in ccdataset.columns:
            ccdataset.rename(columns={i: cc + "_" + i}, inplace=True)

        # 複数企業データの結合
        if(len(dataset) == 0):
            dataset = ccdataset
        else:
            dataset = pd.concat([dataset, ccdataset], axis=1, sort=False, join='inner')

    # ドルとユーロの結合
    quote = read_quate()
    dataset = pd.concat([dataset, quote], axis=1, join='inner')

    return dataset


if __name__ == "__main__":
    ccs = sys.argv
    ccs.pop(0)  # 先頭(script名)を削除
    merge_companies(ccs)