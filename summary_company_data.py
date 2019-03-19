#!/usr/bin/python
# -*- coding: utf8
# usage: python merge_companies.py code_a code_b ... code_z
#    ex) python merge_companies.py 1330 6701 6702

import sys
import os
import fnmatch
import pandas as pd

# 個社特殊事情への対応
def refine_by_company(dataset, cc):
    ret = dataset
    if cc =='6701':   ## NECは2017-09-27に株価が統合(10倍)された。
        retA = dataset[dataset.index < '2017-09-27'] * 10  # '2017-09-27'以前は、価格を10倍
        retA['volume'] = dataset['volume']                 # でも、volumeは元のまま
        retB = dataset[dataset.index >= '2017-09-27']
        ret = pd.concat([retA, retB])

    return ret

def make_company_list(year):

    dirname = "./stock_cc_year/"
    filename = 'stocks_*_1d_%s.csv' % year

    ret = pd.DataFrame()
    for file in os.listdir(dirname):
        if fnmatch.fnmatch(file, filename):
            company_code = file[7:11]
            print(company_code)
            # csvのread
            readdata = pd.read_csv(dirname + file)

            # dateカラムを日付型のindexにする。
            readdata['X'] = pd.to_datetime(readdata['date'], format='%Y/%m/%d %H:%M:%S')
            readdata.set_index('X', inplace=True)
            readdata.drop(columns=['date'], inplace=True)

            # 個社特別対応
            readdata = refine_by_company(readdata, company_code)

            readdata['openhigh'] = (readdata['high']/readdata['open'] - 1)
            readdata['closelow'] = (readdata['close']/readdata['low'] - 1)

            zzz = readdata.std()
            mmm = pd.DataFrame({company_code: zzz})
            ret = ret.append(mmm.T)
            #ret.append(pd.DataFrame({company_code: readdata.std()}).T)

    return ret

import argparse
if __name__ == "__main__":

    # arg パーサの生成
    parser = argparse.ArgumentParser(description='個社データを読み、個社ごとの特徴表示')

    # オプション群の設定
    parser.add_argument('--year', type=int, help='year')

    args = parser.parse_args()  # 引数の解析を実行

    companies = make_company_list(args.year)
    companies.to_csv('company_std.csv')