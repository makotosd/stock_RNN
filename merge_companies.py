#!/usr/bin/python
# -*- coding: utf8
# usage: python merge_companies.py code_a code_b ... code_z
#    ex) python merge_companies.py 1330 6701 6702

import sys
import os
import fnmatch
import pandas as pd

ccs = sys.argv
ccs.pop(0) ## 先頭(script名)を削除

dataset = pd.DataFrame()
for cc in ccs:
    print(cc)
    dir = "./stock_cc_year/"
    filename = 'stocks_%s_1d_*.csv' % (cc)

    ccdataset = pd.DataFrame()
    for file in os.listdir(dir):
        if fnmatch.fnmatch(file, filename):
            print(file)
            readdata = pd.read_csv(dir + file, index_col=0)
            if len(ccdataset) == 0:
                ccdataset = readdata
            else:
                ccdataset = pd.concat([ccdataset, readdata])

    ccdataset = ccdataset.sort_index()

    for i in ccdataset.columns:
        ccdataset.rename(columns={i: cc + "_" + i}, inplace=True)

    if(len(dataset) == 0):
        dataset = ccdataset
    else:
        dataset = pd.concat([dataset, ccdataset], axis=1, sort=False, join='inner')

print(dataset)