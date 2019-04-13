#!/usr/bin/python
# coding: utf-8
#
# csvファイルを全部mysqlに突っ込む。
# 予め、空っぽのdatabaseを作っておく。
#   url = 'mysql+pymysql://root@192.168.1.11:3306/stockdb'

import os
import pandas as pd
import sqlalchemy as sa

def read_stock_csv(csv_file):
    cc = csv_file[23:27]
    year = csv_file[31:35]
    data = pd.read_csv(csv_file)

    return cc, int(year), data

def create_mysqldb():

    dirname = "./stock_cc_year/"
    if not os.path.exists(dirname):
        exit(-1)

    for file in os.listdir(dirname):
        print(file)
        cc, year, data = read_stock_csv(csv_file=dirname+file)

        table_name = 'stocktable_%s' % (cc)
        url = 'mysql+pymysql://root@192.168.1.11:3306/stockdb'
        engine = sa.create_engine(url, echo=True)
        data.to_sql(table_name, engine, index=False, if_exists='append')

if __name__ == "__main__":
    create_mysqldb()