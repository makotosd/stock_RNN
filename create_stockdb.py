#!/usr/bin/python
# coding: utf-8
#
# csvファイルを全部mysqlに突っ込む。
# 予め、空っぽのdatabaseを作っておく。
#   url = 'mysql+pymysql://root@192.168.1.11:3306/stockdb'

import os
import pandas as pd
import sqlalchemy as sa

#url = 'mysql+pymysql://root@192.168.1.11:3306/stockdb'
#url = 'mysql+pymysql://root:murano2002@localhost:3306/stockdb'
url = 'mysql+pymysql://stock@localhost:3306/stockdb'

def read_stock_csv(csv_file):
    cc = csv_file[23:27]
    year = csv_file[31:35]
    data = pd.read_csv(csv_file)

    return cc, int(year), data

def create_mysqldb():

    dirname = "./stock_cc_year/"
    if not os.path.exists(dirname):
        exit(-1)

    files = os.listdir(dirname)
    #files = ['stocks_2046_1d_2015.csv']
    i = 0
    for file in files:
        print("###################### ", file, i, len(files))
        cc, year, data = read_stock_csv(csv_file=dirname+file)

        table_name = 'stocktable_%s' % (cc)
        engine = sa.create_engine(url, echo=True)
        try:
            data.to_sql(table_name, engine, index=False, if_exists='append')
        except sa.exc.InternalError as e:
            print("###################################### catch InternalError: ", e)
        else:
            pass  # 成功

        i = i+1

if __name__ == "__main__":
    create_mysqldb()
