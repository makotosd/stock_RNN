#!/usr/bin/python
# coding: utf-8
#
# csvファイルを全部mysqlに突っ込む。
# 予め、空っぽのdatabaseを作っておく。
#   url = 'mysql+pymysql://root@192.168.1.11:3306/stockdb'

import os
import pandas as pd
import sqlalchemy as sa
from urllib.parse import urlparse
# import mysql.connector
# import pandas.io.sql as psql
import numpy as np

#url = 'mysql+pymysql://root@192.168.1.11:3306/stockdb'
#url = 'mysql+pymysql://root:murano2002@localhost:3306/stockdb'
url = 'mysql+pymysql://stock@localhost:3306/stockdb'

def read_stock_csv(csv_file):
    cc = csv_file[23:27]
    year = csv_file[31:35]
    data = pd.read_csv(csv_file)

    data['volume'] = data['volume'].astype(np.float)
    data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d %H:%M:%S')

    return cc, int(year), data

def create_mysqldb(url):
    parsed_url = urlparse(url)

    dirname = "./stock_cc_year/"
    if not os.path.exists(dirname):
        exit(-1)

    files = os.listdir(dirname)
    #files = ['stocks_2046_1d_2015.csv']
    i = 0
    cc_dict = {}
    for file in files:
        print("###################### ", file, i, len(files))
        cc, year, data = read_stock_csv(csv_file=dirname+file)

        table_name = 'stocktable_%s' % (cc)
        engine = sa.create_engine(url, echo=True)
        try:
            data.to_sql(table_name, engine, index=False, if_exists='append')
        except sa.exc.InternalError as e:
            print("###################################### catch InternalError: ", e)
        else: #　成功
            if cc not in cc_dict :
                sql = "ALTER TABLE %s ADD PRIMARY KEY(date);" % (table_name)
                try:
                    with engine.connect() as conn:
                        conn.execute(sql)
                except sa.exc.IntegrityError as e:
                    print("################## catch IntegrityError: ", e)
                else:
                    cc_dict[cc] = True

        i = i+1

if __name__ == "__main__":
    create_mysqldb(url)
