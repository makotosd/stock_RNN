#!/usr/bin/python
# coding: utf-8

from urllib.parse import urlparse
import mysql.connector
import pandas.io.sql as psql
import pandas as pd
from time import sleep
import argparse
from datetime import datetime


# task class
class Task():
    def __init__(self, cc, target_feature, rnn, num_of_neuron, num_train):
        self.cc = cc
        self.target_feature = target_feature
        self.rnn = rnn
        self.num_of_neuron = num_of_neuron
        self.num_train = num_train

        self.table_name = 'queue_task'
        url = urlparse('mysql://stockdb:bdkcots@192.168.1.11:3306/stockdb') # for Ops
        # url = urlparse('mysql+pymysql://stock@localhost:3306/stockdb')  # for Dev

        self.conn = mysql.connector.connect(
            host = url.hostname,
            port = url.port,
            user = url.username,
            database = url.path[1:],
            password = url.password
        )

    # 同じタスクがないか？
    def is_queued(self):

        sql = 'SELECT COUNT(*) FROM %s ' \
              'WHERE cc="%s" ' \
              'AND target_feature="%s" ' \
              'AND num_of_neuron=%d ' \
              'AND num_train=%d ' \
              'AND rnn="%s";' % (self.table_name, self.cc, self.target_feature,
                                 self.num_of_neuron, self.num_train, self.rnn)
        result_mysql = psql.execute(sql, self.conn)

        return result_mysql.fetchone()[0] != 0

    # 同じタスクがあった場合の処理
    def ignore(self):
        pass

    # タスクをqueueにinsertする
    # waiting -> running -> done.
    # (skip)
    def insert_to_queue(self):
        now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        keys = ",".join(['cc', 'target_feature', 'num_of_neuron', 'num_train', 'rnn', 'status', 'queued_at'])
        values = '"%s","%s",%d, %d,"%s","%s", "%s"' % (
            ','.join(self.cc), self.target_feature, self.num_of_neuron, self.num_train, self.rnn, 'waiting', now
        )
        sql = "INSERT INTO {} ({}) VALUES ({});".format(self.table_name, keys, values)
        psql.execute(sql, self.conn)
        self.conn.commit()


    # close
    def close(self):
        self.conn.close()


    # テーブル作成(IF NOT EXISTS)
    def create_table(self):
        ############################################################
        ## table 作成
        ############################################################
        sql = "CREATE TABLE IF NOT EXISTS queue_task (" \
              "cc char(64) NOT NULL," \
              "target_feature char(64) NOT NULL," \
              "num_of_neuron int unsigned NOT NULL, " \
              "num_train int NOT NULL," \
              "rnn char(64) NOT NULL," \
              "status char(64) NOT NULL," \
              "queued_at datetime," \
              "dequeued_at datetime," \
              "complete_at datetime," \
              "PRIMARY KEY(cc, target_feature, num_of_neuron, rnn)" \
              ")"

        psql.execute(sql, self.conn)
        self.conn.commit()


#################################################################
def queue_task(cc, target_feature, rnn, num_of_neuron, num_train):


    ############################################################
    # task 作成
    ############################################################
    task = Task(cc, target_feature, rnn, num_of_neuron, num_train)

    ############################################################
    # table 作成(もしなければ)
    ############################################################
    task.create_table()

    ############################################################
    # task Entry
    ############################################################
    if task.is_queued():
        task.ignore()
    else:
        task.insert_to_queue()
    task.close()

############################################################
# main
############################################################
if __name__ == "__main__":
    ##########################################################################
    # 起動オプションのパース
    ##########################################################################
    # arg パーサの生成
    parser = argparse.ArgumentParser(description='予測値と真値の比較、保存、可視化')

    # オプション群の設定
    parser.add_argument('--cc', nargs='*', help='company code')
    # parser.add_argument('--feature', nargs='*', help='[open, close, high, low, volume, highopen]',
    #                    default=['open', 'close', 'high', 'low', 'highopen'])
    # parser.add_argument('--quote', nargs='*', help='[USD, EUR]', default=[])
    parser.add_argument('--target_feature', help='6702_close', default='')
    parser.add_argument('--rnn', help='[BasicLSTMCell|BasicRNNCell]', default='BasicRNNCell')
    parser.add_argument('--num_of_neuron', help='60', default='60', type=int)
    parser.add_argument('--batch_size', help='60', default='32', type=int)
    parser.add_argument('--num_train', help='2000', default='2000', type=int)

    args = parser.parse_args()  # 引数の解析を実行

    ##########################################################################
    # Queuing
    ##########################################################################
    queue_task(args.cc, args.target_feature, args.rnn, args.num_of_neuron, args.num_train)
