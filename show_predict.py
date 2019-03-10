#!/usr/bin/python
# -*- coding: Shift_JIS -*-

#######################################################################
#  jupyter notebook上で動かす。真の値と、予測値をグラフ化します。
#######################################################################
# インポート＆実行済みの場合、以下の3行はなくてもよい
import pandas as pd
import cufflinks as cf
import argparse
import predict
import merge_companies

# arg パーサの生成
parser = argparse.ArgumentParser(description='予測値と真値の比較、保存、可視化')

# オプション群の設定
parser.add_argument('--cc', nargs='*', help='company code')
parser.add_argument('--output', help='予測値と真値の結果(csv)の出力。可視化は行わない。')
parser.add_argument('--input', help='予測値と真値の結果(csv)の入力。予測は行わない。')

args = parser.parse_args()  # 引数の解析を実行


# 真値の読み込み
input_dataset = merge_companies.merge_companies(args.cc)

#  予測を実行
if args.input is None:  # 読み込みファイルの指定がない　＝　予測の実施
    predict_dataset = predict.predict(input_dataset)

    if args.output is not None:  # 書き込みファイルの指定がある ＝ ファイルを書く
        predict_dataset.to_csv(str(args.output))

else:
    predict_dataset = pd.read_csv(args.input, index_col=0)


# 正解データと予測データ
correct_data = input_dataset[input_dataset.index >= '2007']
predict_data = predict_dataset

cf.go_offline()  # plotlyをオフラインで実行。
for feature in predict_dataset.columns:
    plot_data = pd.DataFrame({
        'correct': correct_data[feature],
        'predicted': predict_data[feature]
    }).iplot(
      asFigure = True,
      title = feature
    )
    plot_data['layout']['paper_bgcolor'] = '#FFFFFF'
    plot_data.iplot()
